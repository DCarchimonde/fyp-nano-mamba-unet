"""Run full-cine spatio-temporal analysis with the audited 3D checkpoint.

This script closes the temporal-analysis gap without pretending that the
historical Nano-Mamba bottleneck learned across cardiac time.  It:

1. loads every 3D frame of the complete ACDC 4D cine sequence;
2. applies the checkpoint-compatible spatial 3D segmentation network;
3. compares frame-wise predictions with fixed circular temporal probability
   fusion (0.25 previous + 0.50 current + 0.25 next);
4. derives LV/RV/MYO volume curves, ED/ES phase, EDV, ESV, SV, EF, LV-centroid
   displacement, and a global myocardial radial-distance surrogate; and
5. validates labelled ED/ES endpoints and metadata-defined phase indices on
   the existing held-out validation patients.

The output is segmentation-derived global motion/function analysis.  It is not
optical flow, dense deformable registration, or local myocardial strain.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cardiac_motion_metrics import (
    analyse_lv_curve,
    bootstrap_mean_ci,
    dice_by_class,
    finite_mean,
    functional_indices,
    parse_info_cfg_text,
    pearson_correlation,
    require_columns,
    segmentation_frame_metrics,
)


DEFAULT_PROJECT_ROOT = Path(r"D:\AI_FYP")
SPATIAL_SIZE = (256, 256, 16)
TEMPORAL_WEIGHTS = (0.25, 0.50, 0.25)
SUPPORTED_MODELS = (
    "UNet3D",
    "NanoMambaUNet",
    "Ablation_NoMamba_UNet",
    "Ablation_HalfMamba_UNet",
    "AttentionUNet",
    "SegResNet16",
)
METHODS = ("framewise", "temporal_fusion")
FRAME_FIELDS = (
    "patient_id",
    "group",
    "method",
    "frame",
    "num_frames",
    "normalized_cycle",
    "is_reference_ed",
    "is_reference_es",
    "rv_volume_ml",
    "myo_volume_ml",
    "lv_volume_ml",
    "lv_centroid_x_mm",
    "lv_centroid_y_mm",
    "lv_centroid_z_mm",
    "mean_myo_radius_mm",
)


def natural_sort_key(text: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def label_container_for(image_container: Path) -> Path:
    text = str(image_container)
    if text.endswith(".nii.gz"):
        return Path(text[:-7] + "_gt.nii.gz")
    if text.endswith(".nii"):
        return Path(text[:-4] + "_gt.nii")
    return image_container.with_name(image_container.name + "_gt")


def nifti_files(container: Path) -> List[Path]:
    candidates: List[Path] = []
    if container.is_dir():
        for pattern in ("*.nii", "*.nii.gz", "**/*.nii", "**/*.nii.gz"):
            candidates.extend(
                Path(match) for match in glob.glob(str(container / pattern), recursive=True)
            )
    elif is_nonempty_file(container):
        candidates.append(container)
    return sorted(
        {path for path in candidates if is_nonempty_file(path)},
        key=lambda path: natural_sort_key(str(path)),
    )


def first_endpoint_nifti(container: Path, want_label: bool) -> Path:
    accepted: List[Path] = []
    for path in nifti_files(container):
        name = path.name.lower()
        if "4d" in name:
            continue
        has_gt = "_gt" in name
        if want_label and not has_gt and not container.is_dir():
            continue
        if not want_label and has_gt:
            continue
        accepted.append(path)
    if not accepted:
        kind = "label" if want_label else "image"
        raise FileNotFoundError(f"No non-empty endpoint {kind} NIfTI in {container}")
    return accepted[0]


def discover_endpoint_pairs(patient_dir: Path) -> Dict[int, Dict[str, Path]]:
    containers = sorted(
        [
            Path(match)
            for match in glob.glob(str(patient_dir / "patient*_frame*.nii*"))
            if "_gt" not in Path(match).name.lower()
            and "4d" not in Path(match).name.lower()
        ],
        key=lambda path: natural_sort_key(path.name),
    )
    endpoints: Dict[int, Dict[str, Path]] = {}
    for image_container in containers:
        match = re.search(r"frame(\d+)", image_container.name, flags=re.IGNORECASE)
        if not match:
            continue
        frame = int(match.group(1))
        if frame in endpoints:
            raise ValueError(f"Duplicate endpoint frame {frame} in {patient_dir}")
        endpoints[frame] = {
            "image": first_endpoint_nifti(image_container, want_label=False),
            "label": first_endpoint_nifti(
                label_container_for(image_container), want_label=True
            ),
        }
    return endpoints


def patient_nifti_candidates(patient_dir: Path) -> List[Path]:
    """Return files that could be NIfTIs, including files inside .nii directories.

    Some locally unpacked ACDC copies represent an original NIfTI archive as a
    directory whose name ends in ``.nii`` and place the real NIfTI below it with
    a non-standard name. Candidate discovery must therefore inspect the full
    relative path rather than require a canonical top-level filename.
    """

    candidates: List[Path] = []
    for path in patient_dir.rglob("*"):
        if not is_nonempty_file(path):
            continue
        relative_parts = [part.lower() for part in path.relative_to(patient_dir).parts]
        if any(part.endswith((".nii", ".nii.gz")) for part in relative_parts):
            candidates.append(path)
    return sorted(set(candidates), key=lambda path: natural_sort_key(str(path)))


def _cine_candidate_diagnostic(
    patient_dir: Path,
    inspected: Sequence[Tuple[Path, Optional[Tuple[int, ...]], Optional[str]]],
) -> str:
    if not inspected:
        return "no NIfTI-like files were found recursively"
    details: List[str] = []
    for path, shape, error in inspected[:20]:
        relative = path.relative_to(patient_dir).as_posix()
        if shape is not None:
            details.append(f"{relative} -> shape={shape}")
        else:
            details.append(f"{relative} -> unreadable ({error})")
    if len(inspected) > 20:
        details.append(f"... {len(inspected) - 20} additional candidates omitted")
    return "; ".join(details)


def discover_cine_nifti(patient_dir: Path, expected_frames: int, nib: Any) -> Path:
    """Identify the cine by its NIfTI header, not by a fragile filename rule."""

    if expected_frames < 2:
        raise ValueError(f"Invalid expected cine frame count: {expected_frames}")
    inspected: List[Tuple[Path, Optional[Tuple[int, ...]], Optional[str]]] = []
    matches: List[Path] = []
    for path in patient_nifti_candidates(patient_dir):
        try:
            image = nib.load(str(path))
            shape = tuple(int(value) for value in image.shape)
        except Exception as exc:  # nibabel raises format-specific exception classes
            inspected.append((path, None, f"{type(exc).__name__}: {exc}"))
            continue
        inspected.append((path, shape, None))
        if len(shape) == 4 and shape[-1] == expected_frames:
            matches.append(path)

    if not matches:
        diagnostic = _cine_candidate_diagnostic(patient_dir, inspected)
        raise FileNotFoundError(
            f"No NIfTI with shape (X, Y, Z, {expected_frames}) was found recursively "
            f"in {patient_dir}. Inspected: {diagnostic}"
        )

    # A path carrying the original 4D marker is preferred only after its header
    # has proved that it is the requested cine. Never choose by file size.
    marked = [
        path
        for path in matches
        if any(
            "4d" in part.lower()
            for part in path.relative_to(patient_dir).parts
        )
    ]
    eligible = marked or matches
    if len(eligible) != 1:
        raise ValueError(
            f"Ambiguous full-cine NIfTIs for NbFrame={expected_frames} in "
            f"{patient_dir}: "
            + ", ".join(path.relative_to(patient_dir).as_posix() for path in eligible)
        )
    return eligible[0]


def preflight_patient_inputs(
    data_dir: Path,
    patient_ids: Sequence[str],
    nib: Any,
) -> Dict[str, Dict[str, object]]:
    """Resolve every patient's required input before the first model inference."""

    resolved: Dict[str, Dict[str, object]] = {}
    errors: List[str] = []
    for patient_id in patient_ids:
        try:
            patient_dir = data_dir / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"patient directory is missing: {patient_dir}")
            info_path = patient_dir / "Info.cfg"
            if not info_path.is_file():
                raise FileNotFoundError(f"Info.cfg is missing: {info_path}")
            info = parse_info_cfg_text(info_path.read_text(encoding="utf-8-sig"))
            endpoints = discover_endpoint_pairs(patient_dir)
            for frame in (int(info["ED"]), int(info["ES"])):
                if frame not in endpoints:
                    raise ValueError(
                        f"Info.cfg frame {frame} has no standalone image/label pair"
                    )
            cine_path = discover_cine_nifti(patient_dir, int(info["NbFrame"]), nib)
            resolved[patient_id] = {
                "patient_dir": patient_dir,
                "info_path": info_path,
                "info": info,
                "endpoints": endpoints,
                "cine_path": cine_path,
            }
        except Exception as exc:
            message = str(exc).replace("\n", "\n    ")
            errors.append(f"{patient_id}: {type(exc).__name__}: {message}")

    if errors:
        raise RuntimeError(
            "Full-cine dataset preflight failed before inference. Correct every listed "
            "patient and rerun; no result was generated:\n  - "
            + "\n  - ".join(errors)
        )
    return resolved


def load_split(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    train = list(payload.get("train_patients", []))
    validation = list(payload.get("val_patients", []))
    if len(train) != 80 or len(validation) != 20:
        raise ValueError("Expected the audited 80/20 patient split")
    if len(set(train)) != 80 or len(set(validation)) != 20 or set(train) & set(validation):
        raise ValueError("Patient split contains duplicates or leakage")
    return payload


def load_historical_per_case(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "dice_RV", "dice_MYO", "dice_LV", "mean_dice"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(sorted(missing))}"
            )
        for raw in reader:
            case_id = raw["case_id"].strip()
            if case_id in rows:
                raise ValueError(f"Duplicate historical case row: {case_id}")
            rows[case_id] = {
                key: float(raw[key])
                for key in ("dice_RV", "dice_MYO", "dice_LV", "mean_dice")
            }
    if len(rows) != 40:
        raise ValueError(f"Expected 40 historical validation cases, found {len(rows)}")
    return rows


def _load_rigorous_pipeline_module() -> Any:
    path = Path(__file__).with_name("21_rigorous_experiment_pipeline.py")
    spec = importlib.util.spec_from_file_location("rigorous_pipeline_for_cine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model definitions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    device: str,
    manifest_path: Optional[Path],
) -> Tuple[Any, Dict[str, object], Any, Any, Any]:
    try:
        import nibabel as nib
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("nibabel, PyTorch, and MONAI are required for cine inference") from exc

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        matches = [
            row for row in manifest.get("checkpoints", []) if row.get("model_name") == model_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Checkpoint manifest has no unique {model_name} row")
        observed_hash = sha256(checkpoint_path)
        if observed_hash != matches[0].get("sha256"):
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {model_name}: {observed_hash}"
            )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") != model_name:
        raise ValueError(
            f"Checkpoint model_name={checkpoint.get('model_name')!r}, expected {model_name!r}"
        )
    configured_size = tuple(checkpoint.get("config", {}).get("spatial_size", ()))
    if configured_size and configured_size != SPATIAL_SIZE:
        raise ValueError(
            f"Checkpoint spatial_size={configured_size}, expected {SPATIAL_SIZE}"
        )

    pipeline = _load_rigorous_pipeline_module()
    model = pipeline.build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    return model, checkpoint, nib, torch, functional


def unit_scale(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Image contains NaN or infinity")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        raise ValueError("Image frame has no intensity variation")
    return (values - minimum) / (maximum - minimum)


def normalized_input_mae(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError(f"Endpoint image shape mismatch: {first.shape} vs {second.shape}")
    return float(np.mean(np.abs(unit_scale(first) - unit_scale(second))))


def nifti_spatial_zooms(image: Any) -> Tuple[float, float, float]:
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if (
        len(zooms) != 3
        or not np.isfinite(zooms).all()
        or any(value <= 0.0 for value in zooms)
    ):
        raise ValueError(f"Invalid NIfTI spatial voxel sizes: {zooms}")
    return zooms


def _nifti_form_code(image: Any, field: str) -> Optional[int]:
    try:
        return int(np.asarray(image.header[field]).item())
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def spatial_grid_record(
    reference_image: Any,
    candidate_image: Any,
    reference_name: str,
    candidate_name: str,
) -> Dict[str, object]:
    """Prove voxel-grid compatibility while auditing unreliable raw affines.

    The public ACDC release can encode byte-identical 4D and standalone
    endpoint frames with different qform/sform affines. Array correspondence
    is therefore established separately from raw affine equality: spatial
    shape and header voxel sizes must match, and callers additionally compare
    endpoint image content. Raw affine equality remains recorded for audit.
    """

    reference_shape = tuple(int(value) for value in reference_image.shape[:3])
    candidate_shape = tuple(int(value) for value in candidate_image.shape)
    if len(candidate_shape) != 3 or candidate_shape != reference_shape:
        raise ValueError(
            f"{candidate_name} shape={candidate_shape} differs from "
            f"{reference_name} spatial shape={reference_shape}"
        )
    reference_zooms = nifti_spatial_zooms(reference_image)
    candidate_zooms = nifti_spatial_zooms(candidate_image)
    if not np.allclose(reference_zooms, candidate_zooms, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"{candidate_name} voxel sizes={candidate_zooms} differ from "
            f"{reference_name} voxel sizes={reference_zooms}"
        )
    reference_affine = np.asarray(reference_image.affine, dtype=np.float64)
    candidate_affine = np.asarray(candidate_image.affine, dtype=np.float64)
    for name, matrix in (
        (reference_name, reference_affine),
        (candidate_name, candidate_affine),
    ):
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} has an invalid affine matrix")
    return {
        "spatial_shape": list(reference_shape),
        "reference_zooms_mm": list(reference_zooms),
        "candidate_zooms_mm": list(candidate_zooms),
        "affine_matches": bool(
            np.allclose(reference_affine, candidate_affine, rtol=1e-5, atol=1e-5)
        ),
        "reference_qform_code": _nifti_form_code(reference_image, "qform_code"),
        "reference_sform_code": _nifti_form_code(reference_image, "sform_code"),
        "candidate_qform_code": _nifti_form_code(candidate_image, "qform_code"),
        "candidate_sform_code": _nifti_form_code(candidate_image, "sform_code"),
    }


def physical_metric_geometry(
    image: Any,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    """Build a millimetre metric from header zooms and affine directions.

    Only distances and volumes are reported. They are invariant to translation,
    rotation, and axis flips. Header zooms supply scale because some valid ACDC
    endpoint headers have sform matrices whose column norms disagree with
    ``pixdim``. The raw affine is retained only for direction when orthonormal.
    """

    zooms = np.asarray(nifti_spatial_zooms(image), dtype=np.float64)
    raw_affine = np.asarray(image.affine, dtype=np.float64)
    if raw_affine.shape != (4, 4) or not np.isfinite(raw_affine).all():
        raise ValueError("Cine has an invalid affine matrix")
    linear = raw_affine[:3, :3]
    column_norms = np.linalg.norm(linear, axis=0)
    directions_valid = bool(
        np.all(np.isfinite(column_norms)) and np.all(column_norms > 0.0)
    )
    if directions_valid:
        directions = linear / column_norms[None, :]
        directions_valid = bool(
            np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5)
        )
    metric_affine = np.eye(4, dtype=np.float64)
    if directions_valid:
        metric_affine[:3, :3] = directions @ np.diag(zooms)
        metric_affine[:3, 3] = raw_affine[:3, 3]
        policy = "header_zooms_with_orthonormal_affine_directions"
    else:
        metric_affine[:3, :3] = np.diag(zooms)
        policy = "header_zooms_with_diagonal_direction_fallback"
    voxel_volume_ml = float(np.prod(zooms) / 1000.0)
    return metric_affine, voxel_volume_ml, {
        "policy": policy,
        "spatial_zooms_mm": [float(value) for value in zooms],
        "raw_affine_axis_scales": [float(value) for value in column_norms],
        "raw_affine_scales_match_header_zooms": bool(
            np.allclose(column_norms, zooms, rtol=1e-5, atol=1e-5)
        ),
        "voxel_volume_ml": voxel_volume_ml,
        "qform_code": _nifti_form_code(image, "qform_code"),
        "sform_code": _nifti_form_code(image, "sform_code"),
    }


def preprocess_frame(frame: np.ndarray, torch: Any, functional: Any) -> Any:
    values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Each cine frame must be a finite 3D array")
    tensor = torch.from_numpy(np.ascontiguousarray(values))[None, None]
    resized = functional.interpolate(
        tensor, size=SPATIAL_SIZE, mode="trilinear", align_corners=False
    )
    minimum = resized.amin()
    maximum = resized.amax()
    if not torch.isfinite(minimum) or not torch.isfinite(maximum) or maximum <= minimum:
        raise ValueError("Resized cine frame has invalid intensity range")
    return (resized - minimum) / (maximum - minimum)


def resize_label(label: np.ndarray, torch: Any, functional: Any) -> np.ndarray:
    values = np.asarray(label)
    rounded = np.rint(values)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Endpoint label must be a finite 3D array")
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("Endpoint label contains non-integer values")
    if not set(np.unique(rounded.astype(np.int64))).issubset({0, 1, 2, 3}):
        raise ValueError("Endpoint label contains classes outside 0/1/2/3")
    tensor = torch.from_numpy(np.ascontiguousarray(rounded.astype(np.float32)))[None, None]
    return (
        functional.interpolate(tensor, size=SPATIAL_SIZE, mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .to(torch.uint8)
        .numpy()
    )


def resize_mask_to_native(
    mask: Any, native_shape: Sequence[int], torch: Any, functional: Any
) -> np.ndarray:
    tensor = mask.to(dtype=torch.float32)[None, None]
    return (
        functional.interpolate(tensor, size=tuple(native_shape), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _autocast_context(torch: Any, device: Any, enabled: bool) -> Any:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext

    return nullcontext()


def infer_probabilities(
    cine: np.ndarray,
    model: Any,
    torch: Any,
    functional: Any,
    device: Any,
    batch_size: int,
    amp: bool,
) -> Tuple[Any, Any]:
    probabilities = []
    framewise_masks = []
    for start in range(0, cine.shape[-1], batch_size):
        frames = [
            preprocess_frame(cine[..., index], torch, functional)
            for index in range(start, min(cine.shape[-1], start + batch_size))
        ]
        batch = torch.cat(frames, dim=0).to(device)
        with torch.inference_mode(), _autocast_context(torch, device, amp):
            logits = model(batch)
            probs = torch.softmax(logits.float(), dim=1)
        if not torch.isfinite(probs).all():
            raise ValueError("Model produced non-finite probabilities")
        framewise_masks.append(
            torch.argmax(logits.float(), dim=1).to(device="cpu", dtype=torch.uint8)
        )
        probabilities.append(probs.to(device="cpu", dtype=torch.float16))
    return torch.cat(probabilities, dim=0), torch.cat(framewise_masks, dim=0)


def prediction_masks(
    probabilities: Any,
    framewise_masks: Any,
    frame_index: int,
    torch: Any,
) -> Dict[str, Any]:
    num_frames = int(probabilities.shape[0])
    current = probabilities[frame_index].float()
    previous = probabilities[(frame_index - 1) % num_frames].float()
    following = probabilities[(frame_index + 1) % num_frames].float()
    fused = (
        TEMPORAL_WEIGHTS[0] * previous
        + TEMPORAL_WEIGHTS[1] * current
        + TEMPORAL_WEIGHTS[2] * following
    )
    return {
        "framewise": framewise_masks[frame_index],
        "temporal_fusion": torch.argmax(fused, dim=0).to(torch.uint8),
    }


def safe_float(value: object) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def sanitize_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if safe_float(row.get(field)) is None and isinstance(row.get(field), float) else row.get(field, "")
                    for field in fields
                }
            )


def endpoint_metrics(
    predicted_resized: np.ndarray,
    predicted_native: np.ndarray,
    reference_native: np.ndarray,
    torch: Any,
    functional: Any,
) -> Dict[str, float]:
    reference_resized = resize_label(reference_native, torch, functional)
    resized = dice_by_class(predicted_resized, reference_resized)
    native = dice_by_class(predicted_native, np.rint(reference_native).astype(np.uint8))
    return {
        **{f"resized_{key}": value for key, value in resized.items()},
        **{f"native_{key}": value for key, value in native.items()},
    }


def centroid_peak_displacement(rows: Sequence[Mapping[str, object]], ed_frame: int) -> float:
    ed_rows = [row for row in rows if int(row["frame"]) == ed_frame]
    if len(ed_rows) != 1:
        raise ValueError("Cannot identify unique reference-ED centroid")
    reference = np.array(
        [ed_rows[0][f"lv_centroid_{axis}_mm"] for axis in "xyz"], dtype=np.float64
    )
    displacements = []
    for row in rows:
        current = np.array(
            [row[f"lv_centroid_{axis}_mm"] for axis in "xyz"], dtype=np.float64
        )
        if np.isfinite(reference).all() and np.isfinite(current).all():
            displacements.append(float(np.linalg.norm(current - reference)))
    if not displacements:
        return float("nan")
    return float(max(displacements))


def create_patient_rows(
    patient_id: str,
    group: str,
    info: Mapping[str, object],
    frame_rows: Sequence[Mapping[str, object]],
    endpoint_dice: Mapping[str, Mapping[str, Mapping[str, float]]],
    reference_metrics: Mapping[str, Mapping[str, float]],
    input_mae: Mapping[str, float],
    historical_difference: Mapping[str, float],
) -> List[Dict[str, object]]:
    ed_frame = int(info["ED"])
    es_frame = int(info["ES"])
    reference_function = functional_indices(
        reference_metrics["ED"]["lv_volume_ml"],
        reference_metrics["ES"]["lv_volume_ml"],
    )
    patient_rows: List[Dict[str, object]] = []
    for method in METHODS:
        rows = [row for row in frame_rows if row["method"] == method]
        rows.sort(key=lambda row: int(row["frame"]))
        lv_curve = [float(row["lv_volume_ml"]) for row in rows]
        curve = analyse_lv_curve(lv_curve, ed_frame, es_frame, smoothing_window=3)
        annotated = functional_indices(
            lv_curve[ed_frame - 1],
            lv_curve[es_frame - 1],
            require_physiological_order=False,
        )
        ed_radius = float(rows[ed_frame - 1]["mean_myo_radius_mm"])
        es_radius = float(rows[es_frame - 1]["mean_myo_radius_mm"])
        row: Dict[str, object] = {
            "patient_id": patient_id,
            "group": group,
            "method": method,
            "num_frames": len(rows),
            "reference_ed_frame": ed_frame,
            "reference_es_frame": es_frame,
            "predicted_ed_frame": curve["predicted_ed_frame"],
            "predicted_es_frame": curve["predicted_es_frame"],
            "ed_frame_error": curve["ed_frame_error"],
            "es_frame_error": curve["es_frame_error"],
            "reference_edv_ml": reference_metrics["ED"]["lv_volume_ml"],
            "reference_esv_ml": reference_metrics["ES"]["lv_volume_ml"],
            "reference_sv_ml": reference_function["sv_ml"],
            "reference_ef_percent": reference_function["ef_percent"],
            "annotated_edv_ml": lv_curve[ed_frame - 1],
            "annotated_esv_ml": lv_curve[es_frame - 1],
            "annotated_sv_ml": annotated["sv_ml"],
            "annotated_ef_percent": annotated["ef_percent"],
            "annotated_phase_order_valid": int(
                lv_curve[ed_frame - 1] >= lv_curve[es_frame - 1]
            ),
            "annotated_ef_abs_error_pp": abs(
                annotated["ef_percent"] - reference_function["ef_percent"]
            ),
            "curve_edv_ml": curve["curve_edv_ml"],
            "curve_esv_ml": curve["curve_esv_ml"],
            "curve_sv_ml": curve["curve_sv_ml"],
            "curve_ef_percent": curve["curve_ef_percent"],
            "curve_ef_abs_error_pp": abs(
                float(curve["curve_ef_percent"]) - reference_function["ef_percent"]
            ),
            "normalized_second_difference": curve["normalized_second_difference"],
            "normalized_circular_total_variation": curve[
                "normalized_circular_total_variation"
            ],
            "peak_lv_centroid_displacement_mm": centroid_peak_displacement(rows, ed_frame),
            "myo_radius_ed_mm": ed_radius,
            "myo_radius_es_mm": es_radius,
            "myo_radius_ed_minus_es_mm": ed_radius - es_radius,
            "endpoint_input_mae_ed": input_mae["ED"],
            "endpoint_input_mae_es": input_mae["ES"],
            "historical_endpoint_max_abs_dice_difference": historical_difference[
                "maximum"
            ],
            "historical_ed_max_abs_dice_difference": historical_difference["ED"],
            "historical_es_max_abs_dice_difference": historical_difference["ES"],
        }
        for phase in ("ED", "ES"):
            for key, value in endpoint_dice[method][phase].items():
                row[f"{phase.lower()}_{key}"] = value
        patient_rows.append(row)
    return patient_rows


def aggregate_results(patient_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    required = (
        "patient_id",
        "method",
        "ed_frame_error",
        "es_frame_error",
        "reference_ef_percent",
        "annotated_ef_percent",
        "annotated_ef_abs_error_pp",
        "curve_ef_abs_error_pp",
        "ed_resized_mean_dice",
        "es_resized_mean_dice",
        "normalized_second_difference",
    )
    require_columns(patient_rows, required)
    result: Dict[str, object] = {}
    by_method: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in patient_rows:
        by_method[str(row["method"])].append(row)

    for method in METHODS:
        rows = by_method[method]
        if not rows:
            raise ValueError(f"No patient rows for {method}")
        reference_ef = [float(row["reference_ef_percent"]) for row in rows]
        annotated_ef = [float(row["annotated_ef_percent"]) for row in rows]
        endpoint_mean = [
            0.5 * (float(row["ed_resized_mean_dice"]) + float(row["es_resized_mean_dice"]))
            for row in rows
        ]
        ed_errors = [int(row["ed_frame_error"]) for row in rows]
        es_errors = [int(row["es_frame_error"]) for row in rows]
        result[method] = {
            "patients": len(rows),
            "mean_endpoint_resized_dice": finite_mean(endpoint_mean),
            "mean_ed_resized_dice": finite_mean(
                [float(row["ed_resized_mean_dice"]) for row in rows]
            ),
            "mean_es_resized_dice": finite_mean(
                [float(row["es_resized_mean_dice"]) for row in rows]
            ),
            "ed_phase_exact_fraction": float(np.mean(np.asarray(ed_errors) == 0)),
            "es_phase_exact_fraction": float(np.mean(np.asarray(es_errors) == 0)),
            "ed_phase_within_one_fraction": float(np.mean(np.asarray(ed_errors) <= 1)),
            "es_phase_within_one_fraction": float(np.mean(np.asarray(es_errors) <= 1)),
            "ed_phase_mae_frames": finite_mean(ed_errors),
            "es_phase_mae_frames": finite_mean(es_errors),
            "annotated_ef_mae_percentage_points": finite_mean(
                [float(row["annotated_ef_abs_error_pp"]) for row in rows]
            ),
            "curve_ef_mae_percentage_points": finite_mean(
                [float(row["curve_ef_abs_error_pp"]) for row in rows]
            ),
            "annotated_ef_pearson_r": pearson_correlation(reference_ef, annotated_ef),
            "mean_normalized_second_difference": finite_mean(
                [float(row["normalized_second_difference"]) for row in rows]
            ),
            "median_peak_lv_centroid_displacement_mm": float(
                np.nanmedian(
                    [float(row["peak_lv_centroid_displacement_mm"]) for row in rows]
                )
            ),
            "mean_myo_radius_ed_minus_es_mm": finite_mean(
                [float(row["myo_radius_ed_minus_es_mm"]) for row in rows]
            ),
            "patient_bootstrap_95_ci": {
                "endpoint_resized_dice": bootstrap_mean_ci(endpoint_mean),
                "annotated_ef_abs_error_pp": bootstrap_mean_ci(
                    [float(row["annotated_ef_abs_error_pp"]) for row in rows]
                ),
                "ed_phase_error_frames": bootstrap_mean_ci(ed_errors),
                "es_phase_error_frames": bootstrap_mean_ci(es_errors),
            },
        }

    framewise = result["framewise"]
    fused = result["temporal_fusion"]
    result["temporal_fusion_minus_framewise"] = {
        "endpoint_resized_dice": fused["mean_endpoint_resized_dice"]
        - framewise["mean_endpoint_resized_dice"],
        "annotated_ef_mae_percentage_points": fused[
            "annotated_ef_mae_percentage_points"
        ]
        - framewise["annotated_ef_mae_percentage_points"],
        "curve_smoothness": fused["mean_normalized_second_difference"]
        - framewise["mean_normalized_second_difference"],
    }
    framewise_by_patient = {
        str(row["patient_id"]): row for row in by_method["framewise"]
    }
    fused_by_patient = {
        str(row["patient_id"]): row for row in by_method["temporal_fusion"]
    }
    if set(framewise_by_patient) != set(fused_by_patient):
        raise ValueError("Frame-wise and fused methods do not cover the same patients")
    paired_endpoint_deltas = []
    paired_ef_mae_deltas = []
    paired_smoothness_deltas = []
    for patient_id in sorted(framewise_by_patient, key=natural_sort_key):
        raw = framewise_by_patient[patient_id]
        temporal = fused_by_patient[patient_id]
        raw_endpoint = 0.5 * (
            float(raw["ed_resized_mean_dice"]) + float(raw["es_resized_mean_dice"])
        )
        temporal_endpoint = 0.5 * (
            float(temporal["ed_resized_mean_dice"])
            + float(temporal["es_resized_mean_dice"])
        )
        paired_endpoint_deltas.append(temporal_endpoint - raw_endpoint)
        paired_ef_mae_deltas.append(
            float(temporal["annotated_ef_abs_error_pp"])
            - float(raw["annotated_ef_abs_error_pp"])
        )
        paired_smoothness_deltas.append(
            float(temporal["normalized_second_difference"])
            - float(raw["normalized_second_difference"])
        )
    result["temporal_fusion_minus_framewise"]["paired_patient_bootstrap_95_ci"] = {
        "endpoint_resized_dice": bootstrap_mean_ci(paired_endpoint_deltas),
        "annotated_ef_abs_error_pp": bootstrap_mean_ci(paired_ef_mae_deltas),
        "curve_smoothness": bootstrap_mean_ci(paired_smoothness_deltas),
    }
    return result


def _load_pyplot(output_dir: Path) -> Any:
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano_mamba_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_patient_curve(
    patient_id: str,
    info: Mapping[str, object],
    frame_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    plt = _load_pyplot(output_path.parent.parent)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    for method, color, label in (
        ("framewise", "#7F7F7F", "Frame-wise 3D"),
        ("temporal_fusion", "#C55A11", "Temporal probability fusion"),
    ):
        rows = sorted(
            [row for row in frame_rows if row["method"] == method],
            key=lambda row: int(row["frame"]),
        )
        axis.plot(
            [int(row["frame"]) for row in rows],
            [float(row["lv_volume_ml"]) for row in rows],
            marker="o" if method == "temporal_fusion" else None,
            markersize=2.8,
            linewidth=1.6,
            color=color,
            label=label,
        )
    axis.axvline(int(info["ED"]), color="#4472C4", linestyle="--", label="Reference ED")
    axis.axvline(int(info["ES"]), color="#70AD47", linestyle="--", label="Reference ES")
    axis.set_xlabel("Cardiac frame (one-based)")
    axis.set_ylabel("Predicted LV-cavity volume (mL)")
    axis.set_title(f"{patient_id}: full-cine LV volume trajectory")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_population_curve(
    frame_rows: Sequence[Mapping[str, object]],
    patient_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    plt = _load_pyplot(output_path.parent.parent)
    ed_lookup = {
        str(row["patient_id"]): int(row["reference_ed_frame"])
        for row in patient_rows
        if row["method"] == "temporal_fusion"
    }
    target = np.linspace(0.0, 1.0, 101)
    curves = []
    for patient_id, ed_frame in sorted(ed_lookup.items(), key=lambda item: natural_sort_key(item[0])):
        rows = sorted(
            [
                row
                for row in frame_rows
                if row["patient_id"] == patient_id and row["method"] == "temporal_fusion"
            ],
            key=lambda row: int(row["frame"]),
        )
        volumes = np.asarray([float(row["lv_volume_ml"]) for row in rows], dtype=np.float64)
        aligned = np.roll(volumes, -(ed_frame - 1))
        maximum = float(np.max(aligned))
        if maximum <= 0.0:
            raise ValueError(f"Non-positive LV curve for {patient_id}")
        normalized = aligned / maximum
        source = np.linspace(0.0, 1.0, len(normalized) + 1)
        periodic = np.append(normalized, normalized[0])
        curves.append(np.interp(target, source, periodic))
    matrix = np.asarray(curves)
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    axis.plot(target * 100.0, mean * 100.0, color="#C55A11", linewidth=2.2)
    axis.fill_between(
        target * 100.0,
        np.maximum(0.0, mean - sd) * 100.0,
        (mean + sd) * 100.0,
        color="#F4B183",
        alpha=0.35,
        label="Mean ± SD",
    )
    axis.set_xlabel("Normalized cardiac cycle from reference ED (%)")
    axis.set_ylabel("LV volume relative to patient maximum (%)")
    axis.set_title("Validation-cohort full-cine LV motion trajectory")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_ef_scatter(patient_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    plt = _load_pyplot(output_path.parent.parent)
    figure, axis = plt.subplots(figsize=(6.4, 5.6), dpi=180)
    all_values = []
    for method, color, marker, label in (
        ("framewise", "#7F7F7F", "o", "Frame-wise 3D"),
        ("temporal_fusion", "#C55A11", "s", "Temporal probability fusion"),
    ):
        rows = [row for row in patient_rows if row["method"] == method]
        reference = [float(row["reference_ef_percent"]) for row in rows]
        predicted = [float(row["annotated_ef_percent"]) for row in rows]
        all_values.extend(reference + predicted)
        axis.scatter(reference, predicted, color=color, marker=marker, alpha=0.82, label=label)
    low = math.floor(min(all_values) / 5.0) * 5.0
    high = math.ceil(max(all_values) / 5.0) * 5.0
    axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Reference EF from labelled ED/ES masks (%)")
    axis.set_ylabel("Predicted EF at reference ED/ES (%)")
    axis.set_title("Segmentation-derived ejection fraction")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_endpoint_overlay(
    patient_id: str,
    phase_arrays: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
) -> None:
    plt = _load_pyplot(output_path.parent.parent)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), dpi=170)
    for row_index, phase in enumerate(("ED", "ES")):
        image = phase_arrays[phase]["image"]
        label = phase_arrays[phase]["label"]
        foreground_per_slice = np.sum(label > 0, axis=(0, 1))
        slice_index = int(np.argmax(foreground_per_slice))
        for col_index, method in enumerate(("framewise", "temporal_fusion")):
            axis = axes[row_index, col_index]
            prediction = phase_arrays[phase][method]
            axis.imshow(image[:, :, slice_index].T, cmap="gray", origin="lower")
            axis.contour(
                label[:, :, slice_index].T,
                levels=[0.5, 1.5, 2.5, 3.5],
                colors="#70AD47",
                linewidths=0.8,
                linestyles="--",
            )
            axis.contour(
                prediction[:, :, slice_index].T,
                levels=[0.5, 1.5, 2.5, 3.5],
                colors="#C55A11",
                linewidths=0.8,
            )
            title = "Frame-wise 3D" if method == "framewise" else "Temporal fusion"
            axis.set_title(f"{phase}: {title}")
            axis.axis("off")
    figure.suptitle(
        f"{patient_id} endpoint audit — green dashed: reference; orange: prediction",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def patient_metric_fields(rows: Sequence[Mapping[str, object]]) -> List[str]:
    preferred = [
        "patient_id",
        "group",
        "method",
        "num_frames",
        "reference_ed_frame",
        "reference_es_frame",
        "predicted_ed_frame",
        "predicted_es_frame",
        "ed_frame_error",
        "es_frame_error",
        "reference_edv_ml",
        "reference_esv_ml",
        "reference_sv_ml",
        "reference_ef_percent",
        "annotated_edv_ml",
        "annotated_esv_ml",
        "annotated_sv_ml",
        "annotated_ef_percent",
        "annotated_ef_abs_error_pp",
        "curve_edv_ml",
        "curve_esv_ml",
        "curve_sv_ml",
        "curve_ef_percent",
        "curve_ef_abs_error_pp",
        "normalized_second_difference",
        "normalized_circular_total_variation",
        "peak_lv_centroid_displacement_mm",
        "myo_radius_ed_mm",
        "myo_radius_es_mm",
        "myo_radius_ed_minus_es_mm",
        "endpoint_input_mae_ed",
        "endpoint_input_mae_es",
        "historical_endpoint_max_abs_dice_difference",
        "historical_ed_max_abs_dice_difference",
        "historical_es_max_abs_dice_difference",
    ]
    observed = {key for row in rows for key in row}
    return preferred + sorted(observed.difference(preferred))


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    framewise = summary["methods"]["framewise"]
    fused = summary["methods"]["temporal_fusion"]
    delta = summary["methods"]["temporal_fusion_minus_framewise"]
    lines = [
        "# Full-Cine Spatio-Temporal Analysis Report",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Patients: {summary['patients']} held-out validation patients.",
        f"Frames: {summary['frames']} complete cine frames.",
        "",
        "## What was implemented",
        "",
        "The audited 3D checkpoint was applied to every cine frame. A fixed circular ",
        "probability fusion used weights 0.25/0.50/0.25 for previous/current/next ",
        "frames. Full-cycle segmentation trajectories were converted into chamber ",
        "volume curves, ED/ES phase estimates, EDV, ESV, SV, EF, LV-centroid ",
        "displacement, and a global myocardial radial-distance surrogate.",
        "",
        "## Main validation metrics",
        "",
        "| Metric | Frame-wise 3D | Temporal fusion |",
        "|---|---:|---:|",
        f"| Endpoint resized-grid mean Dice | {framewise['mean_endpoint_resized_dice'] * 100:.2f}% | {fused['mean_endpoint_resized_dice'] * 100:.2f}% |",
        f"| ED phase exact | {framewise['ed_phase_exact_fraction'] * 100:.1f}% | {fused['ed_phase_exact_fraction'] * 100:.1f}% |",
        f"| ES phase exact | {framewise['es_phase_exact_fraction'] * 100:.1f}% | {fused['es_phase_exact_fraction'] * 100:.1f}% |",
        f"| ED phase within ±1 frame | {framewise['ed_phase_within_one_fraction'] * 100:.1f}% | {fused['ed_phase_within_one_fraction'] * 100:.1f}% |",
        f"| ES phase within ±1 frame | {framewise['es_phase_within_one_fraction'] * 100:.1f}% | {fused['es_phase_within_one_fraction'] * 100:.1f}% |",
        f"| EF MAE at reference ED/ES | {framewise['annotated_ef_mae_percentage_points']:.2f} pp | {fused['annotated_ef_mae_percentage_points']:.2f} pp |",
        f"| EF Pearson r | {framewise['annotated_ef_pearson_r']:.3f} | {fused['annotated_ef_pearson_r']:.3f} |",
        f"| Normalized curve second difference | {framewise['mean_normalized_second_difference']:.4f} | {fused['mean_normalized_second_difference']:.4f} |",
        "",
        "## Temporal-fusion change",
        "",
        f"- Endpoint Dice change: {delta['endpoint_resized_dice'] * 100:+.3f} percentage points.",
        f"- EF MAE change: {delta['annotated_ef_mae_percentage_points']:+.3f} percentage points (negative is better).",
        f"- Curve smoothness change: {delta['curve_smoothness']:+.5f} (negative is smoother).",
        "",
        "## Scientific boundary",
        "",
        "This is a real full-cine temporal analysis, but it remains segmentation-derived ",
        "global motion/function analysis. The 3D backbone was not retrained as a learned ",
        "temporal network, and the outputs are not optical flow, dense displacement, ",
        "local strain, or externally validated clinical measurements.",
        "Raw endpoint/4D affine equality is audited but is not used as the grid ",
        "registration criterion because valid ACDC files can encode identical endpoint ",
        "voxel arrays with different qform/sform affines. Shape, voxel size, and ED/ES ",
        "image content must match; physical magnitudes use cine header voxel sizes.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--historical-per-case", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="NanoMambaUNet")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--example-patient", default=None)
    parser.add_argument(
        "--endpoint-input-mae-tolerance",
        type=float,
        default=0.01,
        help="Fail when a 4D endpoint frame does not match its standalone 3D image",
    )
    parser.add_argument(
        "--historical-dice-tolerance",
        type=float,
        default=0.005,
        help="Maximum allowed endpoint Dice difference from the audited case table",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        help="Smoke-test subset; omitting this runs all 20 validation patients",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    project_root = args.project_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (args.data_dir or project_root / "Data" / "ACDC" / "database" / "training").resolve()
    checkpoint = (
        args.checkpoint
        or project_root
        / "experiment_outputs"
        / "rigorous_patient_split"
        / "checkpoints"
        / f"best_{args.model}.pth"
    ).resolve()
    split_path = (args.split or repo_root / "evidence" / "rigorous_patient_split" / "patient_split_seed42.json").resolve()
    manifest_path = (
        args.checkpoint_manifest
        or repo_root / "evidence" / "rigorous_patient_split" / "checkpoint_manifest.json"
    ).resolve()
    historical_per_case_path = (
        args.historical_per_case
        or repo_root
        / "evidence"
        / "rigorous_patient_split"
        / f"per_case_{args.model}.csv"
    ).resolve()
    output_dir = args.output_dir.resolve()

    for path, label in (
        (data_dir, "data directory"),
        (checkpoint, "checkpoint"),
        (split_path, "patient split"),
        (manifest_path, "checkpoint manifest"),
        (historical_per_case_path, "historical per-case metrics"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    split = load_split(split_path)
    historical_per_case = load_historical_per_case(historical_per_case_path)
    validation_patients = list(split["val_patients"])
    selected_patients = args.patients or validation_patients
    unknown = sorted(set(selected_patients).difference(validation_patients))
    if unknown:
        raise ValueError(
            "Temporal analysis is restricted to audited validation patients; unknown: "
            + ", ".join(unknown)
        )
    if len(selected_patients) != len(set(selected_patients)):
        raise ValueError("Duplicate patient IDs were requested")
    selected_patients = sorted(selected_patients, key=natural_sort_key)
    full_run = selected_patients == sorted(validation_patients, key=natural_sort_key)
    example_patient = args.example_patient or selected_patients[0]
    if example_patient not in selected_patients:
        raise ValueError("--example-patient must be in the selected validation patients")

    import torch

    device_text = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    model, checkpoint_payload, nib, torch, functional = load_checkpoint(
        checkpoint, args.model, device_text, manifest_path
    )
    device = torch.device(device_text)

    patient_inputs = preflight_patient_inputs(data_dir, selected_patients, nib)
    print(
        f"Full-cine dataset preflight: PASS ({len(patient_inputs)} patients; "
        "NIfTI headers match each Info.cfg NbFrame)"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()

    all_frame_rows: List[Dict[str, object]] = []
    all_patient_rows: List[Dict[str, object]] = []
    input_records: List[Dict[str, object]] = []
    total_start = time.perf_counter()

    for patient_index, patient_id in enumerate(selected_patients, start=1):
        patient_start = time.perf_counter()
        inputs = patient_inputs[patient_id]
        patient_dir = inputs["patient_dir"]
        info_path = inputs["info_path"]
        info = inputs["info"]
        endpoints = inputs["endpoints"]
        cine_path = inputs["cine_path"]
        if not isinstance(patient_dir, Path) or not isinstance(info_path, Path):
            raise TypeError("Internal preflight path record is invalid")
        if not isinstance(info, dict) or not isinstance(endpoints, dict):
            raise TypeError("Internal preflight metadata record is invalid")
        if not isinstance(cine_path, Path):
            raise TypeError("Internal preflight cine record is invalid")
        cine_image = nib.load(str(cine_path))
        if len(cine_image.shape) != 4:
            raise ValueError(f"Expected a 4D NIfTI for {patient_id}, got {cine_image.shape}")
        if cine_image.shape[-1] != int(info["NbFrame"]):
            raise ValueError(
                f"{patient_id} NIfTI time dimension {cine_image.shape[-1]} does not match "
                f"Info.cfg NbFrame={info['NbFrame']}"
            )
        cine = np.asarray(cine_image.dataobj, dtype=np.float32)
        if not np.isfinite(cine).all():
            raise ValueError(f"{patient_id} full cine contains NaN or infinity")
        native_shape = cine.shape[:3]
        metric_affine, voxel_volume_ml, cine_geometry = physical_metric_geometry(
            cine_image
        )
        if not np.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0.0:
            raise ValueError(f"{patient_id} has invalid physical voxel volume")

        input_mae: Dict[str, float] = {}
        endpoint_geometry: Dict[str, Dict[str, object]] = {}
        labels_native: Dict[str, np.ndarray] = {}
        endpoint_images: Dict[str, np.ndarray] = {}
        for phase, frame in (("ED", int(info["ED"])), ("ES", int(info["ES"]))):
            endpoint_image = nib.load(str(endpoints[frame]["image"]))
            endpoint_label = nib.load(str(endpoints[frame]["label"]))
            label_pair_geometry = spatial_grid_record(
                endpoint_image,
                endpoint_label,
                f"{patient_id} {phase} endpoint image",
                f"{patient_id} {phase} endpoint label",
            )
            if not bool(label_pair_geometry["affine_matches"]):
                raise ValueError(
                    f"{patient_id} {phase} endpoint image/label affine differs; "
                    "their label registration cannot be established"
                )
            endpoint_geometry[phase] = {
                "image_vs_cine": spatial_grid_record(
                    cine_image,
                    endpoint_image,
                    f"{patient_id} cine",
                    f"{patient_id} {phase} endpoint image",
                ),
                "label_vs_cine": spatial_grid_record(
                    cine_image,
                    endpoint_label,
                    f"{patient_id} cine",
                    f"{patient_id} {phase} endpoint label",
                ),
                "label_vs_endpoint_image": label_pair_geometry,
            }
            image_values = np.asarray(endpoint_image.dataobj, dtype=np.float32)
            label_values = np.asarray(endpoint_label.dataobj)
            input_mae[phase] = normalized_input_mae(cine[..., frame - 1], image_values)
            if input_mae[phase] > args.endpoint_input_mae_tolerance:
                raise ValueError(
                    f"{patient_id} {phase} 4D-to-3D normalized MAE={input_mae[phase]:.6f} "
                    f"exceeds {args.endpoint_input_mae_tolerance}; refusing a misregistered cine"
                )
            labels_native[phase] = np.rint(label_values).astype(np.uint8)
            endpoint_images[phase] = image_values

        probabilities, framewise_masks = infer_probabilities(
            cine, model, torch, functional, device, args.batch_size, args.amp
        )
        if tuple(probabilities.shape) != (
            cine.shape[-1],
            4,
            *SPATIAL_SIZE,
        ):
            raise RuntimeError(f"Unexpected probability tensor: {tuple(probabilities.shape)}")
        if tuple(framewise_masks.shape) != (
            cine.shape[-1],
            *SPATIAL_SIZE,
        ):
            raise RuntimeError(
                f"Unexpected frame-wise mask tensor: {tuple(framewise_masks.shape)}"
            )

        frame_rows: List[Dict[str, object]] = []
        endpoint_masks_resized: Dict[str, Dict[str, np.ndarray]] = {
            method: {} for method in METHODS
        }
        endpoint_masks_native: Dict[str, Dict[str, np.ndarray]] = {
            method: {} for method in METHODS
        }
        for frame_index in range(cine.shape[-1]):
            masks = prediction_masks(
                probabilities, framewise_masks, frame_index, torch
            )
            frame_number = frame_index + 1
            for method, resized_mask_tensor in masks.items():
                native_mask = resize_mask_to_native(
                    resized_mask_tensor, native_shape, torch, functional
                )
                physical = segmentation_frame_metrics(
                    native_mask, metric_affine, voxel_volume_ml
                )
                row: Dict[str, object] = {
                    "patient_id": patient_id,
                    "group": str(info.get("Group", "unknown")),
                    "method": method,
                    "frame": frame_number,
                    "num_frames": cine.shape[-1],
                    "normalized_cycle": frame_index / float(cine.shape[-1]),
                    "is_reference_ed": int(frame_number == int(info["ED"])),
                    "is_reference_es": int(frame_number == int(info["ES"])),
                    **physical,
                }
                frame_rows.append(row)
                if frame_number == int(info["ED"]):
                    endpoint_masks_resized[method]["ED"] = resized_mask_tensor.numpy()
                    endpoint_masks_native[method]["ED"] = native_mask
                if frame_number == int(info["ES"]):
                    endpoint_masks_resized[method]["ES"] = resized_mask_tensor.numpy()
                    endpoint_masks_native[method]["ES"] = native_mask

        endpoint_dice: Dict[str, Dict[str, Dict[str, float]]] = {
            method: {} for method in METHODS
        }
        reference_metrics = {
            phase: segmentation_frame_metrics(label, metric_affine, voxel_volume_ml)
            for phase, label in labels_native.items()
        }
        for method in METHODS:
            for phase in ("ED", "ES"):
                endpoint_dice[method][phase] = endpoint_metrics(
                    endpoint_masks_resized[method][phase],
                    endpoint_masks_native[method][phase],
                    labels_native[phase],
                    torch,
                    functional,
                )

        historical_difference: Dict[str, float] = {}
        for phase, frame in (("ED", int(info["ED"])), ("ES", int(info["ES"]))):
            case_id = f"{patient_id}_frame{frame:02d}"
            if case_id not in historical_per_case:
                raise ValueError(f"Historical case table is missing {case_id}")
            expected = historical_per_case[case_id]
            observed = endpoint_dice["framewise"][phase]
            differences = [
                abs(float(observed[f"resized_{field}"]) - float(expected[field]))
                for field in ("dice_RV", "dice_MYO", "dice_LV", "mean_dice")
            ]
            historical_difference[phase] = max(differences)
            if historical_difference[phase] > args.historical_dice_tolerance:
                raise ValueError(
                    f"{case_id} full-cine endpoint differs from the audited case table by "
                    f"{historical_difference[phase]:.6f}, exceeding "
                    f"{args.historical_dice_tolerance}; refusing a mismatched pipeline"
                )
        historical_difference["maximum"] = max(
            historical_difference["ED"], historical_difference["ES"]
        )

        patient_rows = create_patient_rows(
            patient_id,
            str(info.get("Group", "unknown")),
            info,
            frame_rows,
            endpoint_dice,
            reference_metrics,
            input_mae,
            historical_difference,
        )
        all_frame_rows.extend(frame_rows)
        all_patient_rows.extend(patient_rows)
        plot_patient_curve(
            patient_id, info, frame_rows, figures_dir / f"{patient_id}_lv_curve.png"
        )

        if patient_id == example_patient:
            phase_arrays: Dict[str, Dict[str, np.ndarray]] = {}
            for phase in ("ED", "ES"):
                phase_arrays[phase] = {
                    "image": endpoint_images[phase],
                    "label": labels_native[phase],
                    **{
                        method: endpoint_masks_native[method][phase]
                        for method in METHODS
                    },
                }
            plot_endpoint_overlay(
                patient_id,
                phase_arrays,
                figures_dir / f"{patient_id}_endpoint_overlay.png",
            )

        input_records.append(
            {
                "patient_id": patient_id,
                "group": str(info.get("Group", "unknown")),
                "cine_relative_path": cine_path.relative_to(data_dir).as_posix(),
                "cine_bytes": cine_path.stat().st_size,
                "cine_sha256": sha256(cine_path),
                "cine_shape": list(cine_image.shape),
                "zooms": [float(value) for value in cine_image.header.get_zooms()],
                "voxel_volume_ml": voxel_volume_ml,
                "cine_physical_geometry": cine_geometry,
                "endpoint_grid_audit": endpoint_geometry,
                "reference_ed_frame": int(info["ED"]),
                "reference_es_frame": int(info["ES"]),
                "endpoint_input_mae_ed": input_mae["ED"],
                "endpoint_input_mae_es": input_mae["ES"],
                "historical_endpoint_max_abs_dice_difference": historical_difference[
                    "maximum"
                ],
                "info_cfg_sha256": sha256(info_path),
                "endpoint_label_sha256": {
                    phase: sha256(endpoints[int(info[phase])]["label"])
                    for phase in ("ED", "ES")
                },
            }
        )
        elapsed = time.perf_counter() - patient_start
        fused_row = next(row for row in patient_rows if row["method"] == "temporal_fusion")
        print(
            f"[{patient_index:02d}/{len(selected_patients):02d}] {patient_id}: "
            f"{cine.shape[-1]} frames, endpoint Dice="
            f"{0.5 * (float(fused_row['ed_resized_mean_dice']) + float(fused_row['es_resized_mean_dice'])) * 100:.2f}%, "
            f"EF error={float(fused_row['annotated_ef_abs_error_pp']):.2f} pp, "
            f"{elapsed:.1f} s"
        )
        del probabilities, framewise_masks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    methods_summary = aggregate_results(all_patient_rows)
    plot_population_curve(
        all_frame_rows, all_patient_rows, figures_dir / "population_lv_motion_curve.png"
    )
    plot_ef_scatter(all_patient_rows, figures_dir / "ejection_fraction_validation.png")

    frame_csv = output_dir / "frame_metrics.csv"
    patient_csv = output_dir / "patient_metrics.csv"
    write_csv(frame_csv, all_frame_rows, FRAME_FIELDS)
    write_csv(patient_csv, all_patient_rows, patient_metric_fields(all_patient_rows))

    groups: Dict[str, Dict[str, object]] = {}
    for group in sorted({str(row["group"]) for row in all_patient_rows}):
        rows = [
            row
            for row in all_patient_rows
            if row["group"] == group and row["method"] == "temporal_fusion"
        ]
        groups[group] = {
            "patients": len(rows),
            "annotated_ef_mae_percentage_points": finite_mean(
                [float(row["annotated_ef_abs_error_pp"]) for row in rows]
            ),
            "endpoint_resized_mean_dice": finite_mean(
                [
                    0.5
                    * (
                        float(row["ed_resized_mean_dice"])
                        + float(row["es_resized_mean_dice"])
                    )
                    for row in rows
                ]
            ),
        }

    summary: Dict[str, object] = {
        "schema_version": 1,
        "status": "complete_validation_analysis" if full_run else "smoke_test_subset",
        "analysis_name": "full-cine segmentation-derived spatio-temporal motion and function analysis",
        "patients": len(selected_patients),
        "frames": len(all_frame_rows) // len(METHODS),
        "validation_scope": "audited held-out validation patients; not an independent test set",
        "model": args.model,
        "spatial_backbone_temporal_scope": (
            "The checkpoint-compatible backbone is spatial 3D. Temporal information is "
            "introduced only by fixed adjacent-frame probability fusion and full-cycle analysis."
        ),
        "temporal_probability_fusion_weights": {
            "previous": TEMPORAL_WEIGHTS[0],
            "current": TEMPORAL_WEIGHTS[1],
            "next": TEMPORAL_WEIGHTS[2],
            "boundary": "circular",
        },
        "methods": methods_summary,
        "pathology_groups_descriptive": groups,
        "scientific_boundaries": [
            "Intermediate cine frames have no manual segmentation labels in ACDC training data.",
            "ED/ES endpoint masks and Info.cfg phase indices provide the available temporal validation.",
            "Raw ACDC endpoint and 4D cine affines may differ despite identical endpoint voxel arrays; grid registration is proven by shape, voxel spacing, and ED/ES image content, while physical magnitudes use cine header zooms.",
            "The analysis is not optical flow, dense registration, local strain, or external clinical validation.",
            "The validation cohort was also used historically for checkpoint selection.",
        ],
        "geometry_audit": {
            "physical_metric_policy": "header voxel sizes with orthonormal affine directions when valid; distances and volumes are origin/orientation invariant",
            "patients_with_any_endpoint_affine_mismatch": sorted(
                {
                    str(record["patient_id"])
                    for record in input_records
                    if any(
                        not bool(
                            record["endpoint_grid_audit"][phase][kind][
                                "affine_matches"
                            ]
                        )
                        for phase in ("ED", "ES")
                        for kind in ("image_vs_cine", "label_vs_cine")
                    )
                },
                key=natural_sort_key,
            ),
        },
        "endpoint_input_max_normalized_mae": max(
            max(float(record["endpoint_input_mae_ed"]), float(record["endpoint_input_mae_es"]))
            for record in input_records
        ),
        "historical_endpoint_max_abs_dice_difference": max(
            float(record["historical_endpoint_max_abs_dice_difference"])
            for record in input_records
        ),
        "runtime_seconds": time.perf_counter() - total_start,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(sanitize_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "input_manifest.json").write_text(
        json.dumps(sanitize_json(input_records), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": None,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": bool(args.amp),
        "batch_size": args.batch_size,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "model_name": checkpoint_payload.get("model_name"),
            "epoch": checkpoint_payload.get("epoch"),
            "val_mean_dice": checkpoint_payload.get("val_mean_dice"),
        },
        "split_sha256": sha256(split_path),
        "historical_per_case_sha256": sha256(historical_per_case_path),
        "source_sha256": {
            "cine_analysis": sha256(Path(__file__)),
            "metrics": sha256(Path(__file__).with_name("cardiac_motion_metrics.py")),
            "rigorous_pipeline": sha256(
                Path(__file__).with_name("21_rigorous_experiment_pipeline.py")
            ),
            "nano_mamba_core": sha256(Path(__file__).with_name("nano_mamba_core.py")),
        },
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "selected_patients": selected_patients,
    }
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        provenance["git_commit"] = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    (output_dir / "run_provenance.json").write_text(
        json.dumps(sanitize_json(provenance), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "analysis_report.md", summary)

    artifact_records = []
    for path in sorted(
        [item for item in output_dir.rglob("*") if item.is_file()],
        key=lambda item: natural_sort_key(item.relative_to(output_dir).as_posix()),
    ):
        artifact_records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": summary["status"],
                "manifest_excludes_self": True,
                "artifacts": artifact_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nSPATIO-TEMPORAL CINE ANALYSIS: COMPLETE")
    print(f"Status: {summary['status']}")
    print(f"Patients: {summary['patients']}; cine frames: {summary['frames']}")
    print(f"Output: {output_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
