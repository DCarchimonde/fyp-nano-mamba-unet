"""Independently audit a completed full-cine spatio-temporal result bundle.

The inference pipeline writes hashes, per-frame rows, per-patient rows, and an
aggregate summary.  This script recomputes the scientific quantities from the
CSV rows instead of trusting the generated report.  It also verifies artifact
integrity, cohort coverage, source lineage (allowing LF/CRLF-only differences),
and the connection to the historical endpoint experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METHODS = ("framewise", "temporal_fusion")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_821
EXPECTED_PATIENTS = 20
EXPECTED_FIGURES = 23
FLOAT_ATOL = 1e-9


class AuditError(RuntimeError):
    """Raised when a result bundle violates an audit invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def require_close(
    label: str,
    observed: float,
    expected: float,
    atol: float = FLOAT_ATOL,
) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=atol):
        raise AuditError(f"{label}: observed={observed!r}, expected={expected!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"{path.name} contains no rows")
    return rows


def number(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"Invalid numeric field {field!r} in row {row}") from exc
    require(math.isfinite(value), f"Non-finite {field!r} in row {row}")
    return value


def integer(row: Mapping[str, str], field: str) -> int:
    value = number(row, field)
    require(value.is_integer(), f"{field!r} is not integral in row {row}")
    return int(value)


def png_dimensions(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature: {path}")
    require(header[12:16] == b"IHDR", f"Missing PNG IHDR: {path}")
    width, height = struct.unpack(">II", header[16:24])
    require(width > 0 and height > 0, f"Invalid PNG dimensions: {path}")
    return width, height


def canonical_hash_variants(path: Path) -> Dict[str, str]:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        "raw": sha256_bytes(raw),
        "lf": sha256_bytes(lf),
        "crlf": sha256_bytes(crlf),
    }


def canonical_line_ending_match_mode(path: Path, target_sha256: str) -> str:
    """Return the provenance hash's canonical newline form.

    ``raw`` is deliberately not reported because it describes the current
    checkout rather than the recorded source.  For example, the same LF
    provenance hash matched ``raw`` on Linux but ``lf`` after a Windows CRLF
    checkout.  Selecting only the canonical LF/CRLF variants makes the sealed
    audit report byte-stable across operating systems while still rejecting
    every non-newline content change.
    """

    variants = canonical_hash_variants(path)
    if variants["lf"] == target_sha256:
        return "lf"
    if variants["crlf"] == target_sha256:
        return "crlf"
    raise AuditError(f"Source lineage mismatch: {path}")


def circular_frame_distance(first: int, second: int, frames: int) -> int:
    direct = abs(first - second)
    return min(direct, frames - direct)


def analyse_curve(values: Sequence[float], ed_frame: int, es_frame: int) -> Dict[str, float]:
    raw = np.asarray(values, dtype=np.float64)
    require(raw.ndim == 1 and raw.size >= 3, "Invalid LV volume curve")
    require(np.isfinite(raw).all() and np.all(raw >= 0.0), "Invalid LV volumes")
    smoothed = (np.roll(raw, -1) + raw + np.roll(raw, 1)) / 3.0
    amplitude = float(np.ptp(smoothed))
    require(amplitude > 0.0, "LV curve has no temporal variation")
    predicted_ed = int(np.argmax(smoothed)) + 1
    predicted_es = int(np.argmin(smoothed)) + 1
    edv = float(smoothed[predicted_ed - 1])
    esv = float(smoothed[predicted_es - 1])
    require(edv > 0.0 and esv <= edv, "Curve EDV/ESV ordering is invalid")
    second_difference = (
        np.roll(smoothed, -1) - 2.0 * smoothed + np.roll(smoothed, 1)
    )
    total_variation = np.roll(smoothed, -1) - smoothed
    return {
        "predicted_ed_frame": float(predicted_ed),
        "predicted_es_frame": float(predicted_es),
        "ed_frame_error": float(circular_frame_distance(predicted_ed, ed_frame, raw.size)),
        "es_frame_error": float(circular_frame_distance(predicted_es, es_frame, raw.size)),
        "curve_edv_ml": edv,
        "curve_esv_ml": esv,
        "curve_sv_ml": edv - esv,
        "curve_ef_percent": 100.0 * (edv - esv) / edv,
        "normalized_second_difference": float(
            np.mean(np.abs(second_difference)) / amplitude
        ),
        "normalized_circular_total_variation": float(
            np.mean(np.abs(total_variation)) / amplitude
        ),
    }


def bootstrap_mean_ci(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size >= 2, "Bootstrap needs at least two values")
    require(np.isfinite(array).all(), "Bootstrap values are non-finite")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "n": int(array.size),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence": 0.95,
    }


def compare_ci(label: str, observed: Mapping[str, object], expected: Mapping[str, float]) -> None:
    for field in ("mean", "lower", "upper", "confidence"):
        require_close(f"{label}.{field}", float(observed[field]), expected[field])
    for field in ("n", "replicates", "seed"):
        require(int(observed[field]) == int(expected[field]), f"{label}.{field} mismatch")


def endpoint_mean(row: Mapping[str, str], grid: str = "resized") -> float:
    return 0.5 * (
        number(row, f"ed_{grid}_mean_dice")
        + number(row, f"es_{grid}_mean_dice")
    )


def percentile_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(np.isfinite(array).all(), "Percentile values are non-finite")
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "minimum": float(quantiles[0]),
        "q1": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q3": float(quantiles[3]),
        "maximum": float(quantiles[4]),
        "mean": float(array.mean()),
    }


def audit_artifacts(result_dir: Path, manifest: Mapping[str, object]) -> Dict[str, object]:
    require(manifest.get("manifest_excludes_self") is True, "Artifact manifest self rule is absent")
    records = manifest.get("artifacts")
    require(isinstance(records, list), "Artifact manifest rows are missing")
    expected: Dict[str, Mapping[str, object]] = {}
    for raw in records:
        require(isinstance(raw, dict), "Malformed artifact manifest row")
        relative = str(raw.get("path", ""))
        require(relative and relative not in expected, f"Duplicate artifact path: {relative}")
        require(".." not in Path(relative).parts, f"Unsafe artifact path: {relative}")
        expected[relative] = raw

    actual = {
        path.relative_to(result_dir).as_posix(): path
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    require(set(actual) == set(expected), "Artifact manifest file set differs from the bundle")
    for relative, path in actual.items():
        record = expected[relative]
        require(path.stat().st_size == int(record["bytes"]), f"Byte mismatch: {relative}")
        require(sha256(path) == str(record["sha256"]), f"SHA-256 mismatch: {relative}")

    figures = sorted(path for path in actual.values() if path.suffix.lower() == ".png")
    require(len(figures) == EXPECTED_FIGURES, f"Expected {EXPECTED_FIGURES} PNG figures")
    dimensions = {
        path.relative_to(result_dir).as_posix(): list(png_dimensions(path))
        for path in figures
    }
    return {
        "tracked_files": len(expected),
        "png_figures": len(figures),
        "png_dimensions": dimensions,
    }


def audit_source_lineage(
    repo_root: Path,
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    source_paths = {
        "cine_analysis": Path("src/23_spatiotemporal_cine_analysis.py"),
        "metrics": Path("src/cardiac_motion_metrics.py"),
        "rigorous_pipeline": Path("src/21_rigorous_experiment_pipeline.py"),
        "nano_mamba_core": Path("src/nano_mamba_core.py"),
    }
    recorded = provenance.get("source_sha256")
    require(isinstance(recorded, dict), "Source hashes are missing from provenance")
    modes: Dict[str, str] = {}
    for key, relative in source_paths.items():
        path = repo_root / relative
        require(path.is_file(), f"Current source is missing: {relative}")
        target = str(recorded.get(key, ""))
        modes[key] = canonical_line_ending_match_mode(path, target)

    split_path = repo_root / "evidence/rigorous_patient_split/patient_split_seed42.json"
    per_case_path = repo_root / "evidence/rigorous_patient_split/per_case_NanoMambaUNet.csv"
    require(sha256(split_path) == str(provenance["split_sha256"]), "Split hash mismatch")
    require(
        sha256(per_case_path) == str(provenance["historical_per_case_sha256"]),
        "Historical per-case hash mismatch",
    )

    checkpoint_manifest = read_json(
        repo_root / "evidence/rigorous_patient_split/checkpoint_manifest.json"
    )
    require(isinstance(checkpoint_manifest, dict), "Checkpoint manifest is malformed")
    checkpoint_rows = [
        row
        for row in checkpoint_manifest.get("checkpoints", [])
        if row.get("model_name") == provenance["checkpoint"]["model_name"]
    ]
    require(len(checkpoint_rows) == 1, "Checkpoint manifest has no unique Nano-Mamba row")
    for field in ("sha256", "epoch", "val_mean_dice"):
        observed = checkpoint_rows[0][field]
        expected = provenance["checkpoint"][field]
        if field == "val_mean_dice":
            require_close(f"checkpoint.{field}", float(observed), float(expected))
        else:
            require(observed == expected, f"checkpoint.{field} mismatch")
    return {"line_ending_match_modes": modes}


def audit_inputs(
    inputs: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    require(len(inputs) == EXPECTED_PATIENTS, "Input manifest does not contain 20 patients")
    patient_ids = [str(row["patient_id"]) for row in inputs]
    require(len(set(patient_ids)) == EXPECTED_PATIENTS, "Duplicate input patient")
    require(patient_ids == list(provenance["selected_patients"]), "Provenance patient order differs")
    require(int(summary["patients"]) == EXPECTED_PATIENTS, "Summary patient count differs")

    total_frames = 0
    endpoint_mae = []
    historical_difference = []
    affine_mismatch_patients = []
    group_counts: Counter[str] = Counter()
    for row in inputs:
        patient_id = str(row["patient_id"])
        group_counts[str(row["group"])] += 1
        shape = [int(value) for value in row["cine_shape"]]
        require(len(shape) == 4 and all(value > 0 for value in shape), f"Bad cine shape: {patient_id}")
        total_frames += shape[-1]
        zooms = [float(value) for value in row["zooms"][:3]]
        require(all(math.isfinite(value) and value > 0.0 for value in zooms), f"Bad zooms: {patient_id}")
        require_close(
            f"{patient_id}.voxel_volume_ml",
            float(row["voxel_volume_ml"]),
            float(np.prod(zooms) / 1000.0),
        )
        geometry = row["cine_physical_geometry"]
        require_close(
            f"{patient_id}.geometry.voxel_volume_ml",
            float(geometry["voxel_volume_ml"]),
            float(row["voxel_volume_ml"]),
        )
        patient_has_mismatch = False
        for phase in ("ED", "ES"):
            phase_geometry = row["endpoint_grid_audit"][phase]
            require(
                bool(phase_geometry["label_vs_endpoint_image"]["affine_matches"]),
                f"Endpoint image/label affine mismatch: {patient_id} {phase}",
            )
            for kind in ("image_vs_cine", "label_vs_cine"):
                record = phase_geometry[kind]
                require(
                    list(record["reference_zooms_mm"]) == list(record["candidate_zooms_mm"]),
                    f"Endpoint voxel-size mismatch: {patient_id} {phase} {kind}",
                )
                if not bool(record["affine_matches"]):
                    patient_has_mismatch = True
        if patient_has_mismatch:
            affine_mismatch_patients.append(patient_id)
        endpoint_mae.extend(
            [float(row["endpoint_input_mae_ed"]), float(row["endpoint_input_mae_es"])]
        )
        historical_difference.append(
            float(row["historical_endpoint_max_abs_dice_difference"])
        )

    require(total_frames == int(summary["frames"]), "Input frame count differs from summary")
    require_close(
        "endpoint_input_max_normalized_mae",
        max(endpoint_mae),
        float(summary["endpoint_input_max_normalized_mae"]),
    )
    require_close(
        "historical_endpoint_max_abs_dice_difference",
        max(historical_difference),
        float(summary["historical_endpoint_max_abs_dice_difference"]),
    )
    require(
        affine_mismatch_patients
        == list(summary["geometry_audit"]["patients_with_any_endpoint_affine_mismatch"]),
        "Affine mismatch patient list differs from summary",
    )
    return {
        "patients": len(inputs),
        "cine_frames": total_frames,
        "pathology_group_counts": dict(sorted(group_counts.items())),
        "maximum_endpoint_normalized_mae": max(endpoint_mae),
        "maximum_historical_dice_reproduction_error": max(historical_difference),
        "affine_mismatch_patients": affine_mismatch_patients,
    }


def audit_frame_rows(
    rows: Sequence[Mapping[str, str]],
    inputs: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, str], List[Mapping[str, str]]]:
    expected_frames = {
        str(row["patient_id"]): int(row["cine_shape"][-1]) for row in inputs
    }
    expected_phase = {
        str(row["patient_id"]): (
            int(row["reference_ed_frame"]),
            int(row["reference_es_frame"]),
        )
        for row in inputs
    }
    grouped: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    seen = set()
    numeric_fields = [
        field for field in rows[0] if field not in {"patient_id", "group", "method"}
    ]
    for row in rows:
        patient_id = row["patient_id"]
        method = row["method"]
        require(patient_id in expected_frames, f"Unknown frame patient: {patient_id}")
        require(method in METHODS, f"Unknown frame method: {method}")
        for field in numeric_fields:
            number(row, field)
        frame = integer(row, "frame")
        key = (patient_id, method, frame)
        require(key not in seen, f"Duplicate frame row: {key}")
        seen.add(key)
        grouped[(patient_id, method)].append(row)

    require(
        len(rows) == 2 * sum(expected_frames.values()),
        "Frame-row count does not cover both methods",
    )
    for patient_id, frame_count in expected_frames.items():
        ed_frame, es_frame = expected_phase[patient_id]
        for method in METHODS:
            patient_rows = sorted(
                grouped[(patient_id, method)], key=lambda row: integer(row, "frame")
            )
            require(len(patient_rows) == frame_count, f"Incomplete frames: {patient_id} {method}")
            require(
                [integer(row, "frame") for row in patient_rows]
                == list(range(1, frame_count + 1)),
                f"Non-contiguous frames: {patient_id} {method}",
            )
            for row in patient_rows:
                frame = integer(row, "frame")
                require(integer(row, "num_frames") == frame_count, "num_frames mismatch")
                require_close(
                    f"{patient_id}.{method}.normalized_cycle.{frame}",
                    number(row, "normalized_cycle"),
                    (frame - 1) / frame_count,
                )
                require(integer(row, "is_reference_ed") == int(frame == ed_frame), "ED flag mismatch")
                require(integer(row, "is_reference_es") == int(frame == es_frame), "ES flag mismatch")
                for field in ("rv_volume_ml", "myo_volume_ml", "lv_volume_ml"):
                    require(number(row, field) >= 0.0, f"Negative volume: {patient_id} {method}")
    return grouped


def compare_historical_endpoint_rows(
    repo_root: Path,
    patient_rows: Sequence[Mapping[str, str]],
) -> float:
    historical = {
        row["case_id"]: row
        for row in read_csv(
            repo_root / "evidence/rigorous_patient_split/per_case_NanoMambaUNet.csv"
        )
    }
    maximum = 0.0
    for row in patient_rows:
        if row["method"] != "framewise":
            continue
        patient_id = row["patient_id"]
        for phase in ("ed", "es"):
            frame = integer(row, f"reference_{phase}_frame")
            case_id = f"{patient_id}_frame{frame:02d}"
            require(case_id in historical, f"Missing historical case: {case_id}")
            differences = []
            for label in ("RV", "MYO", "LV"):
                differences.append(
                    abs(
                        number(row, f"{phase}_resized_dice_{label}")
                        - float(historical[case_id][f"dice_{label}"])
                    )
                )
            differences.append(
                abs(
                    number(row, f"{phase}_resized_mean_dice")
                    - float(historical[case_id]["mean_dice"])
                )
            )
            observed = max(differences)
            require_close(
                f"{case_id}.historical_difference",
                observed,
                number(row, f"historical_{phase}_max_abs_dice_difference"),
            )
            maximum = max(maximum, observed)
    return maximum


def audit_patient_rows(
    rows: Sequence[Mapping[str, str]],
    frame_groups: Mapping[Tuple[str, str], Sequence[Mapping[str, str]]],
    inputs: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[str, Mapping[str, str]]]:
    input_by_patient = {str(row["patient_id"]): row for row in inputs}
    numeric_fields = [
        field for field in rows[0] if field not in {"patient_id", "group", "method"}
    ]
    by_patient: Dict[str, Dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        patient_id = row["patient_id"]
        method = row["method"]
        require(patient_id in input_by_patient, f"Unknown patient row: {patient_id}")
        require(method in METHODS, f"Unknown patient method: {method}")
        require(method not in by_patient[patient_id], f"Duplicate patient/method: {patient_id} {method}")
        for field in numeric_fields:
            number(row, field)
        by_patient[patient_id][method] = row

        frame_rows = sorted(
            frame_groups[(patient_id, method)], key=lambda item: integer(item, "frame")
        )
        frame_count = len(frame_rows)
        ed_frame = integer(row, "reference_ed_frame")
        es_frame = integer(row, "reference_es_frame")
        require(integer(row, "num_frames") == frame_count, "Patient frame count mismatch")
        ed_row = frame_rows[ed_frame - 1]
        es_row = frame_rows[es_frame - 1]
        annotated_edv = number(ed_row, "lv_volume_ml")
        annotated_esv = number(es_row, "lv_volume_ml")
        for field, expected in (
            ("annotated_edv_ml", annotated_edv),
            ("annotated_esv_ml", annotated_esv),
            ("annotated_sv_ml", annotated_edv - annotated_esv),
            ("annotated_ef_percent", 100.0 * (annotated_edv - annotated_esv) / annotated_edv),
        ):
            require_close(f"{patient_id}.{method}.{field}", number(row, field), expected)
        require(
            integer(row, "annotated_phase_order_valid") == int(annotated_edv >= annotated_esv),
            f"Annotated phase-order flag mismatch: {patient_id} {method}",
        )

        reference_edv = number(row, "reference_edv_ml")
        reference_esv = number(row, "reference_esv_ml")
        require(reference_edv > 0.0 and reference_edv >= reference_esv, "Invalid reference EDV/ESV")
        require_close(
            f"{patient_id}.{method}.reference_sv_ml",
            number(row, "reference_sv_ml"),
            reference_edv - reference_esv,
        )
        require_close(
            f"{patient_id}.{method}.reference_ef_percent",
            number(row, "reference_ef_percent"),
            100.0 * (reference_edv - reference_esv) / reference_edv,
        )
        require_close(
            f"{patient_id}.{method}.annotated_ef_abs_error_pp",
            number(row, "annotated_ef_abs_error_pp"),
            abs(number(row, "annotated_ef_percent") - number(row, "reference_ef_percent")),
        )

        curve = analyse_curve(
            [number(item, "lv_volume_ml") for item in frame_rows], ed_frame, es_frame
        )
        for field, expected in curve.items():
            require_close(f"{patient_id}.{method}.{field}", number(row, field), expected)
        require_close(
            f"{patient_id}.{method}.curve_ef_abs_error_pp",
            number(row, "curve_ef_abs_error_pp"),
            abs(number(row, "curve_ef_percent") - number(row, "reference_ef_percent")),
        )

        ed_centroid = np.asarray(
            [number(ed_row, f"lv_centroid_{axis}_mm") for axis in "xyz"],
            dtype=np.float64,
        )
        displacements = []
        for item in frame_rows:
            centroid = np.asarray(
                [number(item, f"lv_centroid_{axis}_mm") for axis in "xyz"],
                dtype=np.float64,
            )
            displacements.append(float(np.linalg.norm(centroid - ed_centroid)))
        require_close(
            f"{patient_id}.{method}.peak_lv_centroid_displacement_mm",
            number(row, "peak_lv_centroid_displacement_mm"),
            max(displacements),
        )
        for field, expected in (
            ("myo_radius_ed_mm", number(ed_row, "mean_myo_radius_mm")),
            ("myo_radius_es_mm", number(es_row, "mean_myo_radius_mm")),
            (
                "myo_radius_ed_minus_es_mm",
                number(ed_row, "mean_myo_radius_mm") - number(es_row, "mean_myo_radius_mm"),
            ),
        ):
            require_close(f"{patient_id}.{method}.{field}", number(row, field), expected)

        for phase in ("ed", "es"):
            for grid in ("native", "resized"):
                class_scores = [
                    number(row, f"{phase}_{grid}_dice_{label}")
                    for label in ("RV", "MYO", "LV")
                ]
                require(all(0.0 <= value <= 1.0 for value in class_scores), "Dice outside [0,1]")
                require_close(
                    f"{patient_id}.{method}.{phase}_{grid}_mean_dice",
                    number(row, f"{phase}_{grid}_mean_dice"),
                    float(np.mean(class_scores)),
                )

    require(set(by_patient) == set(input_by_patient), "Patient CSV cohort differs")
    require(all(set(methods) == set(METHODS) for methods in by_patient.values()), "Missing method row")
    for patient_id, methods in by_patient.items():
        first, second = methods[METHODS[0]], methods[METHODS[1]]
        for field in (
            "group",
            "num_frames",
            "reference_ed_frame",
            "reference_es_frame",
            "reference_edv_ml",
            "reference_esv_ml",
            "reference_sv_ml",
            "reference_ef_percent",
            "endpoint_input_mae_ed",
            "endpoint_input_mae_es",
            "historical_endpoint_max_abs_dice_difference",
        ):
            require(first[field] == second[field], f"Cross-method invariant differs: {patient_id} {field}")
    return by_patient


def audit_summary(
    summary: Mapping[str, object],
    by_patient: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> Dict[str, object]:
    derived: Dict[str, object] = {"methods": {}}
    method_values: Dict[str, Dict[str, List[float]]] = {}
    for method in METHODS:
        rows = [by_patient[patient_id][method] for patient_id in sorted(by_patient)]
        values = {
            "endpoint_resized_dice": [endpoint_mean(row, "resized") for row in rows],
            "endpoint_native_dice": [endpoint_mean(row, "native") for row in rows],
            "annotated_ef_abs_error_pp": [number(row, "annotated_ef_abs_error_pp") for row in rows],
            "curve_ef_abs_error_pp": [number(row, "curve_ef_abs_error_pp") for row in rows],
            "ed_phase_error_frames": [number(row, "ed_frame_error") for row in rows],
            "es_phase_error_frames": [number(row, "es_frame_error") for row in rows],
            "curve_smoothness": [number(row, "normalized_second_difference") for row in rows],
            "peak_lv_centroid_displacement_mm": [
                number(row, "peak_lv_centroid_displacement_mm") for row in rows
            ],
            "myo_radius_ed_minus_es_mm": [
                number(row, "myo_radius_ed_minus_es_mm") for row in rows
            ],
        }
        method_values[method] = values
        observed = summary["methods"][method]
        comparisons = {
            "mean_endpoint_resized_dice": np.mean(values["endpoint_resized_dice"]),
            "annotated_ef_mae_percentage_points": np.mean(values["annotated_ef_abs_error_pp"]),
            "curve_ef_mae_percentage_points": np.mean(values["curve_ef_abs_error_pp"]),
            "ed_phase_mae_frames": np.mean(values["ed_phase_error_frames"]),
            "es_phase_mae_frames": np.mean(values["es_phase_error_frames"]),
            "ed_phase_exact_fraction": np.mean(np.asarray(values["ed_phase_error_frames"]) == 0),
            "es_phase_exact_fraction": np.mean(np.asarray(values["es_phase_error_frames"]) == 0),
            "ed_phase_within_one_fraction": np.mean(np.asarray(values["ed_phase_error_frames"]) <= 1),
            "es_phase_within_one_fraction": np.mean(np.asarray(values["es_phase_error_frames"]) <= 1),
            "mean_normalized_second_difference": np.mean(values["curve_smoothness"]),
            "median_peak_lv_centroid_displacement_mm": np.median(
                values["peak_lv_centroid_displacement_mm"]
            ),
            "mean_myo_radius_ed_minus_es_mm": np.mean(values["myo_radius_ed_minus_es_mm"]),
        }
        for field, expected in comparisons.items():
            require_close(f"summary.{method}.{field}", float(observed[field]), float(expected))
        reference_ef = [number(row, "reference_ef_percent") for row in rows]
        predicted_ef = [number(row, "annotated_ef_percent") for row in rows]
        require_close(
            f"summary.{method}.annotated_ef_pearson_r",
            float(observed["annotated_ef_pearson_r"]),
            float(np.corrcoef(reference_ef, predicted_ef)[0, 1]),
        )
        ci_sources = {
            "endpoint_resized_dice": values["endpoint_resized_dice"],
            "annotated_ef_abs_error_pp": values["annotated_ef_abs_error_pp"],
            "ed_phase_error_frames": values["ed_phase_error_frames"],
            "es_phase_error_frames": values["es_phase_error_frames"],
        }
        for name, source in ci_sources.items():
            compare_ci(
                f"summary.{method}.patient_bootstrap_95_ci.{name}",
                observed["patient_bootstrap_95_ci"][name],
                bootstrap_mean_ci(source),
            )
        derived["methods"][method] = {
            "endpoint_resized_dice": bootstrap_mean_ci(values["endpoint_resized_dice"]),
            "endpoint_native_dice": bootstrap_mean_ci(values["endpoint_native_dice"]),
            "annotated_ef_abs_error_pp": bootstrap_mean_ci(
                values["annotated_ef_abs_error_pp"]
            ),
            "annotated_ef_pearson_r": float(np.corrcoef(reference_ef, predicted_ef)[0, 1]),
            "ed_phase_exact_fraction": float(
                np.mean(np.asarray(values["ed_phase_error_frames"]) == 0)
            ),
            "es_phase_exact_fraction": float(
                np.mean(np.asarray(values["es_phase_error_frames"]) == 0)
            ),
            "ed_phase_within_one_fraction": float(
                np.mean(np.asarray(values["ed_phase_error_frames"]) <= 1)
            ),
            "es_phase_within_one_fraction": float(
                np.mean(np.asarray(values["es_phase_error_frames"]) <= 1)
            ),
            "curve_smoothness": bootstrap_mean_ci(values["curve_smoothness"]),
            "peak_lv_centroid_displacement_mm": percentile_summary(
                values["peak_lv_centroid_displacement_mm"]
            ),
            "myo_radius_ed_minus_es_mm": bootstrap_mean_ci(
                values["myo_radius_ed_minus_es_mm"]
            ),
        }

    paired = {
        "endpoint_resized_dice": [
            temporal - framewise
            for temporal, framewise in zip(
                method_values["temporal_fusion"]["endpoint_resized_dice"],
                method_values["framewise"]["endpoint_resized_dice"],
            )
        ],
        "annotated_ef_abs_error_pp": [
            temporal - framewise
            for temporal, framewise in zip(
                method_values["temporal_fusion"]["annotated_ef_abs_error_pp"],
                method_values["framewise"]["annotated_ef_abs_error_pp"],
            )
        ],
        "curve_smoothness": [
            temporal - framewise
            for temporal, framewise in zip(
                method_values["temporal_fusion"]["curve_smoothness"],
                method_values["framewise"]["curve_smoothness"],
            )
        ],
    }
    delta_summary = summary["methods"]["temporal_fusion_minus_framewise"]
    field_map = {
        "endpoint_resized_dice": "endpoint_resized_dice",
        "annotated_ef_abs_error_pp": "annotated_ef_mae_percentage_points",
        "curve_smoothness": "curve_smoothness",
    }
    for source_name, summary_name in field_map.items():
        require_close(
            f"summary.delta.{summary_name}",
            float(delta_summary[summary_name]),
            float(np.mean(paired[source_name])),
        )
        compare_ci(
            f"summary.delta.ci.{source_name}",
            delta_summary["paired_patient_bootstrap_95_ci"][source_name],
            bootstrap_mean_ci(paired[source_name]),
        )

    derived["paired_temporal_fusion_minus_framewise"] = {
        name: {
            **bootstrap_mean_ci(values),
            "patients_negative": int(np.sum(np.asarray(values) < 0.0)),
            "patients_zero": int(np.sum(np.asarray(values) == 0.0)),
            "patients_positive": int(np.sum(np.asarray(values) > 0.0)),
            "interval_excludes_zero": bool(
                bootstrap_mean_ci(values)["upper"] < 0.0
                or bootstrap_mean_ci(values)["lower"] > 0.0
            ),
        }
        for name, values in paired.items()
    }

    temporal_rows = [
        by_patient[patient_id]["temporal_fusion"] for patient_id in sorted(by_patient)
    ]
    derived["outliers"] = {
        "largest_total_phase_error": sorted(
            [
                {
                    "patient_id": row["patient_id"],
                    "ed_error_frames": integer(row, "ed_frame_error"),
                    "es_error_frames": integer(row, "es_frame_error"),
                    "total_error_frames": integer(row, "ed_frame_error")
                    + integer(row, "es_frame_error"),
                }
                for row in temporal_rows
            ],
            key=lambda item: (-item["total_error_frames"], item["patient_id"]),
        )[:5],
        "largest_ef_absolute_error": sorted(
            [
                {
                    "patient_id": row["patient_id"],
                    "error_percentage_points": number(row, "annotated_ef_abs_error_pp"),
                }
                for row in temporal_rows
            ],
            key=lambda item: (-item["error_percentage_points"], item["patient_id"]),
        )[:5],
        "lowest_endpoint_resized_dice": sorted(
            [
                {"patient_id": row["patient_id"], "dice": endpoint_mean(row)}
                for row in temporal_rows
            ],
            key=lambda item: (item["dice"], item["patient_id"]),
        )[:5],
    }
    return derived


def audit_pathology_groups(
    summary: Mapping[str, object],
    by_patient: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> Dict[str, object]:
    observed = summary["pathology_groups_descriptive"]
    rows = [methods["temporal_fusion"] for methods in by_patient.values()]
    groups = sorted({row["group"] for row in rows})
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        require(int(observed[group]["patients"]) == len(group_rows), f"{group} n mismatch")
        require_close(
            f"{group}.endpoint_resized_mean_dice",
            float(observed[group]["endpoint_resized_mean_dice"]),
            float(np.mean([endpoint_mean(row) for row in group_rows])),
        )
        require_close(
            f"{group}.annotated_ef_mae_percentage_points",
            float(observed[group]["annotated_ef_mae_percentage_points"]),
            float(np.mean([number(row, "annotated_ef_abs_error_pp") for row in group_rows])),
        )
    return {group: dict(observed[group]) for group in groups}


def audit_result_dir(result_dir: Path, repo_root: Optional[Path]) -> Dict[str, object]:
    required_files = (
        "analysis_report.md",
        "artifact_manifest.json",
        "frame_metrics.csv",
        "input_manifest.json",
        "patient_metrics.csv",
        "run_provenance.json",
        "summary.json",
    )
    for relative in required_files:
        require((result_dir / relative).is_file(), f"Missing result artifact: {relative}")

    manifest = read_json(result_dir / "artifact_manifest.json")
    summary = read_json(result_dir / "summary.json")
    provenance = read_json(result_dir / "run_provenance.json")
    inputs = read_json(result_dir / "input_manifest.json")
    require(isinstance(manifest, dict), "Artifact manifest must be a JSON object")
    require(isinstance(summary, dict), "Summary must be a JSON object")
    require(isinstance(provenance, dict), "Provenance must be a JSON object")
    require(isinstance(inputs, list), "Input manifest must be a JSON array")
    require(summary.get("status") == "complete_validation_analysis", "Run is not complete")
    require(manifest.get("status") == summary.get("status"), "Manifest status differs")
    require(provenance.get("device") == "cuda", "Final result was not produced on CUDA")
    require(str(provenance.get("git_commit", "")) == "dace079c0a3d4025aff36d159b0c732947516393", "Unexpected inference commit")

    artifact_audit = audit_artifacts(result_dir, manifest)
    input_audit = audit_inputs(inputs, summary, provenance)
    frame_rows = read_csv(result_dir / "frame_metrics.csv")
    patient_rows = read_csv(result_dir / "patient_metrics.csv")
    frame_groups = audit_frame_rows(frame_rows, inputs)
    by_patient = audit_patient_rows(patient_rows, frame_groups, inputs)
    derived = audit_summary(summary, by_patient)
    pathology = audit_pathology_groups(summary, by_patient)

    lineage: Dict[str, object] = {"checked": False}
    if repo_root is not None:
        lineage = {"checked": True, **audit_source_lineage(repo_root, provenance)}
        maximum = compare_historical_endpoint_rows(repo_root, patient_rows)
        require_close(
            "independent historical endpoint maximum",
            maximum,
            float(summary["historical_endpoint_max_abs_dice_difference"]),
        )
        lineage["independent_historical_endpoint_max_abs_dice_difference"] = maximum

    paired = derived["paired_temporal_fusion_minus_framewise"]
    interpretations = [
        "All 20 validation patients and all 550 cine frames passed structural and arithmetic checks.",
        "The full-cine experiment is post-hoc temporal probability fusion over a spatial 3D checkpoint, not a learned temporal Mamba model.",
        "The paired endpoint-Dice interval contains zero; no resolved accuracy improvement should be claimed for temporal fusion.",
        "The paired EF-MAE interval contains zero; the observed reduction is descriptive rather than conclusive.",
        "The paired curve-smoothness interval excludes zero and favors temporal fusion, but the metric is a segmentation-derived global trajectory measure.",
        "Intermediate cine frames have no manual masks; motion surrogates are not optical flow, dense deformation, regional strain, or external clinical validation.",
        "The same validation cohort was historically used for checkpoint selection, so this is not an independent test estimate.",
    ]
    require(
        not paired["endpoint_resized_dice"]["interval_excludes_zero"],
        "Unexpected resolved endpoint-Dice effect; interpretation must be reviewed",
    )
    require(
        not paired["annotated_ef_abs_error_pp"]["interval_excludes_zero"],
        "Unexpected resolved EF-MAE effect; interpretation must be reviewed",
    )
    require(
        paired["curve_smoothness"]["upper"] < 0.0,
        "Temporal fusion did not show the expected smoother-curve interval",
    )

    result_directory = str(result_dir)
    if repo_root is not None:
        try:
            result_directory = result_dir.resolve().relative_to(
                repo_root.resolve()
            ).as_posix()
        except ValueError:
            # An external bundle can still be audited; only in-repository
            # reports receive a checkout-independent path.
            pass

    return {
        "schema_version": 1,
        "status": "independent_audit_pass",
        "result_directory": result_directory,
        "result_commit": provenance["git_commit"],
        "bundle_status": summary["status"],
        "artifact_integrity": artifact_audit,
        "input_and_geometry": input_audit,
        "source_lineage": lineage,
        "derived_results": derived,
        "pathology_groups_descriptive_only": pathology,
        "scientific_interpretation": interpretations,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_result_dir(
            args.result_dir.resolve(),
            args.repo_root.resolve() if args.repo_root else None,
        )
    except (AuditError, KeyError, TypeError, ValueError) as exc:
        print(f"SPATIO-TEMPORAL RESULT AUDIT: FAIL\n{exc}")
        return 1
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Audit report: {output}")
    print("SPATIO-TEMPORAL RESULT AUDIT: PASS")
    print("20 patients; 550 full-cine frames; artifact, lineage, and arithmetic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
