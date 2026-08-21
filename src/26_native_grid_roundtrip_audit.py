"""Quantify label discretization caused by the executed resize/native round trip.

This diagnostic does not train or run a model.  It takes the 40 labelled ED/ES
validation masks, applies exactly the categorical geometry used by the project
(native -> 256x256x16 -> native, both nearest-neighbour), and measures Dice
against the original labels.  The result isolates label-boundary fidelity from
model error and makes the resized-versus-native interpretation defensible.

The round-trip score is a diagnostic, not a universal mathematical upper bound:
a particular prediction can differ from the resized reference in ways that
occasionally improve its restored native overlap.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


EXPECTED_VALIDATION_PATIENTS = 20
EXPECTED_ENDPOINT_ROWS = 40
CLASSES = ("RV", "MYO", "LV")


def load_cine_module(repo_root: Path):
    script = repo_root / "src" / "23_spatiotemporal_cine_analysis.py"
    spec = importlib.util.spec_from_file_location("cine_roundtrip_dependency", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a non-empty finite numeric sequence")
    return float(array.mean())


def summarize_roundtrip_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if len(rows) != EXPECTED_ENDPOINT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ENDPOINT_ROWS} validation endpoint rows, got {len(rows)}"
        )
    patients = sorted({str(row["patient_id"]) for row in rows})
    if len(patients) != EXPECTED_VALIDATION_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_PATIENTS} validation patients, got {len(patients)}"
        )
    for patient_id in patients:
        patient_rows = [row for row in rows if row["patient_id"] == patient_id]
        if sorted(str(row["phase"]) for row in patient_rows) != ["ED", "ES"]:
            raise ValueError(f"{patient_id} does not have exactly one ED and one ES row")

    class_means = {
        class_name: finite_mean(
            [float(row[f"roundtrip_dice_{class_name}"]) for row in rows]
        )
        for class_name in CLASSES
    }
    case_means = [float(row["roundtrip_mean_dice"]) for row in rows]
    return {
        "patients": len(patients),
        "endpoint_rows": len(rows),
        "roundtrip_mean_dice": finite_mean(case_means),
        "roundtrip_mean_dice_by_class": class_means,
        "roundtrip_mean_loss_percentage_points": 100.0 * (1.0 - finite_mean(case_means)),
        "minimum_case_mean_dice": float(min(case_means)),
        "maximum_case_mean_dice": float(max(case_means)),
    }


def load_observed_grid_gap(path: Path) -> Dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("method") == "framewise"]
    if len(rows) != EXPECTED_VALIDATION_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_PATIENTS} frame-wise patient rows in {path}"
        )
    resized_values: List[float] = []
    native_values: List[float] = []
    class_gaps: Dict[str, List[float]] = {name: [] for name in CLASSES}
    for row in rows:
        for phase in ("ed", "es"):
            for class_name in CLASSES:
                resized = float(row[f"{phase}_resized_dice_{class_name}"])
                native = float(row[f"{phase}_native_dice_{class_name}"])
                if not math.isfinite(resized) or not math.isfinite(native):
                    raise ValueError("Observed endpoint table contains a non-finite Dice")
                resized_values.append(resized)
                native_values.append(native)
                class_gaps[class_name].append(resized - native)
    resized_mean = finite_mean(resized_values)
    native_mean = finite_mean(native_values)
    return {
        "framewise_resized_endpoint_mean_dice": resized_mean,
        "framewise_native_endpoint_mean_dice": native_mean,
        "resized_minus_native_percentage_points": 100.0 * (resized_mean - native_mean),
        "resized_minus_native_by_class_percentage_points": {
            name: 100.0 * finite_mean(values) for name, values in class_gaps.items()
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "patient_id",
        "phase",
        "frame",
        "native_x",
        "native_y",
        "native_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "roundtrip_dice_RV",
        "roundtrip_dice_MYO",
        "roundtrip_dice_LV",
        "roundtrip_mean_dice",
        "label_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\AI_FYP"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--observed-patient-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    project_root = args.project_root.resolve()
    data_dir = (
        args.data_dir
        or project_root / "Data" / "ACDC" / "database" / "training"
    ).resolve()
    split_path = (
        args.split
        or repo_root / "evidence" / "rigorous_patient_split" / "patient_split_seed42.json"
    ).resolve()
    observed_path = (
        args.observed_patient_metrics
        or repo_root / "evidence" / "spatiotemporal_cine" / "raw" / "patient_metrics.csv"
    ).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or project_root / "experiment_outputs" / f"native_grid_roundtrip_{timestamp}"
    ).resolve()

    for path, label in (
        (data_dir, "ACDC training directory"),
        (split_path, "audited patient split"),
        (observed_path, "committed cine patient metrics"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    try:
        import nibabel as nib
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("nibabel and PyTorch are required for the round-trip audit") from exc

    cine = load_cine_module(repo_root)
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    patients = sorted(split.get("val_patients", []), key=cine.natural_sort_key)
    if len(patients) != EXPECTED_VALIDATION_PATIENTS or len(patients) != len(set(patients)):
        raise ValueError("The audited split must contain 20 unique validation patients")

    rows: List[Dict[str, object]] = []
    for patient_id in patients:
        patient_dir = data_dir / patient_id
        info_path = patient_dir / "Info.cfg"
        info = cine.parse_info_cfg_text(info_path.read_text(encoding="utf-8-sig"))
        endpoints = cine.discover_endpoint_pairs(patient_dir)
        for phase in ("ED", "ES"):
            frame = int(info[phase])
            if frame not in endpoints:
                raise FileNotFoundError(f"{patient_id} has no discovered {phase} frame {frame}")
            label_path = Path(endpoints[frame]["label"])
            label_image = nib.load(str(label_path))
            label = np.asarray(label_image.dataobj)
            if label.ndim != 3:
                raise ValueError(f"{patient_id} {phase} label is not 3D: {label.shape}")
            rounded = np.rint(label)
            if not np.isfinite(label).all() or not np.allclose(label, rounded, atol=1e-6):
                raise ValueError(f"{patient_id} {phase} label is not finite categorical data")
            original = rounded.astype(np.uint8)
            if not set(np.unique(original)).issubset({0, 1, 2, 3}):
                raise ValueError(f"{patient_id} {phase} contains an unexpected class")

            resized = cine.resize_label(original, torch, functional)
            restored = cine.resize_mask_to_native(
                torch.from_numpy(np.ascontiguousarray(resized)),
                original.shape,
                torch,
                functional,
            )
            if restored.shape != original.shape:
                raise RuntimeError(
                    f"{patient_id} {phase} axis/shape restoration failed: "
                    f"{restored.shape} vs {original.shape}"
                )
            if not set(np.unique(restored)).issubset({0, 1, 2, 3}):
                raise RuntimeError(f"{patient_id} {phase} interpolation created a new class")
            dice = cine.dice_by_class(restored, original)
            zooms = cine.nifti_spatial_zooms(label_image)
            rows.append(
                {
                    "patient_id": patient_id,
                    "phase": phase,
                    "frame": frame,
                    "native_x": int(original.shape[0]),
                    "native_y": int(original.shape[1]),
                    "native_z": int(original.shape[2]),
                    "spacing_x_mm": zooms[0],
                    "spacing_y_mm": zooms[1],
                    "spacing_z_mm": zooms[2],
                    "roundtrip_dice_RV": dice["dice_RV"],
                    "roundtrip_dice_MYO": dice["dice_MYO"],
                    "roundtrip_dice_LV": dice["dice_LV"],
                    "roundtrip_mean_dice": dice["mean_dice"],
                    "label_path": str(label_path),
                }
            )

    summary = summarize_roundtrip_rows(rows)
    observed = load_observed_grid_gap(observed_path)
    payload = {
        "schema_version": 1,
        "status": "native_label_roundtrip_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spatial_size": list(cine.SPATIAL_SIZE),
        "interpolation": "nearest-neighbour for native->resized and resized->native",
        "split_path": str(split_path),
        "data_dir": str(data_dir),
        "summary": summary,
        "observed_model_grid_gap": observed,
        "interpretation": [
            "The label round trip isolates categorical boundary discretization from model error.",
            "It is a diagnostic fidelity score, not a strict universal upper bound on restored prediction Dice.",
            "The observed model gap must not be attributed solely to the Z axis; in-plane resizing, through-plane resizing, boundary thickness, and model error can all contribute.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "native_label_roundtrip_per_endpoint.csv"
    json_path = output_dir / "native_label_roundtrip_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("NATIVE-GRID ROUND-TRIP AUDIT: PASS")
    print(f"Patients/endpoints: {summary['patients']}/{summary['endpoint_rows']}")
    print(
        "Native-label round-trip mean Dice: "
        f"{100.0 * float(summary['roundtrip_mean_dice']):.3f}%"
    )
    print(
        "Observed frame-wise resized-minus-native gap: "
        f"{float(observed['resized_minus_native_percentage_points']):.3f} pp"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
