"""Audit aggregate P2 consistency without requiring PyTorch, MONAI, or ACDC.

The committed summary CSV is the numerical source used by the thesis.  This
tool checks its arithmetic, its JSON mirror, the deterministic patient split,
the discovery snapshot, and evidence-file hashes.  These checks establish
internal consistency only; they do not by themselves prove that the aggregate
files were produced by a particular training run.

If original per-case rows and epoch logs are recovered, the validator checks
their completeness and agreement with the aggregate table before computing
patient-level bootstrap intervals and paired differences as post-hoc
descriptive analyses.  ``--strict-closure`` also requires confirmed checkpoint,
environment, and command evidence and exits non-zero while any gap remains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "evidence" / "rigorous_patient_split"

EXPECTED_MODELS = (
    "UNet3D",
    "NanoMambaUNet",
    "Ablation_NoMamba_UNet",
    "Ablation_HalfMamba_UNet",
    "AttentionUNet",
    "SegResNet16",
)
AGGREGATE_FILES = (
    "summary_metrics.csv",
    "summary_metrics.json",
    "patient_split_seed42.json",
    "data_discovery_report.json",
)
METRIC_FIELDS = ("dice_RV", "dice_MYO", "dice_LV", "mean_dice")
SUMMARY_DICE_FIELDS = (
    "val_dice_RV",
    "val_dice_MYO",
    "val_dice_LV",
    "val_mean_dice",
)
MAX_EPOCHS = 150
LINEAGE_FILE = "historical_source_lineage.json"
DATASET_MANIFEST_FILE = "posthoc_dataset_manifest.json"


class EvidenceError(RuntimeError):
    """Raised when supplied P2 evidence is absent or internally inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_git_text(path: Path) -> str:
    """Hash text as Git stores it, independent of checkout line endings.

    Git may materialize tracked text with CRLF in a Windows working tree even
    though the repository blob and recorded audit hash use LF.  Normalize only
    CRLF pairs so line-ending conversion cannot create a false lineage failure;
    every other byte remains covered by the SHA-256 check.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def natural_patient_key(patient_id: str) -> Tuple[str, int]:
    prefix = patient_id.rstrip("0123456789")
    suffix = patient_id[len(prefix) :]
    return prefix, int(suffix) if suffix else -1


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def require_files(evidence_dir: Path) -> Dict[str, Path]:
    paths = {name: evidence_dir / name for name in AGGREGATE_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise EvidenceError("Missing aggregate evidence: " + ", ".join(missing))
    return paths


def validate_split(split: Mapping[str, Any]) -> Dict[str, Any]:
    seed = split.get("seed")
    fraction = float(split.get("val_fraction", -1))
    train = list(split.get("train_patients", []))
    val = list(split.get("val_patients", []))
    train_set, val_set = set(train), set(val)
    expected = {f"patient{i:03d}" for i in range(1, 101)}

    errors = []
    if seed != 42:
        errors.append(f"seed is {seed!r}, expected 42")
    if not close(fraction, 0.20):
        errors.append(f"val_fraction is {fraction!r}, expected 0.20")
    if len(train) != 80 or len(train_set) != 80:
        errors.append("training split must contain 80 unique patients")
    if len(val) != 20 or len(val_set) != 20:
        errors.append("validation split must contain 20 unique patients")
    if train_set & val_set:
        errors.append("training and validation patient sets overlap")
    if train_set | val_set != expected:
        errors.append("split does not cover patient001 through patient100 exactly")

    shuffled = sorted(expected, key=natural_patient_key)
    random.Random(42).shuffle(shuffled)
    expected_val = set(shuffled[:20])
    if val_set != expected_val:
        errors.append("validation patients do not match random.Random(42)")
    if errors:
        raise EvidenceError("Invalid patient split: " + "; ".join(errors))

    return {
        "seed": seed,
        "val_fraction": fraction,
        "train_patients": len(train_set),
        "val_patients": len(val_set),
        "overlap": 0,
        "coverage": len(train_set | val_set),
    }


def numeric_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {"model_name": str(row["model_name"])}
    integer_fields = ("best_epoch", "num_train_cases", "num_val_cases")
    float_fields = (
        "val_dice_RV",
        "val_dice_MYO",
        "val_dice_LV",
        "val_mean_dice",
        "params_m",
        "fps",
        "latency_ms",
        "peak_vram_mb",
    )
    for field in integer_fields:
        converted[field] = int(row[field])
    for field in float_fields:
        converted[field] = float(row[field])
    if "params" in row and row["params"] not in (None, ""):
        converted["params"] = int(row["params"])
    if "checkpoint_path" in row:
        converted["checkpoint_path"] = str(row["checkpoint_path"])
    for field in ("per_case_csv", "training_log_csv"):
        if field in row:
            converted[field] = str(row[field])
    return converted


def validate_summary(
    csv_rows_raw: Sequence[Mapping[str, Any]], json_rows_raw: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    if len(csv_rows_raw) != len(EXPECTED_MODELS):
        raise EvidenceError(
            f"summary CSV has {len(csv_rows_raw)} rows, expected {len(EXPECTED_MODELS)}"
        )
    if len(json_rows_raw) != len(EXPECTED_MODELS):
        raise EvidenceError(
            f"summary JSON has {len(json_rows_raw)} rows, expected {len(EXPECTED_MODELS)}"
        )
    csv_rows = [numeric_row(row) for row in csv_rows_raw]
    json_rows = [numeric_row(row) for row in json_rows_raw]
    csv_by_model = {row["model_name"]: row for row in csv_rows}
    json_by_model = {row["model_name"]: row for row in json_rows}
    if len(csv_by_model) != len(csv_rows):
        raise EvidenceError("summary CSV contains duplicate model names")
    if len(json_by_model) != len(json_rows):
        raise EvidenceError("summary JSON contains duplicate model names")
    if set(csv_by_model) != set(EXPECTED_MODELS):
        raise EvidenceError("summary CSV model set differs from the six audited models")
    if set(json_by_model) != set(EXPECTED_MODELS):
        raise EvidenceError("summary JSON model set differs from the six audited models")

    errors = []
    for model in EXPECTED_MODELS:
        csv_row, json_row = csv_by_model[model], json_by_model[model]
        class_mean = sum(csv_row[field] for field in SUMMARY_DICE_FIELDS[:3]) / 3.0
        if not close(class_mean, csv_row["val_mean_dice"], 1e-8):
            errors.append(f"{model}: mean Dice is not the class arithmetic mean")
        if not 1 <= csv_row["best_epoch"] <= MAX_EPOCHS:
            errors.append(f"{model}: best_epoch must be within 1..{MAX_EPOCHS}")
        if csv_row["num_train_cases"] != 160 or csv_row["num_val_cases"] != 40:
            errors.append(f"{model}: expected 160 train and 40 validation cases")
        for field in SUMMARY_DICE_FIELDS:
            value = csv_row[field]
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(f"{model}: {field} must be finite and within [0, 1]")
        for field in ("params_m", "fps", "latency_ms", "peak_vram_mb"):
            value = csv_row[field]
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"{model}: {field} must be finite and positive")
        if not close(csv_row["fps"] * csv_row["latency_ms"], 1000.0, 1e-6):
            errors.append(f"{model}: FPS and latency are not reciprocal")
        expected_checkpoint = f"best_{model}.pth"
        if Path(csv_row.get("checkpoint_path", "").replace("\\", "/")).name != expected_checkpoint:
            errors.append(f"{model}: checkpoint_path does not end in {expected_checkpoint}")
        if json_row.get("params") != round(json_row["params_m"] * 1_000_000):
            errors.append(f"{model}: exact JSON parameter count disagrees with params_m")
        if Path(json_row.get("per_case_csv", "").replace("\\", "/")).name != f"per_case_{model}.csv":
            errors.append(f"{model}: JSON per_case_csv path has the wrong filename")
        if Path(json_row.get("training_log_csv", "").replace("\\", "/")).name != f"training_log_{model}.csv":
            errors.append(f"{model}: JSON training_log_csv path has the wrong filename")
        for field in (
            "best_epoch",
            "num_train_cases",
            "num_val_cases",
            "val_dice_RV",
            "val_dice_MYO",
            "val_dice_LV",
            "val_mean_dice",
            "params_m",
            "fps",
            "latency_ms",
            "peak_vram_mb",
            "checkpoint_path",
        ):
            if isinstance(csv_row[field], float):
                if not close(csv_row[field], json_row[field], 1e-8):
                    errors.append(f"{model}: CSV/JSON mismatch in {field}")
            elif csv_row[field] != json_row[field]:
                errors.append(f"{model}: CSV/JSON mismatch in {field}")
    if errors:
        raise EvidenceError("Invalid result summary: " + "; ".join(errors))

    nano = csv_by_model["NanoMambaUNet"]
    comparisons: Dict[str, Dict[str, float]] = {}
    for model in ("UNet3D", "SegResNet16", "Ablation_NoMamba_UNet", "Ablation_HalfMamba_UNet"):
        other = csv_by_model[model]
        comparisons[model] = {
            "nano_minus_other_dice_pp": (nano["val_mean_dice"] - other["val_mean_dice"]) * 100.0,
            "nano_parameter_reduction_pct": (1.0 - nano["params_m"] / other["params_m"]) * 100.0,
        }
    return csv_rows, comparisons


def required_columns(
    rows: Sequence[Mapping[str, str]], fields: Iterable[str], filename: str
) -> None:
    if not rows:
        raise EvidenceError(f"{filename} is empty")
    missing = set(fields).difference(rows[0])
    if missing:
        raise EvidenceError(
            f"{filename} is missing columns: {', '.join(sorted(missing))}"
        )


def finite_metric(value: str, context: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise EvidenceError(f"{context} must be finite and within [0, 1]")
    return converted


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceError("Cannot compute a percentile from an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def patient_means(rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, float]]:
    grouped: MutableMapping[str, MutableMapping[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_cases = set()
    for row in rows:
        patient_id, case_id = row["patient_id"], row["case_id"]
        key = (patient_id, case_id)
        if key in seen_cases:
            raise EvidenceError(f"Duplicate per-case row: {patient_id}/{case_id}")
        seen_cases.add(key)
        values = {
            field: finite_metric(row[field], f"{patient_id}/{case_id}/{field}")
            for field in METRIC_FIELDS
        }
        class_mean = sum(values[field] for field in METRIC_FIELDS[:3]) / 3.0
        if not close(class_mean, values["mean_dice"], 1e-8):
            raise EvidenceError(f"Per-case mean mismatch: {patient_id}/{case_id}")
        for field in METRIC_FIELDS:
            grouped[patient_id][field].append(values[field])
    return {
        patient: {field: sum(values) / len(values) for field, values in metrics.items()}
        for patient, metrics in grouped.items()
    }


def validate_per_case_rows(
    rows: Sequence[Mapping[str, str]],
    filename: str,
    expected_patients: set[str],
    summary_row: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    required_columns(rows, ("patient_id", "case_id", *METRIC_FIELDS), filename)
    if len(rows) != 40:
        raise EvidenceError(f"{filename} has {len(rows)} rows, expected 40")
    counts: MutableMapping[str, int] = defaultdict(int)
    for row in rows:
        patient_id = row["patient_id"]
        case_id = row["case_id"]
        if patient_id not in expected_patients:
            raise EvidenceError(f"{filename} contains non-validation patient {patient_id}")
        if not case_id.startswith(patient_id + "_"):
            raise EvidenceError(f"{filename} case {case_id} is not owned by {patient_id}")
        counts[patient_id] += 1
    if set(counts) != expected_patients or any(count != 2 for count in counts.values()):
        raise EvidenceError(f"{filename} must contain exactly two cases per validation patient")

    means = patient_means(rows)
    field_pairs = (
        ("dice_RV", "val_dice_RV"),
        ("dice_MYO", "val_dice_MYO"),
        ("dice_LV", "val_dice_LV"),
        ("mean_dice", "val_mean_dice"),
    )
    for case_field, summary_field in field_pairs:
        aggregate = sum(float(row[case_field]) for row in rows) / len(rows)
        if not close(aggregate, float(summary_row[summary_field]), 1e-8):
            raise EvidenceError(
                f"{filename} aggregate {case_field} differs from the summary"
            )
    return means


def validate_training_log(
    rows: Sequence[Mapping[str, str]], filename: str, summary_row: Mapping[str, Any]
) -> Dict[str, Any]:
    fields = (
        "epoch",
        "train_loss",
        "val_mean_dice",
        "val_dice_RV",
        "val_dice_MYO",
        "val_dice_LV",
    )
    required_columns(rows, fields, filename)
    if len(rows) != MAX_EPOCHS:
        raise EvidenceError(f"{filename} has {len(rows)} rows, expected {MAX_EPOCHS}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, MAX_EPOCHS + 1)):
        raise EvidenceError(f"{filename} epochs must be contiguous from 1 to {MAX_EPOCHS}")
    for row in rows:
        epoch = int(row["epoch"])
        train_loss = float(row["train_loss"])
        if not math.isfinite(train_loss) or train_loss < 0.0:
            raise EvidenceError(f"{filename} epoch {epoch} train_loss is invalid")
        dice = {
            field: finite_metric(row[field], f"{filename} epoch {epoch} {field}")
            for field in fields[2:]
        }
        class_mean = sum(dice[field] for field in fields[3:]) / 3.0
        if not close(class_mean, dice["val_mean_dice"], 1e-8):
            raise EvidenceError(f"{filename} epoch {epoch} class mean is inconsistent")

    best = max(rows, key=lambda row: float(row["val_mean_dice"]))
    if int(best["epoch"]) != int(summary_row["best_epoch"]):
        raise EvidenceError(f"{filename} best epoch differs from the summary")
    for field in SUMMARY_DICE_FIELDS:
        if not close(float(best[field]), float(summary_row[field]), 1e-8):
            raise EvidenceError(f"{filename} best-row {field} differs from the summary")
    final = rows[-1]
    return {
        "rows": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_val_mean_dice": float(best["val_mean_dice"]),
        "final_val_mean_dice": float(final["val_mean_dice"]),
        "best_minus_final": float(best["val_mean_dice"])
        - float(final["val_mean_dice"]),
        "initial_train_loss": float(rows[0]["train_loss"]),
        "final_train_loss": float(final["train_loss"]),
        "minimum_train_loss": min(float(row["train_loss"]) for row in rows),
    }


def per_case_diagnostics(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    """Summarize recovered rows without inferring unrecorded label contents."""
    class_fields = METRIC_FIELDS[:3]
    exact_zero = {
        field: sum(float(row[field]) == 0.0 for row in rows) for field in class_fields
    }
    exact_one = {
        field: sum(float(row[field]) == 1.0 for row in rows) for field in class_fields
    }
    return {
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "exact_zero_class_dice": exact_zero,
        "exact_unity_class_dice": exact_one,
        "empty_empty_rule_trigger_count": sum(exact_one.values()),
        "empty_empty_scope_note": (
            "The executed metric returns exactly 1.0 for an empty prediction and "
            "empty reference. Zero exact-unity class scores therefore establish "
            "that this branch did not trigger in the recovered per-case table."
        ),
    }


def validate_discovery(discovery: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors = []
    if int(discovery.get("num_patients_with_dirs", -1)) != 100:
        errors.append("discovery report does not record 100 patient directories")
    if int(discovery.get("num_cases_found", -1)) != 200:
        errors.append("discovery report does not record 200 paired paths")
    if int(discovery.get("num_skipped_empty_or_invalid_containers", -1)) < 0:
        errors.append("discovery report has an invalid skipped-container count")
    if errors:
        raise EvidenceError("; ".join(errors))

    manifest = discovery.get("case_manifest")
    gaps: List[str] = []
    manifest_status = "complete"
    if manifest is None:
        manifest_status = "not supplied in historical bundle"
        gaps.append("data_discovery_report.json (full 200-case manifest unavailable)")
    else:
        if not isinstance(manifest, list) or len(manifest) != 200:
            raise EvidenceError("data discovery case_manifest must contain 200 records")
        case_ids = set()
        patient_counts: MutableMapping[str, int] = defaultdict(int)
        for record in manifest:
            patient_id = str(record["patient_id"])
            case_id = str(record["case_id"])
            if not re.fullmatch(r"patient\d{3}", patient_id):
                raise EvidenceError(f"invalid patient identifier in case_manifest: {patient_id}")
            if not case_id.startswith(patient_id + "_") or case_id in case_ids:
                raise EvidenceError(f"invalid or duplicate case identifier: {case_id}")
            case_ids.add(case_id)
            patient_counts[patient_id] += 1
            if int(record["image_bytes"]) <= 0 or int(record["label_bytes"]) <= 0:
                raise EvidenceError(f"non-positive file size in case_manifest: {case_id}")
        expected_patients = {f"patient{i:03d}" for i in range(1, 101)}
        if set(patient_counts) != expected_patients or any(
            count != 2 for count in patient_counts.values()
        ):
            raise EvidenceError(
                "case_manifest must contain two cases for each patient001 through patient100"
            )
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != discovery.get("case_manifest_sha256"):
            raise EvidenceError("case_manifest_sha256 does not match the manifest")
    return (
        {
            "patient_directories": int(discovery["num_patients_with_dirs"]),
            "paired_paths": int(discovery["num_cases_found"]),
            "skipped_containers": int(
                discovery.get("num_skipped_empty_or_invalid_containers", 0)
            ),
            "full_case_manifest": manifest_status,
            "scope_note": (
                "Discovery establishes non-empty paired paths, not independent NIfTI "
                "content, label-set, orientation, or spacing validation."
            ),
        },
        gaps,
    )


def validate_posthoc_dataset_manifest(path: Path) -> Dict[str, Any]:
    """Validate a de-identified 200-case NIfTI content manifest."""
    manifest = read_json(path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise EvidenceError("posthoc dataset manifest has an unsupported schema")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 200:
        raise EvidenceError("posthoc dataset manifest must contain 200 records")
    if int(manifest.get("patients", -1)) != 100 or int(
        manifest.get("cases", -1)
    ) != 200:
        raise EvidenceError("posthoc dataset manifest cardinality is invalid")

    expected_patients = {f"patient{i:03d}" for i in range(1, 101)}
    patient_counts: MutableMapping[str, int] = defaultdict(int)
    case_ids = set()
    relative_paths = set()
    file_hashes = set()
    foreground_totals = {"1": 0, "2": 0, "3": 0}
    for record in records:
        patient_id = str(record.get("patient_id", ""))
        case_id = str(record.get("case_id", ""))
        if patient_id not in expected_patients:
            raise EvidenceError(
                f"invalid patient identifier in posthoc manifest: {patient_id}"
            )
        if not case_id.startswith(patient_id + "_") or case_id in case_ids:
            raise EvidenceError(
                f"invalid or duplicate posthoc dataset case: {case_id}"
            )
        case_ids.add(case_id)
        patient_counts[patient_id] += 1
        for field in ("image_relative_path", "label_relative_path"):
            relative_path = str(record.get(field, ""))
            if not relative_path or Path(relative_path).is_absolute() or ".." in Path(
                relative_path
            ).parts:
                raise EvidenceError(
                    f"unsafe or empty {field} in posthoc manifest: {case_id}"
                )
            if relative_path in relative_paths:
                raise EvidenceError(
                    f"duplicate NIfTI path in posthoc manifest: {relative_path}"
                )
            relative_paths.add(relative_path)
        for field in ("image_sha256", "label_sha256"):
            digest = str(record.get(field, ""))
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise EvidenceError(f"invalid {field} in posthoc manifest: {case_id}")
            file_hashes.add(digest)
        if int(record.get("image_bytes", 0)) <= 0 or int(
            record.get("label_bytes", 0)
        ) <= 0:
            raise EvidenceError(f"non-positive file size in posthoc manifest: {case_id}")

        shape = record.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(int(dimension) <= 0 for dimension in shape)
        ):
            raise EvidenceError(f"invalid NIfTI shape in posthoc manifest: {case_id}")
        voxel_count = math.prod(int(dimension) for dimension in shape)
        for field in ("image_zooms", "label_zooms"):
            zooms = record.get(field)
            if (
                not isinstance(zooms, list)
                or len(zooms) != len(shape)
                or any(
                    not math.isfinite(float(value)) or float(value) <= 0
                    for value in zooms
                )
            ):
                raise EvidenceError(
                    f"invalid {field} in posthoc manifest: {case_id}"
                )
        if any(
            not close(float(a), float(b), 1e-5)
            for a, b in zip(record["image_zooms"], record["label_zooms"])
        ):
            raise EvidenceError(f"image/label spacing mismatch: {case_id}")
        for field in ("image_orientation", "label_orientation"):
            orientation = record.get(field)
            if (
                not isinstance(orientation, list)
                or len(orientation) != 3
                or any(
                    value not in {"L", "R", "A", "P", "S", "I"}
                    for value in orientation
                )
            ):
                raise EvidenceError(
                    f"invalid {field} in posthoc manifest: {case_id}"
                )
        if record["image_orientation"] != record["label_orientation"]:
            raise EvidenceError(f"image/label orientation mismatch: {case_id}")
        affine = record.get("affine")
        if (
            not isinstance(affine, list)
            or len(affine) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in affine)
            or any(
                not math.isfinite(float(value)) for row in affine for value in row
            )
        ):
            raise EvidenceError(f"invalid affine in posthoc manifest: {case_id}")
        image_min = float(record.get("image_min", math.nan))
        image_max = float(record.get("image_max", math.nan))
        if (
            not math.isfinite(image_min)
            or not math.isfinite(image_max)
            or image_min > image_max
        ):
            raise EvidenceError(f"invalid image range in posthoc manifest: {case_id}")
        label_counts = record.get("label_voxel_counts")
        if not isinstance(label_counts, dict) or set(label_counts) != {
            "0",
            "1",
            "2",
            "3",
        }:
            raise EvidenceError(f"invalid label-count fields: {case_id}")
        converted_counts = {
            key: int(value) for key, value in label_counts.items()
        }
        if any(value < 0 for value in converted_counts.values()) or sum(
            converted_counts.values()
        ) != voxel_count:
            raise EvidenceError(f"invalid label voxel counts: {case_id}")
        if sum(converted_counts[key] for key in ("1", "2", "3")) <= 0:
            raise EvidenceError(f"no foreground labels in posthoc manifest: {case_id}")
        for key in foreground_totals:
            foreground_totals[key] += converted_counts[key]

    if set(patient_counts) != expected_patients or any(
        count != 2 for count in patient_counts.values()
    ):
        raise EvidenceError(
            "posthoc dataset manifest must contain two cases per patient"
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    records_hash = hashlib.sha256(canonical).hexdigest()
    if records_hash != manifest.get("records_sha256"):
        raise EvidenceError("posthoc dataset records_sha256 is invalid")
    if any(total <= 0 for total in foreground_totals.values()):
        raise EvidenceError("posthoc dataset lacks one or more foreground classes")

    return {
        "status": "validated",
        "sha256": sha256(path),
        "records_sha256": records_hash,
        "patients": len(patient_counts),
        "cases": len(records),
        "unique_file_hashes": len(file_hashes),
        "foreground_voxel_totals": foreground_totals,
        "historical_dataset_snapshot_confirmed": bool(
            manifest.get("historical_dataset_snapshot_confirmed")
        ),
        "scope_note": (
            "The manifest checks the current 200 image/label pairs, including "
            "content hashes, finite values, geometry, and labels 0--3. It binds "
            "the historical run only when the snapshot is truthfully confirmed."
        ),
    }


def validate_source_lineage(
    evidence_dir: Path, aggregate_paths: Mapping[str, Path]
) -> Dict[str, Any]:
    """Validate the reconstructed Git/ZIP timeline without treating it as closure."""
    path = evidence_dir / LINEAGE_FILE
    if not path.is_file():
        return {
            "status": "not supplied",
            "scope": "optional reconstructed context; not a strict-closure artefact",
        }
    lineage = read_json(path)
    if int(lineage.get("schema_version", -1)) != 1:
        raise EvidenceError("historical_source_lineage.json has an unsupported schema")

    archive = lineage.get("supplied_archive", {})
    expected_aggregate_hashes = {
        "summary_metrics.csv": archive.get("summary_metrics_csv_sha256"),
        "summary_metrics.json": archive.get("summary_metrics_json_sha256"),
        "patient_split_seed42.json": archive.get("patient_split_sha256"),
        "data_discovery_report.json": archive.get("data_discovery_sha256"),
    }
    for name, expected_hash in expected_aggregate_hashes.items():
        if expected_hash != sha256(aggregate_paths[name]):
            raise EvidenceError(f"historical source lineage hash mismatch for {name}")

    current_sources = lineage.get("current_hardened_sources", {})
    for relative_path, expected_hash in current_sources.items():
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file() or sha256_git_text(source_path) != expected_hash:
            raise EvidenceError(
                f"historical source lineage current-source mismatch: {relative_path}"
            )

    git_objects_checked = False
    if (REPO_ROOT / ".git").exists():
        historical = lineage.get(
            "historical_sources_present_before_the_summary_timestamp", {}
        )
        for record in historical.values():
            commit = str(record["commit"])
            relative_path = str(record["path"])
            try:
                completed = subprocess.run(
                    ["git", "show", f"{commit}:{relative_path}"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise EvidenceError(
                    f"cannot verify historical Git object {commit}:{relative_path}"
                ) from exc
            if hashlib.sha256(completed.stdout).hexdigest() != record.get("sha256"):
                raise EvidenceError(
                    f"historical Git-object hash mismatch for {commit}:{relative_path}"
                )
        git_objects_checked = True

    return {
        "status": "validated reconstructed context",
        "sha256": sha256(path),
        "git_objects_checked": git_objects_checked,
        "scope": (
            "Temporal/source context only; the lineage file explicitly does not "
            "close missing run-level provenance."
        ),
    }


def bootstrap_patient_metrics(
    per_model: Mapping[str, Dict[str, Dict[str, float]]],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    patient_sets = [set(rows) for rows in per_model.values()]
    if not patient_sets or any(patients != patient_sets[0] for patients in patient_sets[1:]):
        raise EvidenceError("Per-case model files do not cover the same patients")
    patients = sorted(patient_sets[0], key=natural_patient_key)
    if len(patients) != 20:
        raise EvidenceError(f"Per-case files cover {len(patients)} patients, expected 20")
    rng = random.Random(seed)
    draws = [[rng.choice(patients) for _ in patients] for _ in range(samples)]

    intervals: Dict[str, Any] = {}
    for model, rows in per_model.items():
        point = sum(rows[p]["mean_dice"] for p in patients) / len(patients)
        boot = [sum(rows[p]["mean_dice"] for p in draw) / len(draw) for draw in draws]
        intervals[model] = {
            "patient_macro_mean_dice": point,
            "bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        }

    nano_rows = per_model["NanoMambaUNet"]
    paired: Dict[str, Any] = {}
    for model, rows in per_model.items():
        if model == "NanoMambaUNet":
            continue
        point = sum(nano_rows[p]["mean_dice"] - rows[p]["mean_dice"] for p in patients) / len(patients)
        boot = [
            sum(nano_rows[p]["mean_dice"] - rows[p]["mean_dice"] for p in draw) / len(draw)
            for draw in draws
        ]
        paired[f"NanoMambaUNet_minus_{model}"] = {
            "paired_patient_difference": point,
            "bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        }
    return {
        "method": "patient-level percentile bootstrap; post-hoc descriptive only",
        "seed": seed,
        "samples": samples,
        "model_intervals": intervals,
        "paired_differences": paired,
    }


def inspect_closure(
    evidence_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    val_patients: Iterable[str],
    bootstrap_samples: int,
    initial_gaps: Iterable[str] = (),
) -> Tuple[Dict[str, Any], List[str]]:
    gaps: List[str] = list(initial_gaps)
    closure: Dict[str, Any] = {}
    expected_val = set(val_patients)
    per_model: Dict[str, Dict[str, Dict[str, float]]] = {}
    recovered_case_diagnostics: Dict[str, Dict[str, Any]] = {}
    recovered_log_diagnostics: Dict[str, Dict[str, Any]] = {}
    summary_by_model = {row["model_name"]: row for row in summary_rows}

    for model in EXPECTED_MODELS:
        per_case = evidence_dir / f"per_case_{model}.csv"
        training_log = evidence_dir / f"training_log_{model}.csv"
        if not per_case.is_file():
            gaps.append(per_case.name)
        else:
            rows = read_csv(per_case)
            means = validate_per_case_rows(
                rows,
                per_case.name,
                expected_val,
                summary_by_model[model],
            )
            per_model[model] = means
            recovered_case_diagnostics[model] = per_case_diagnostics(rows)
        if not training_log.is_file():
            gaps.append(training_log.name)
        else:
            rows = read_csv(training_log)
            recovered_log_diagnostics[model] = validate_training_log(
                rows, training_log.name, summary_by_model[model]
            )

    closure["per_case_diagnostics"] = recovered_case_diagnostics
    closure["training_log_diagnostics"] = recovered_log_diagnostics

    checkpoint_manifest = evidence_dir / "checkpoint_manifest.json"
    if not checkpoint_manifest.is_file():
        gaps.append(checkpoint_manifest.name)
    else:
        manifest = read_json(checkpoint_manifest)
        if not manifest.get("all_expected_present"):
            gaps.append("checkpoint_manifest.json (one or more checkpoints missing)")
        if not manifest.get("historical_checkpoint_set_confirmed"):
            gaps.append("checkpoint_manifest.json (historical set not confirmed)")
        records = manifest.get("checkpoints", [])
        if not isinstance(records, list):
            raise EvidenceError("checkpoint_manifest.json checkpoints must be a list")
        records_by_model = {record.get("model_name"): record for record in records}
        if len(records_by_model) != len(records):
            raise EvidenceError("checkpoint manifest contains duplicate model records")
        if manifest.get("all_expected_present") and set(records_by_model) != set(EXPECTED_MODELS):
            raise EvidenceError("checkpoint manifest model set differs from the audited models")
        checkpoint_hashes = [str(record.get("sha256", "")) for record in records]
        if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
            raise EvidenceError("checkpoint manifest contains duplicate file hashes")
        for model, record in records_by_model.items():
            if model not in EXPECTED_MODELS:
                raise EvidenceError(f"unexpected checkpoint model record: {model}")
            if record.get("metadata_error"):
                raise EvidenceError(f"checkpoint metadata could not be read for {model}")
            if record.get("filename") != f"best_{model}.pth":
                raise EvidenceError(f"checkpoint filename is invalid for {model}")
            if int(record.get("bytes", 0)) <= 0 or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("sha256", ""))
            ):
                raise EvidenceError(f"checkpoint size/hash is invalid for {model}")
            if int(record.get("state_dict_tensors", 0)) <= 0 or int(
                record.get("state_dict_numel", 0)
            ) <= 0:
                raise EvidenceError(f"checkpoint state dictionary is empty for {model}")
            shape_hash = record.get("state_dict_shape_sha256")
            if shape_hash is not None and not re.fullmatch(
                r"[0-9a-f]{64}", str(shape_hash)
            ):
                raise EvidenceError(f"checkpoint shape hash is invalid for {model}")
            expected = summary_by_model[model]
            if int(record.get("epoch", -1)) != int(expected["best_epoch"]):
                raise EvidenceError(f"checkpoint epoch differs from the summary for {model}")
            if not close(
                float(record.get("val_mean_dice", -1)),
                float(expected["val_mean_dice"]),
                1e-8,
            ):
                raise EvidenceError(f"checkpoint Dice differs from the summary for {model}")
            config = record.get("config", {})
            if config.get("seed") != 42 or not close(
                float(config.get("val_fraction", -1)), 0.20
            ):
                raise EvidenceError(f"checkpoint split config is invalid for {model}")
            if tuple(config.get("spatial_size", ())) != (256, 256, 16):
                raise EvidenceError(f"checkpoint spatial size is invalid for {model}")
            if int(config.get("max_epochs", -1)) != MAX_EPOCHS:
                raise EvidenceError(f"checkpoint max_epochs is invalid for {model}")
        closure["checkpoint_manifest_summary"] = {
            "schema_version": manifest.get("schema_version"),
            "all_expected_present": bool(manifest.get("all_expected_present")),
            "historical_checkpoint_set_confirmed": bool(
                manifest.get("historical_checkpoint_set_confirmed")
            ),
            "models": sorted(records_by_model),
            "unique_checkpoint_hashes": len(set(checkpoint_hashes)),
        }

    environment_file = evidence_dir / "environment.json"
    if not environment_file.is_file():
        gaps.append(environment_file.name)
    else:
        environment = read_json(environment_file)
        environment_confirmed = environment.get(
            "historical_experiment_environment_confirmed"
        )
        if not environment_confirmed:
            gaps.append("environment.json (historical environment not confirmed)")
        if environment_confirmed:
            packages = environment.get("packages", {})
            if not packages.get("torch") or not packages.get("monai"):
                raise EvidenceError("environment.json lacks PyTorch or MONAI version evidence")
            runtime = environment.get("pytorch_runtime") or {}
            if not runtime.get("cuda_available") or not runtime.get("gpus"):
                raise EvidenceError("environment.json lacks a confirmed CUDA GPU runtime")
            if not any("RTX 4060" in str(name).upper() for name in runtime["gpus"]):
                raise EvidenceError(
                    "environment.json does not support the reported RTX 4060 hardware"
                )

    transcript = evidence_dir / "run_transcript.txt"
    if not transcript.is_file():
        gaps.append(transcript.name)
    else:
        transcript_text = transcript.read_text(encoding="utf-8-sig").strip()
        if (
            not transcript_text
            or "UNCONFIRMED" in transcript_text.upper()
            or "USER CONFIRMED" not in transcript_text.upper()
            or "21_RIGOROUS_EXPERIMENT_PIPELINE.PY" not in transcript_text.upper()
        ):
            gaps.append("run_transcript.txt (historical command not confirmed)")

    if len(per_model) == len(EXPECTED_MODELS):
        closure["posthoc_patient_bootstrap"] = bootstrap_patient_metrics(
            per_model, samples=bootstrap_samples, seed=20260820
        )
    else:
        closure["posthoc_patient_bootstrap"] = {
            "status": "not computed",
            "reason": "original per-case CSVs were not supplied",
        }
    closure["missing_artefacts"] = sorted(gaps)
    closure["status"] = "complete" if not gaps else "incomplete"
    return closure, gaps


def audit(evidence_dir: Path, bootstrap_samples: int) -> Dict[str, Any]:
    if bootstrap_samples <= 0:
        raise EvidenceError("bootstrap_samples must be positive")
    paths = require_files(evidence_dir)
    split_raw = read_json(paths["patient_split_seed42.json"])
    split_report = validate_split(split_raw)
    summary_rows, comparisons = validate_summary(
        read_csv(paths["summary_metrics.csv"]), read_json(paths["summary_metrics.json"])
    )
    discovery, discovery_gaps = validate_discovery(
        read_json(paths["data_discovery_report.json"])
    )
    dataset_manifest_path = evidence_dir / DATASET_MANIFEST_FILE
    if dataset_manifest_path.is_file():
        posthoc_dataset = validate_posthoc_dataset_manifest(dataset_manifest_path)
        discovery["posthoc_dataset_content_audit"] = posthoc_dataset
        discovery_gaps = [
            gap
            for gap in discovery_gaps
            if not gap.startswith("data_discovery_report.json")
        ]
        if not posthoc_dataset["historical_dataset_snapshot_confirmed"]:
            discovery_gaps.append(
                "posthoc_dataset_manifest.json (current 200-case content audited; "
                "historical dataset identity not confirmed)"
            )
    else:
        discovery["posthoc_dataset_content_audit"] = {
            "status": "not supplied"
        }
    lineage = validate_source_lineage(evidence_dir, paths)

    closure, gaps = inspect_closure(
        evidence_dir,
        summary_rows,
        split_raw["val_patients"],
        bootstrap_samples,
        initial_gaps=discovery_gaps,
    )
    return {
        "schema_version": 3,
        "evidence_directory": str(evidence_dir.resolve()),
        "aggregate_consistency_status": "pass",
        "strict_closure_status": "pass" if not gaps else "incomplete",
        "scope_note": (
            "A pass establishes internal consistency of the supplied aggregate files. "
            "Only strict closure attempts to bind them to original case rows, epoch logs, "
            "checkpoints, environment, and command evidence."
        ),
        "aggregate_sha256": {name: sha256(path) for name, path in paths.items()},
        "split": split_report,
        "discovery": discovery,
        "reconstructed_source_lineage": lineage,
        "summary_rows": summary_rows,
        "nano_comparisons": comparisons,
        "closure": closure,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--strict-closure",
        action="store_true",
        help="Fail unless all original closure artefacts are present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(args.evidence_dir, args.bootstrap_samples)
    except (EvidenceError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"P2 EVIDENCE AUDIT: FAIL\n{exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("P2 EVIDENCE AUDIT: AGGREGATE CONSISTENCY PASS")
    print(f"Strict closure status: {report['strict_closure_status']}")
    if report["closure"]["missing_artefacts"]:
        print("Missing closure artefacts:")
        for name in report["closure"]["missing_artefacts"]:
            print(f"  - {name}")
    if args.strict_closure and report["strict_closure_status"] != "pass":
        print("Strict closure requested: FAIL", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
