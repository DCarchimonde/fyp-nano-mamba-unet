"""Audit all 200 ACDC image/label pairs and write a de-identified manifest.

This is a post-hoc content audit.  It records file hashes, NIfTI geometry,
finite-value checks, and label counts without copying patient data.  The
historical confirmation flag must only be used when the caller has established
that the current dataset tree is the same snapshot used by the reported run.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EXPECTED_PATIENTS = {f"patient{index:03d}" for index in range(1, 101)}
ALLOWED_LABELS = {0, 1, 2, 3}


def natural_sort_key(text: str) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


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


def first_nifti(container: Path, want_label: bool) -> Optional[Path]:
    candidates: List[Path] = []
    if container.is_dir():
        for pattern in ("*.nii", "*.nii.gz", "**/*.nii", "**/*.nii.gz"):
            candidates.extend(
                Path(match)
                for match in glob.glob(str(container / pattern), recursive=True)
            )
    elif is_nonempty_file(container):
        candidates.append(container)
    else:
        return None

    accepted: List[Path] = []
    for candidate in candidates:
        name = candidate.name.lower()
        if "4d" in name:
            continue
        has_gt = "_gt" in name
        if want_label and not has_gt and not container.is_dir():
            continue
        if not want_label and has_gt:
            continue
        if is_nonempty_file(candidate):
            accepted.append(candidate)
    if not accepted:
        return None
    return sorted(set(accepted), key=lambda path: natural_sort_key(str(path)))[0]


def discover_pairs(data_dir: Path) -> List[Dict[str, Any]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ACDC data directory does not exist: {data_dir}")
    patient_dirs = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_dir() and re.fullmatch(r"patient\d{3}", path.name)
        ],
        key=lambda path: natural_sort_key(path.name),
    )
    discovered_patients = {path.name for path in patient_dirs}
    if discovered_patients != EXPECTED_PATIENTS:
        missing = sorted(EXPECTED_PATIENTS - discovered_patients)
        extra = sorted(discovered_patients - EXPECTED_PATIENTS)
        raise ValueError(
            "Expected patient001 through patient100 exactly; "
            f"missing={missing}, extra={extra}"
        )

    pairs: List[Dict[str, Any]] = []
    for patient_dir in patient_dirs:
        containers = sorted(
            [
                Path(match)
                for match in glob.glob(
                    str(patient_dir / "patient*_frame*.nii*")
                )
                if "_gt" not in Path(match).name.lower()
                and "4d" not in Path(match).name.lower()
            ],
            key=lambda path: natural_sort_key(path.name),
        )
        for image_container in containers:
            image_path = first_nifti(image_container, want_label=False)
            label_path = first_nifti(
                label_container_for(image_container), want_label=True
            )
            if image_path is None or label_path is None:
                continue
            frame_match = re.search(r"(frame\d+)", image_container.name)
            frame_id = (
                frame_match.group(1) if frame_match else image_container.stem
            )
            pairs.append(
                {
                    "patient_id": patient_dir.name,
                    "case_id": f"{patient_dir.name}_{frame_id}",
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )

    case_ids = [pair["case_id"] for pair in pairs]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case identifiers were discovered")
    counts = Counter(pair["patient_id"] for pair in pairs)
    if len(pairs) != 200 or set(counts) != EXPECTED_PATIENTS or any(
        count != 2 for count in counts.values()
    ):
        raise ValueError(
            "Expected exactly two labelled cases for each of 100 patients; "
            f"discovered {len(pairs)} cases"
        )
    return pairs


def relative_name(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def rounded(values: Iterable[float], digits: int = 8) -> List[float]:
    return [round(float(value), digits) for value in values]


def inspect_pair(pair: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("nibabel and NumPy are required") from exc

    image_path = Path(pair["image_path"])
    label_path = Path(pair["label_path"])
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_values = np.asanyarray(image.dataobj)
    label_values = np.asanyarray(label.dataobj)

    if tuple(image.shape) != tuple(label.shape):
        raise ValueError(f"Shape mismatch for {pair['case_id']}")
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D labelled frame for {pair['case_id']}")
    if not np.allclose(image.affine, label.affine, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Affine mismatch for {pair['case_id']}")
    if not np.isfinite(image_values).all() or not np.isfinite(label_values).all():
        raise ValueError(f"Non-finite NIfTI values for {pair['case_id']}")

    rounded_labels = np.rint(label_values)
    if not np.allclose(label_values, rounded_labels, rtol=0.0, atol=1e-6):
        raise ValueError(f"Non-integer label values for {pair['case_id']}")
    unique, counts = np.unique(rounded_labels.astype(np.int64), return_counts=True)
    label_counts = {int(key): int(value) for key, value in zip(unique, counts)}
    if not set(label_counts).issubset(ALLOWED_LABELS):
        raise ValueError(
            f"Unexpected labels for {pair['case_id']}: {sorted(label_counts)}"
        )
    if not any(label_counts.get(class_id, 0) > 0 for class_id in (1, 2, 3)):
        raise ValueError(f"No foreground label voxels for {pair['case_id']}")

    image_zooms = image.header.get_zooms()[: len(image.shape)]
    label_zooms = label.header.get_zooms()[: len(label.shape)]
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in image_zooms):
        raise ValueError(f"Invalid image spacing for {pair['case_id']}")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in label_zooms):
        raise ValueError(f"Invalid label spacing for {pair['case_id']}")
    if not np.allclose(image_zooms, label_zooms, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Spacing mismatch for {pair['case_id']}")
    image_orientation = list(nib.aff2axcodes(image.affine))
    label_orientation = list(nib.aff2axcodes(label.affine))
    if any(value is None for value in image_orientation + label_orientation):
        raise ValueError(f"Undefined NIfTI orientation for {pair['case_id']}")

    return {
        "patient_id": pair["patient_id"],
        "case_id": pair["case_id"],
        "image_relative_path": relative_name(image_path, data_dir),
        "label_relative_path": relative_name(label_path, data_dir),
        "image_bytes": image_path.stat().st_size,
        "label_bytes": label_path.stat().st_size,
        "image_sha256": sha256(image_path),
        "label_sha256": sha256(label_path),
        "shape": list(image.shape),
        "image_dtype": str(image.get_data_dtype()),
        "label_dtype": str(label.get_data_dtype()),
        "image_zooms": rounded(image_zooms),
        "label_zooms": rounded(label_zooms),
        "image_orientation": image_orientation,
        "label_orientation": label_orientation,
        "affine": [rounded(row) for row in image.affine.tolist()],
        "image_min": float(np.min(image_values)),
        "image_max": float(np.max(image_values)),
        "label_voxel_counts": {
            str(class_id): label_counts.get(class_id, 0)
            for class_id in sorted(ALLOWED_LABELS)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--historical-dataset-snapshot-confirmed", action="store_true"
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    pairs = discover_pairs(args.data_dir)
    records = [inspect_pair(pair, args.data_dir) for pair in pairs]
    records.sort(key=lambda row: natural_sort_key(row["case_id"]))
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "post-hoc de-identified ACDC NIfTI content manifest",
        "historical_dataset_snapshot_confirmed": (
            args.historical_dataset_snapshot_confirmed
        ),
        "notes": args.notes,
        "patients": 100,
        "cases": 200,
        "records_sha256": hashlib.sha256(canonical).hexdigest(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
