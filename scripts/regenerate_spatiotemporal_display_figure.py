#!/usr/bin/env python3
"""Regenerate the thesis population cine figure from the sealed CSV evidence.

This script deliberately writes only the two convenience/display copies.  It
never changes evidence/spatiotemporal_cine/raw, whose manifest is immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


METHODS = ("framewise", "temporal_fusion")
TARGET = np.linspace(0.0, 1.0, 101)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def population_matrices(
    frame_rows: Sequence[Mapping[str, str]],
    patient_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    ed_lookup = {
        str(row["patient_id"]): int(row["reference_ed_frame"])
        for row in patient_rows
        if row["method"] == "temporal_fusion"
    }
    grouped: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in frame_rows:
        method = str(row["method"])
        if method in METHODS:
            grouped[(str(row["patient_id"]), method)].append(row)

    patient_ids = sorted(ed_lookup)
    if len(patient_ids) != 20:
        raise ValueError(f"Expected 20 patients, found {len(patient_ids)}")

    matrices: Dict[str, List[np.ndarray]] = {method: [] for method in METHODS}
    for patient_id in patient_ids:
        ed_frame = ed_lookup[patient_id]
        for method in METHODS:
            rows = sorted(
                grouped[(patient_id, method)], key=lambda row: int(row["frame"])
            )
            if not rows:
                raise ValueError(f"Missing {method} rows for {patient_id}")
            frames = [int(row["frame"]) for row in rows]
            if frames != list(range(1, len(rows) + 1)):
                raise ValueError(f"Non-contiguous frames for {patient_id}/{method}")
            volumes = np.asarray(
                [float(row["lv_volume_ml"]) for row in rows], dtype=np.float64
            )
            if not np.isfinite(volumes).all() or float(volumes.max()) <= 0.0:
                raise ValueError(f"Invalid LV volumes for {patient_id}/{method}")
            aligned = np.roll(volumes, -(ed_frame - 1))
            normalized = aligned / float(aligned.max())
            source = np.linspace(0.0, 1.0, len(normalized) + 1)
            periodic = np.append(normalized, normalized[0])
            matrices[method].append(np.interp(TARGET, source, periodic))

    return patient_ids, {
        method: np.asarray(curves, dtype=np.float64)
        for method, curves in matrices.items()
    }


def render_figure(matrices: Mapping[str, np.ndarray], output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano_mamba_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = TARGET * 100.0
    framewise = matrices["framewise"] * 100.0
    fusion = matrices["temporal_fusion"] * 100.0
    difference = fusion - framewise

    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.4),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.08},
    )
    styles = (
        (framewise, "#6F6F6F", "Frame-wise 3D", "-"),
        (fusion, "#C55A11", "Temporal probability fusion", "--"),
    )
    for matrix, color, label, linestyle in styles:
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        top.fill_between(
            x,
            np.maximum(0.0, mean - sd),
            mean + sd,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        top.plot(x, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)

    delta_mean = difference.mean(axis=0)
    delta_low, delta_high = np.quantile(difference, [0.025, 0.975], axis=0)
    bottom.fill_between(
        x, delta_low, delta_high, color="#F4B183", alpha=0.35, linewidth=0
    )
    bottom.plot(x, delta_mean, color="#C55A11", linewidth=1.6)
    bottom.axhline(0.0, color="#333333", linewidth=0.9, linestyle=":")

    top.set_ylabel("LV volume relative to\nmethod-specific patient maximum (%)")
    top.set_title("Validation-cohort full-cine LV volume trajectories (n=20)")
    top.grid(True, linestyle="--", alpha=0.25)
    top.legend(loc="best", fontsize=8)
    bottom.set_xlabel("Normalized cardiac cycle from reference ED (%)")
    bottom.set_ylabel("Fusion − frame-wise\n(percentage points)")
    bottom.grid(True, linestyle="--", alpha=0.25)
    bottom.text(
        0.995,
        0.04,
        "Band: patient 2.5th–97.5th percentiles",
        transform=bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.12)
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    raw = root / "evidence" / "spatiotemporal_cine" / "raw"
    frame_csv = raw / "frame_metrics.csv"
    patient_csv = raw / "patient_metrics.csv"
    display_output = root / "figures" / "spatiotemporal" / "population_lv_motion_curve.png"
    evidence_copy = root / "evidence" / "spatiotemporal_cine" / "figures" / "population_lv_motion_curve.png"
    provenance_path = root / "figures" / "spatiotemporal" / "population_lv_motion_curve_provenance.json"

    patient_ids, matrices = population_matrices(
        read_rows(frame_csv), read_rows(patient_csv)
    )
    render_figure(matrices, display_output)
    evidence_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(display_output, evidence_copy)
    provenance = {
        "schema_version": 1,
        "status": "display_figure_regenerated_from_sealed_evidence",
        "patients": len(patient_ids),
        "patient_ids": patient_ids,
        "methods": list(METHODS),
        "normalization": "reference-ED alignment; method-specific patient maximum; periodic interpolation to 101 points",
        "top_panel_band": "mean plus/minus one between-patient standard deviation",
        "difference_panel_band": "patient 2.5th to 97.5th percentiles at each normalized phase",
        "sources": {
            "frame_metrics.csv": sha256(frame_csv),
            "patient_metrics.csv": sha256(patient_csv),
        },
        "output_sha256": sha256(display_output),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {display_output}")
    print(f"Wrote {evidence_copy}")
    print(f"Wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
