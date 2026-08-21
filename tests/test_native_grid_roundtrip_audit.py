"""Data-free tests for the native-label round-trip diagnostic."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "26_native_grid_roundtrip_audit.py"
SPEC = importlib.util.spec_from_file_location("native_roundtrip_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class NativeGridRoundtripAuditTests(unittest.TestCase):
    def test_roundtrip_summary_requires_twenty_paired_patients(self) -> None:
        rows = []
        for patient_index in range(20):
            for phase_index, phase in enumerate(("ED", "ES")):
                score = 0.90 + 0.001 * patient_index + 0.0001 * phase_index
                rows.append(
                    {
                        "patient_id": f"patient{patient_index + 1:03d}",
                        "phase": phase,
                        "roundtrip_dice_RV": score,
                        "roundtrip_dice_MYO": score - 0.05,
                        "roundtrip_dice_LV": score + 0.02,
                        "roundtrip_mean_dice": score - 0.01,
                    }
                )
        summary = AUDIT.summarize_roundtrip_rows(rows)
        self.assertEqual(summary["patients"], 20)
        self.assertEqual(summary["endpoint_rows"], 40)
        self.assertAlmostEqual(summary["roundtrip_mean_dice"], 0.89955)
        self.assertAlmostEqual(summary["roundtrip_mean_loss_percentage_points"], 10.045)

    def test_observed_grid_gap_is_derived_from_all_endpoint_classes(self) -> None:
        fields = ["patient_id", "method"] + [
            f"{phase}_{grid}_dice_{class_name}"
            for phase in ("ed", "es")
            for grid in ("resized", "native")
            for class_name in AUDIT.CLASSES
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patient_metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for patient_index in range(20):
                    row = {"patient_id": f"patient{patient_index + 1:03d}", "method": "framewise"}
                    for phase in ("ed", "es"):
                        for class_name in AUDIT.CLASSES:
                            row[f"{phase}_resized_dice_{class_name}"] = 0.85
                            row[f"{phase}_native_dice_{class_name}"] = 0.78
                    writer.writerow(row)
            observed = AUDIT.load_observed_grid_gap(path)
        self.assertAlmostEqual(observed["framewise_resized_endpoint_mean_dice"], 0.85)
        self.assertAlmostEqual(observed["framewise_native_endpoint_mean_dice"], 0.78)
        self.assertAlmostEqual(observed["resized_minus_native_percentage_points"], 7.0)


if __name__ == "__main__":
    unittest.main()
