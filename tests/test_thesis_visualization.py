"""Data-free checks for thesis figure inputs and qualitative split guards."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "16_thesis_visualization.py"
SUMMARY = REPO_ROOT / "evidence" / "rigorous_patient_split" / "summary_metrics.csv"
SPLIT = REPO_ROOT / "evidence" / "rigorous_patient_split" / "patient_split_seed42.json"
EVIDENCE = REPO_ROOT / "evidence" / "rigorous_patient_split"

SPEC = importlib.util.spec_from_file_location("thesis_visualization", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import failure is fatal.
    raise RuntimeError(f"Cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ThesisVisualizationTests(unittest.TestCase):
    def test_audited_summary_is_accepted(self) -> None:
        rows = MODULE.load_summary_metrics(SUMMARY)
        self.assertEqual(set(rows), set(MODULE.MODEL_ORDER))
        self.assertAlmostEqual(rows["NanoMambaUNet"]["val_mean_dice"], 0.847791743526856)

    def test_recovered_training_logs_match_summary(self) -> None:
        rows = MODULE.load_summary_metrics(SUMMARY)
        logs, paths = MODULE.load_training_logs(EVIDENCE, rows)
        self.assertEqual(set(logs), set(MODULE.MODEL_ORDER))
        self.assertEqual(set(paths), set(MODULE.MODEL_ORDER))
        self.assertTrue(all(len(model_rows) == 150 for model_rows in logs.values()))
        nano_best = max(logs["NanoMambaUNet"], key=lambda row: row["val_mean_dice"])
        self.assertEqual(nano_best["epoch"], 121)

    def test_non_finite_summary_efficiency_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "summary.csv"
            with SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["params_m"] = "nan"
            with copied.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "params_m must be positive"):
                MODULE.load_summary_metrics(copied)

    def test_non_finite_training_loss_is_rejected(self) -> None:
        rows = MODULE.load_summary_metrics(SUMMARY)
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir)
            for model_name in MODULE.MODEL_ORDER:
                shutil.copy2(
                    EVIDENCE / f"training_log_{model_name}.csv",
                    copied / f"training_log_{model_name}.csv",
                )
            path = copied / "training_log_UNet3D.csv"
            with path.open(newline="", encoding="utf-8-sig") as handle:
                log_rows = list(csv.DictReader(handle))
                fields = list(log_rows[0])
            log_rows[0]["train_loss"] = "nan"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(log_rows)
            with self.assertRaisesRegex(ValueError, "Invalid train_loss"):
                MODULE.load_training_logs(copied, rows)

    def test_validation_patient_is_accepted_and_training_patient_rejected(self) -> None:
        split = json.loads(SPLIT.read_text(encoding="utf-8-sig"))
        validation_patient = split["val_patients"][0]
        training_patient = split["train_patients"][0]
        accepted = MODULE._load_audited_split(SPLIT, validation_patient)
        self.assertEqual(accepted["seed"], 42)
        with self.assertRaisesRegex(ValueError, "training split"):
            MODULE._load_audited_split(SPLIT, training_patient)

    def test_tampered_split_is_rejected_before_qualitative_inference(self) -> None:
        split = json.loads(SPLIT.read_text(encoding="utf-8-sig"))
        split["seed"] = 7
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps(split), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed must be 42"):
                MODULE._load_audited_split(path, split["val_patients"][0])


if __name__ == "__main__":
    unittest.main()
