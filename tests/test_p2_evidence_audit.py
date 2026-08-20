"""Fast, data-free regression tests for the P2 evidence validator."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "22_p2_evidence_audit.py"
EVIDENCE = REPO_ROOT / "evidence" / "rigorous_patient_split"
AGGREGATE_FILES = (
    "summary_metrics.csv",
    "summary_metrics.json",
    "patient_split_seed42.json",
    "data_discovery_report.json",
)


class P2EvidenceAuditTests(unittest.TestCase):
    def run_audit(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_aggregate_evidence_passes_with_explicit_scope(self) -> None:
        result = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AGGREGATE CONSISTENCY PASS", result.stdout)
        self.assertIn("incomplete", result.stdout)

    def test_strict_closure_fails_on_documented_missing_artefacts(self) -> None:
        result = self.run_audit("--strict-closure")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Strict closure requested: FAIL", result.stderr)

    def test_corrupted_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir)
            for name in AGGREGATE_FILES:
                shutil.copy2(EVIDENCE / name, copied / name)

            csv_path = copied / "summary_metrics.csv"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["val_mean_dice"] = "0.123456"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            result = self.run_audit("--evidence-dir", str(copied))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("mean Dice", result.stderr)

    def test_out_of_range_json_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir)
            for name in AGGREGATE_FILES:
                shutil.copy2(EVIDENCE / name, copied / name)

            json_path = copied / "summary_metrics.json"
            rows = json.loads(json_path.read_text(encoding="utf-8-sig"))
            rows[0]["val_dice_RV"] = 1.25
            json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

            result = self.run_audit("--evidence-dir", str(copied))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("CSV/JSON mismatch", result.stderr)

    def test_out_of_range_best_epoch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir)
            for name in AGGREGATE_FILES:
                shutil.copy2(EVIDENCE / name, copied / name)

            csv_path = copied / "summary_metrics.csv"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["best_epoch"] = "151"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            json_path = copied / "summary_metrics.json"
            json_rows = json.loads(json_path.read_text(encoding="utf-8-sig"))
            json_rows[0]["best_epoch"] = 151
            json_path.write_text(
                json.dumps(json_rows, indent=2) + "\n", encoding="utf-8"
            )

            result = self.run_audit("--evidence-dir", str(copied))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("best_epoch", result.stderr)

    def test_source_lineage_accepts_crlf_but_rejects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir)
            for name in AGGREGATE_FILES:
                shutil.copy2(EVIDENCE / name, copied / name)

            source = copied / "checkout_source.py"
            canonical = b"first = 1\nsecond = 2\n"
            source.write_bytes(canonical.replace(b"\n", b"\r\n"))

            archive_hash_keys = {
                "summary_metrics.csv": "summary_metrics_csv_sha256",
                "summary_metrics.json": "summary_metrics_json_sha256",
                "patient_split_seed42.json": "patient_split_sha256",
                "data_discovery_report.json": "data_discovery_sha256",
            }
            archive = {
                key: hashlib.sha256((copied / name).read_bytes()).hexdigest()
                for name, key in archive_hash_keys.items()
            }
            lineage = {
                "schema_version": 1,
                "supplied_archive": archive,
                "historical_sources_present_before_the_summary_timestamp": {},
                "current_hardened_sources": {
                    str(source): hashlib.sha256(canonical).hexdigest()
                },
            }
            lineage_path = copied / "historical_source_lineage.json"
            lineage_path.write_text(
                json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
            )

            converted = self.run_audit("--evidence-dir", str(copied))
            self.assertEqual(
                converted.returncode, 0, converted.stdout + converted.stderr
            )

            source.write_bytes(b"first = 1\r\nsecond = 999\r\n")
            changed = self.run_audit("--evidence-dir", str(copied))
            self.assertEqual(changed.returncode, 1, changed.stdout + changed.stderr)
            self.assertIn("current-source mismatch", changed.stderr)

    def test_recovered_closure_records_are_reported_and_empty_rule_did_not_fire(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audit.json"
            result = self.run_audit("--output", str(output))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            closure = report["closure"]
            case_diagnostics = closure["per_case_diagnostics"]
            log_diagnostics = closure["training_log_diagnostics"]
            self.assertEqual(len(case_diagnostics), 6)
            self.assertEqual(len(log_diagnostics), 6)
            self.assertTrue(
                all(
                    record["empty_empty_rule_trigger_count"] == 0
                    for record in case_diagnostics.values()
                )
            )
            self.assertTrue(
                all(record["rows"] == 150 for record in log_diagnostics.values())
            )
            self.assertEqual(
                closure["checkpoint_manifest_summary"]["unique_checkpoint_hashes"],
                6,
            )
            self.assertIn("model_intervals", closure["posthoc_patient_bootstrap"])

    def test_confirmed_posthoc_dataset_manifest_closes_full_manifest_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "evidence"
            shutil.copytree(EVIDENCE, copied)
            records = []
            for patient_index in range(1, 101):
                patient_id = f"patient{patient_index:03d}"
                for frame_index in (1, 2):
                    case_id = f"{patient_id}_frame{frame_index:02d}"
                    records.append(
                        {
                            "patient_id": patient_id,
                            "case_id": case_id,
                            "image_relative_path": f"{patient_id}/{case_id}.nii.gz",
                            "label_relative_path": f"{patient_id}/{case_id}_gt.nii.gz",
                            "image_bytes": 100,
                            "label_bytes": 50,
                            "image_sha256": hashlib.sha256(
                                (case_id + "image").encode()
                            ).hexdigest(),
                            "label_sha256": hashlib.sha256(
                                (case_id + "label").encode()
                            ).hexdigest(),
                            "shape": [1, 1, 4],
                            "image_dtype": "float32",
                            "label_dtype": "uint8",
                            "image_zooms": [1.0, 1.0, 1.0],
                            "label_zooms": [1.0, 1.0, 1.0],
                            "image_orientation": ["R", "A", "S"],
                            "label_orientation": ["R", "A", "S"],
                            "affine": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            "image_min": 0.0,
                            "image_max": 1.0,
                            "label_voxel_counts": {
                                "0": 1,
                                "1": 1,
                                "2": 1,
                                "3": 1,
                            },
                        }
                    )
            canonical = json.dumps(
                records, sort_keys=True, separators=(",", ":")
            ).encode()
            manifest = {
                "schema_version": 1,
                "historical_dataset_snapshot_confirmed": True,
                "patients": 100,
                "cases": 200,
                "records_sha256": hashlib.sha256(canonical).hexdigest(),
                "records": records,
            }
            (copied / "posthoc_dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = Path(temp_dir) / "report.json"
            result = self.run_audit(
                "--evidence-dir", str(copied), "--output", str(output)
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn(
                "full 200-case manifest unavailable",
                "\n".join(report["closure"]["missing_artefacts"]),
            )
            self.assertEqual(
                report["discovery"]["posthoc_dataset_content_audit"]["cases"],
                200,
            )


if __name__ == "__main__":
    unittest.main()
