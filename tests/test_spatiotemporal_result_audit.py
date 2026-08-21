"""Regression tests for the independently sealed full-cine result bundle."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "25_spatiotemporal_result_audit.py"
RESULT_DIR = ROOT / "evidence" / "spatiotemporal_cine" / "raw"
SPEC = importlib.util.spec_from_file_location("spatiotemporal_result_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SpatiotemporalResultAuditTests(unittest.TestCase):
    def test_sealed_bundle_passes_full_independent_audit(self) -> None:
        report = AUDIT.audit_result_dir(RESULT_DIR, ROOT)
        self.assertEqual(report["status"], "independent_audit_pass")
        self.assertEqual(report["input_and_geometry"]["patients"], 20)
        self.assertEqual(report["input_and_geometry"]["cine_frames"], 550)
        self.assertTrue(report["source_lineage"]["checked"])

        paired = report["derived_results"][
            "paired_temporal_fusion_minus_framewise"
        ]
        self.assertFalse(paired["endpoint_resized_dice"]["interval_excludes_zero"])
        self.assertFalse(
            paired["annotated_ef_abs_error_pp"]["interval_excludes_zero"]
        )
        self.assertLess(paired["curve_smoothness"]["upper"], 0.0)

    def test_committed_audit_report_matches_recomputation(self) -> None:
        expected = json.loads(
            (RESULT_DIR.parent / "INDEPENDENT_AUDIT.json").read_text(
                encoding="utf-8"
            )
        )
        observed = AUDIT.audit_result_dir(RESULT_DIR, ROOT)

        self.assertEqual(
            observed["result_directory"], "evidence/spatiotemporal_cine/raw"
        )
        self.assertEqual(observed, expected)

    def test_circular_phase_distance_handles_cycle_boundary(self) -> None:
        self.assertEqual(AUDIT.circular_frame_distance(1, 30, 30), 1)
        self.assertEqual(AUDIT.circular_frame_distance(4, 10, 30), 6)


if __name__ == "__main__":
    unittest.main()
