from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_motion_metrics import (  # noqa: E402
    analyse_lv_curve,
    bootstrap_mean_ci,
    circular_frame_distance,
    circular_moving_average,
    dice_by_class,
    functional_indices,
    parse_info_cfg_text,
    pearson_correlation,
    segmentation_frame_metrics,
)


class CardiacMotionMetricTests(unittest.TestCase):
    def test_info_cfg_uses_one_based_frames(self) -> None:
        parsed = parse_info_cfg_text(
            "ED: 1\nES: 12\nGroup: DCM\nHeight: 170\nNbFrame: 30\nWeight: 70\n"
        )
        self.assertEqual(parsed["ED"], 1)
        self.assertEqual(parsed["ES"], 12)
        self.assertEqual(parsed["NbFrame"], 30)
        self.assertEqual(parsed["Group"], "DCM")

    def test_info_cfg_rejects_out_of_range_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside"):
            parse_info_cfg_text("ED: 0\nES: 12\nNbFrame: 30\n")

    def test_circular_moving_average_wraps_cycle(self) -> None:
        values = [3.0, 0.0, 0.0, 0.0, 3.0]
        smoothed = circular_moving_average(values, window=3)
        np.testing.assert_allclose(smoothed, [2.0, 1.0, 0.0, 1.0, 2.0])

    def test_circular_phase_distance_handles_boundary(self) -> None:
        self.assertEqual(circular_frame_distance(1, 30, 30), 1)
        self.assertEqual(circular_frame_distance(4, 10, 30), 6)

    def test_curve_analysis_recovers_phase_and_function(self) -> None:
        curve = [120.0, 105.0, 80.0, 60.0, 80.0, 105.0]
        result = analyse_lv_curve(curve, 1, 4, smoothing_window=1)
        self.assertEqual(result["predicted_ed_frame"], 1)
        self.assertEqual(result["predicted_es_frame"], 4)
        self.assertEqual(result["ed_frame_error"], 0)
        self.assertEqual(result["es_frame_error"], 0)
        self.assertAlmostEqual(result["curve_sv_ml"], 60.0)
        self.assertAlmostEqual(result["curve_ef_percent"], 50.0)

    def test_dice_matches_historical_empty_rule(self) -> None:
        true = np.array([[[0, 1], [2, 3]]])
        pred = np.array([[[0, 1], [2, 0]]])
        result = dice_by_class(pred, true)
        self.assertEqual(result["dice_RV"], 1.0)
        self.assertEqual(result["dice_MYO"], 1.0)
        self.assertEqual(result["dice_LV"], 0.0)
        self.assertAlmostEqual(result["mean_dice"], 2.0 / 3.0)

        empty = dice_by_class(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
        self.assertEqual(empty["mean_dice"], 1.0)

    def test_physical_metrics_use_affine_and_voxel_volume(self) -> None:
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[0, 0, 0] = 3
        mask[2, 0, 0] = 3
        mask[1, 1, 0] = 2
        mask[0, 2, 0] = 1
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        result = segmentation_frame_metrics(mask, affine, voxel_volume_ml=0.024)
        self.assertAlmostEqual(result["lv_volume_ml"], 0.048)
        self.assertAlmostEqual(result["rv_volume_ml"], 0.024)
        self.assertAlmostEqual(result["lv_centroid_x_mm"], 2.0)
        self.assertAlmostEqual(result["lv_centroid_y_mm"], 0.0)
        self.assertTrue(np.isfinite(result["mean_myo_radius_mm"]))

    def test_functional_indices_and_correlation(self) -> None:
        result = functional_indices(150.0, 60.0)
        self.assertAlmostEqual(result["sv_ml"], 90.0)
        self.assertAlmostEqual(result["ef_percent"], 60.0)
        self.assertAlmostEqual(
            pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0
        )
        observed_failure = functional_indices(
            60.0, 90.0, require_physiological_order=False
        )
        self.assertAlmostEqual(observed_failure["sv_ml"], -30.0)
        self.assertAlmostEqual(observed_failure["ef_percent"], -50.0)

    def test_patient_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], replicates=500, seed=42)
        second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], replicates=500, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["mean"], 2.5)
        self.assertLessEqual(first["lower"], first["mean"])
        self.assertGreaterEqual(first["upper"], first["mean"])


if __name__ == "__main__":
    unittest.main()
