from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_PATH = ROOT / "src" / "23_spatiotemporal_cine_analysis.py"
SPEC = importlib.util.spec_from_file_location("cine_analysis_test_module", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
CINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CINE)

try:
    import torch
except ImportError:
    torch = None


class CineDiscoveryTests(unittest.TestCase):
    class FakeNifti:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class FakeNibabel:
        def __init__(self, shapes: dict[str, tuple[int, ...]]) -> None:
            self.shapes = shapes

        def load(self, path: str) -> "CineDiscoveryTests.FakeNifti":
            shape = self.shapes.get(path, (192, 192, 10))
            return CineDiscoveryTests.FakeNifti(shape)

    def test_nested_legacy_layout_discovers_endpoints_by_header_not_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patient = Path(directory) / "patient001"
            patient.mkdir()
            for frame in (1, 12):
                image_dir = patient / f"patient001_frame{frame:02d}.nii"
                label_dir = patient / f"patient001_frame{frame:02d}_gt.nii"
                image_dir.mkdir()
                label_dir.mkdir()
                (image_dir / f"image_{frame}.nii").write_bytes(b"image")
                (label_dir / f"label_{frame}.nii").write_bytes(b"label")

            cine_dir = patient / "patient001_4d.nii"
            cine_dir.mkdir()
            misleading_large = cine_dir / "large_3d.nii"
            actual_cine = cine_dir / "small_4d_payload.nii"
            misleading_large.write_bytes(b"123456789")
            actual_cine.write_bytes(b"123")
            nib = self.FakeNibabel(
                {
                    str(misleading_large): (192, 192, 10),
                    str(actual_cine): (192, 192, 10, 30),
                }
            )

            endpoints = CINE.discover_endpoint_pairs(patient)
            self.assertEqual(sorted(endpoints), [1, 12])
            self.assertEqual(endpoints[1]["image"].name, "image_1.nii")
            self.assertEqual(endpoints[12]["label"].name, "label_12.nii")
            self.assertEqual(
                CINE.discover_cine_nifti(patient, 30, nib).name,
                "small_4d_payload.nii",
            )

    def test_recursive_header_discovery_does_not_require_4d_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patient = Path(directory) / "patient002"
            nested = patient / "converted_cine.nii" / "payload"
            nested.mkdir(parents=True)
            cine = nested / "series_without_nifti_suffix.bin"
            cine.write_bytes(b"nifti")
            nib = self.FakeNibabel({str(cine): (232, 256, 12, 28)})
            self.assertEqual(CINE.discover_cine_nifti(patient, 28, nib), cine)

    def test_header_discovery_reports_observed_shapes_when_cine_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patient = Path(directory) / "patient003"
            patient.mkdir()
            endpoint = patient / "patient003_frame01.nii"
            endpoint.write_bytes(b"endpoint")
            nib = self.FakeNibabel({str(endpoint): (192, 192, 10)})
            with self.assertRaisesRegex(
                FileNotFoundError,
                r"shape=\(192, 192, 10\)",
            ):
                CINE.discover_cine_nifti(patient, 25, nib)

    def test_standard_nifti_suffix_mapping(self) -> None:
        self.assertEqual(
            CINE.label_container_for(Path("patient001_frame01.nii.gz")),
            Path("patient001_frame01_gt.nii.gz"),
        )
        self.assertEqual(
            CINE.label_container_for(Path("patient001_frame01.nii")),
            Path("patient001_frame01_gt.nii"),
        )

    def test_aggregate_keeps_paired_temporal_comparison(self) -> None:
        rows = []
        for patient_index, patient_id in enumerate(("patient002", "patient010")):
            for method_index, method in enumerate(CINE.METHODS):
                dice = 0.80 + 0.02 * patient_index + 0.01 * method_index
                ef_error = 8.0 - patient_index - method_index
                rows.append(
                    {
                        "patient_id": patient_id,
                        "method": method,
                        "ed_frame_error": patient_index,
                        "es_frame_error": patient_index + 1,
                        "reference_ef_percent": 55.0 + patient_index,
                        "annotated_ef_percent": 50.0 + patient_index + method_index,
                        "annotated_ef_abs_error_pp": ef_error,
                        "curve_ef_abs_error_pp": ef_error + 1.0,
                        "ed_resized_mean_dice": dice,
                        "es_resized_mean_dice": dice - 0.01,
                        "normalized_second_difference": 0.20 - 0.02 * method_index,
                        "peak_lv_centroid_displacement_mm": 4.0 + patient_index,
                        "myo_radius_ed_minus_es_mm": 2.0 + patient_index,
                    }
                )
        summary = CINE.aggregate_results(rows)
        self.assertAlmostEqual(
            summary["temporal_fusion_minus_framewise"]["endpoint_resized_dice"],
            0.01,
        )
        self.assertAlmostEqual(
            summary["temporal_fusion_minus_framewise"][
                "annotated_ef_mae_percentage_points"
            ],
            -1.0,
        )
        self.assertIn(
            "paired_patient_bootstrap_95_ci",
            summary["temporal_fusion_minus_framewise"],
        )

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_framewise_mask_is_preserved_before_float16_temporal_storage(self) -> None:
        probabilities = torch.zeros((3, 4, 1, 1, 1), dtype=torch.float16)
        probabilities[:, 0] = 0.6
        probabilities[:, 1] = 0.4
        saved_framewise = torch.ones((3, 1, 1, 1), dtype=torch.uint8)
        masks = CINE.prediction_masks(
            probabilities, saved_framewise, frame_index=1, torch=torch
        )
        self.assertEqual(int(masks["framewise"].item()), 1)
        self.assertEqual(int(masks["temporal_fusion"].item()), 0)


if __name__ == "__main__":
    unittest.main()
