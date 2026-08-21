from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


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
    class FakeHeader:
        def __init__(
            self,
            zooms: tuple[float, ...],
            qform_code: int = 0,
            sform_code: int = 0,
        ) -> None:
            self.zooms = zooms
            self.form_codes = {
                "qform_code": np.asarray(qform_code),
                "sform_code": np.asarray(sform_code),
            }

        def get_zooms(self) -> tuple[float, ...]:
            return self.zooms

        def __getitem__(self, field: str) -> np.ndarray:
            return self.form_codes[field]

    class FakeGeometryImage:
        def __init__(
            self,
            shape: tuple[int, ...],
            affine: np.ndarray,
            zooms: tuple[float, ...],
            qform_code: int = 0,
            sform_code: int = 0,
        ) -> None:
            self.shape = shape
            self.affine = affine
            self.header = CineDiscoveryTests.FakeHeader(
                zooms, qform_code, sform_code
            )

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

    def test_acdc_affine_metadata_mismatch_is_audited_not_misclassified(self) -> None:
        zooms = (1.3671875, 1.3671875, 10.0)
        cine = self.FakeGeometryImage(
            (232, 256, 10, 30),
            np.array(
                [
                    [-zooms[0], 0.0, 0.0, 157.91015625],
                    [0.0, zooms[1], 0.0, -174.31640625],
                    [0.0, 0.0, zooms[2], -45.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            zooms + (1.0,),
        )
        endpoint = self.FakeGeometryImage(
            (232, 256, 10),
            np.diag([-1.0, -1.0, 1.0, 1.0]),
            zooms,
            sform_code=2,
        )
        record = CINE.spatial_grid_record(
            cine, endpoint, "patient002 cine", "patient002 ED endpoint"
        )
        self.assertFalse(record["affine_matches"])
        self.assertEqual(record["reference_zooms_mm"], list(zooms))
        self.assertEqual(record["candidate_zooms_mm"], list(zooms))

    def test_grid_record_rejects_a_real_voxel_size_mismatch(self) -> None:
        reference = self.FakeGeometryImage(
            (32, 32, 4, 20), np.eye(4), (1.4, 1.4, 8.0, 1.0)
        )
        candidate = self.FakeGeometryImage(
            (32, 32, 4), np.eye(4), (1.4, 1.4, 10.0)
        )
        with self.assertRaisesRegex(ValueError, "voxel sizes"):
            CINE.spatial_grid_record(
                reference, candidate, "reference cine", "endpoint image"
            )

    def test_physical_metric_uses_header_zooms_not_inconsistent_affine_scale(self) -> None:
        zooms = (1.3671875, 1.3671875, 10.0)
        image = self.FakeGeometryImage(
            (232, 256, 10, 30),
            np.diag([-1.0, -1.0, 1.0, 1.0]),
            zooms + (1.0,),
            sform_code=2,
        )
        metric_affine, voxel_volume_ml, audit = CINE.physical_metric_geometry(
            image
        )
        self.assertTrue(
            np.allclose(np.linalg.norm(metric_affine[:3, :3], axis=0), zooms)
        )
        self.assertAlmostEqual(voxel_volume_ml, np.prod(zooms) / 1000.0)
        self.assertFalse(audit["raw_affine_scales_match_header_zooms"])

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

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_temporal_probability_fusion_wraps_both_cycle_boundaries(self) -> None:
        probabilities = torch.zeros((4, 4, 1, 1, 1), dtype=torch.float32)
        # Small non-wrapped class-0 evidence at every frame.
        probabilities[:, 0] = 0.2
        # The last frame must be the previous neighbour of frame 0.
        probabilities[-1, 1] = 1.0
        masks = CINE.prediction_masks(
            probabilities,
            torch.zeros((4, 1, 1, 1), dtype=torch.uint8),
            frame_index=0,
            torch=torch,
        )
        self.assertEqual(int(masks["temporal_fusion"].item()), 1)

        # The first frame must be the following neighbour of the last frame.
        probabilities.zero_()
        probabilities[:, 0] = 0.2
        probabilities[0, 1] = 1.0
        masks = CINE.prediction_masks(
            probabilities,
            torch.zeros((4, 1, 1, 1), dtype=torch.uint8),
            frame_index=3,
            torch=torch,
        )
        self.assertEqual(int(masks["temporal_fusion"].item()), 1)

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_native_mask_restoration_preserves_axis_order_and_classes(self) -> None:
        source = torch.zeros((2, 3, 4), dtype=torch.uint8)
        source[0, 0, 0] = 1
        source[-1, -1, -1] = 3
        restored = CINE.resize_mask_to_native(
            source,
            native_shape=(5, 7, 9),
            torch=torch,
            functional=torch.nn.functional,
        )
        self.assertEqual(restored.shape, (5, 7, 9))
        self.assertEqual(int(restored[0, 0, 0]), 1)
        self.assertEqual(int(restored[-1, -1, -1]), 3)
        self.assertTrue(set(np.unique(restored)).issubset({0, 1, 3}))


if __name__ == "__main__":
    unittest.main()
