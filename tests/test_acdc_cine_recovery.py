from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_PATH = ROOT / "src" / "24_restore_missing_acdc_cine.py"
SPEC = importlib.util.spec_from_file_location("cine_recovery_test_module", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)
ANALYSIS = RECOVERY.load_cine_analysis_module()


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


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


class FakeImage:
    def __init__(
        self,
        values: np.ndarray,
        affine: np.ndarray,
        zooms: tuple[float, ...],
        qform_code: int = 0,
        sform_code: int = 0,
    ) -> None:
        self.dataobj = values
        self.shape = values.shape
        self.affine = affine
        self.header = FakeHeader(zooms, qform_code, sform_code)


class FakeNibabel:
    def __init__(self, images: dict[str, FakeImage]) -> None:
        self.images = images

    def load(self, path: str) -> FakeImage:
        return self.images[path]


class FakeAnalysis:
    spatial_grid_record = staticmethod(ANALYSIS.spatial_grid_record)

    @staticmethod
    def parse_info_cfg_text(text: str) -> dict[str, int]:
        return {"ED": 1, "ES": 2, "NbFrame": 3}

    @staticmethod
    def discover_endpoint_pairs(patient_dir: Path) -> dict[int, dict[str, Path]]:
        return {
            1: {"image": patient_dir / "ed.nii", "label": patient_dir / "ed_gt.nii"},
            2: {"image": patient_dir / "es.nii", "label": patient_dir / "es_gt.nii"},
        }

    @staticmethod
    def normalized_input_mae(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.mean(np.abs(first - second)))


class CineRecoveryTests(unittest.TestCase):
    def test_pinned_contract_covers_the_six_observed_missing_cines(self) -> None:
        expected = {
            "patient002",
            "patient042",
            "patient049",
            "patient066",
            "patient071",
            "patient073",
        }
        self.assertEqual(set(RECOVERY.PINNED_CINE_FILES), expected)
        self.assertEqual(
            sum(int(row["bytes"]) for row in RECOVERY.PINNED_CINE_FILES.values()),
            96_008_205,
        )
        for patient_id, row in RECOVERY.PINNED_CINE_FILES.items():
            self.assertRegex(str(row["sha256"]), r"^[0-9a-f]{64}$")
            self.assertIn(RECOVERY.PINNED_REVISION, RECOVERY.source_url(patient_id))

    def test_download_accepts_only_exact_size_and_sha256(self) -> None:
        payload = b"verified-cine"
        patient_id = "patient-test"
        original = RECOVERY.PINNED_CINE_FILES
        try:
            RECOVERY.PINNED_CINE_FILES = {
                patient_id: {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            }

            def opener(*args: object, **kwargs: object) -> FakeResponse:
                return FakeResponse(payload)

            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "cine.nii.gz"
                record = RECOVERY.download_pinned_file(
                    patient_id, destination, opener=opener
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(record["observed_bytes"], len(payload))
        finally:
            RECOVERY.PINNED_CINE_FILES = original

    def test_download_rejects_content_hash_mismatch(self) -> None:
        patient_id = "patient-test"
        original = RECOVERY.PINNED_CINE_FILES
        try:
            RECOVERY.PINNED_CINE_FILES = {
                patient_id: {
                    "bytes": 3,
                    "sha256": hashlib.sha256(b"good").hexdigest(),
                }
            }

            def opener(*args: object, **kwargs: object) -> FakeResponse:
                return FakeResponse(b"bad")

            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "cine.nii.gz"
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    RECOVERY.download_pinned_file(
                        patient_id, destination, opener=opener
                    )
        finally:
            RECOVERY.PINNED_CINE_FILES = original

    def test_identity_validation_matches_both_endpoint_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patient = Path(directory) / "patient002"
            patient.mkdir()
            (patient / "Info.cfg").write_text("ignored", encoding="utf-8")
            cine_path = patient / "patient002_4d.nii.gz"
            cine_affine = np.array(
                [
                    [-1.3671875, 0.0, 0.0, 157.91015625],
                    [0.0, 1.3671875, 0.0, -174.31640625],
                    [0.0, 0.0, 10.0, -45.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            endpoint_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
            spatial_zooms = (1.3671875, 1.3671875, 10.0)
            cine = np.zeros((2, 2, 1, 3), dtype=np.float32)
            cine[..., 0] = 1.0
            cine[..., 1] = 2.0
            images = {
                str(cine_path): FakeImage(
                    cine,
                    cine_affine,
                    spatial_zooms + (1.0,),
                ),
                str(patient / "ed.nii"): FakeImage(
                    cine[..., 0],
                    endpoint_affine,
                    spatial_zooms,
                    sform_code=2,
                ),
                str(patient / "ed_gt.nii"): FakeImage(
                    np.zeros_like(cine[..., 0]),
                    endpoint_affine,
                    spatial_zooms,
                    sform_code=2,
                ),
                str(patient / "es.nii"): FakeImage(
                    cine[..., 1],
                    endpoint_affine,
                    spatial_zooms,
                    sform_code=2,
                ),
                str(patient / "es_gt.nii"): FakeImage(
                    np.zeros_like(cine[..., 1]),
                    endpoint_affine,
                    spatial_zooms,
                    sform_code=2,
                ),
            }
            result = RECOVERY.validate_cine_identity(
                cine_path,
                patient,
                FakeAnalysis,
                FakeNibabel(images),
                endpoint_mae_tolerance=0.0,
            )
            self.assertEqual(result["cine_shape"], [2, 2, 1, 3])
            self.assertEqual(result["endpoint_normalized_mae"], {"ED": 0.0, "ES": 0.0})
            self.assertFalse(
                result["endpoint_grid_audit"]["ED"]["image_vs_cine"][
                    "affine_matches"
                ]
            )
            self.assertTrue(
                result["endpoint_grid_audit"]["ED"][
                    "label_vs_endpoint_image"
                ]["affine_matches"]
            )

            images[str(patient / "ed_gt.nii")] = FakeImage(
                np.zeros_like(cine[..., 0]),
                np.eye(4),
                spatial_zooms,
                sform_code=2,
            )
            with self.assertRaisesRegex(ValueError, "image/label affine differs"):
                RECOVERY.validate_cine_identity(
                    cine_path,
                    patient,
                    FakeAnalysis,
                    FakeNibabel(images),
                    endpoint_mae_tolerance=0.0,
                )
            images[str(patient / "ed_gt.nii")] = FakeImage(
                np.zeros_like(cine[..., 0]),
                endpoint_affine,
                spatial_zooms,
                sform_code=2,
            )

            images[str(patient / "es.nii")] = FakeImage(
                cine[..., 1] + 1.0,
                endpoint_affine,
                spatial_zooms,
                sform_code=2,
            )
            with self.assertRaisesRegex(ValueError, "does not match local endpoints"):
                RECOVERY.validate_cine_identity(
                    cine_path,
                    patient,
                    FakeAnalysis,
                    FakeNibabel(images),
                    endpoint_mae_tolerance=0.0,
                )


if __name__ == "__main__":
    unittest.main()
