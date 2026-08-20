"""Data-free regression tests for the P2 closure collection helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENVIRONMENT = load_script(
    "p2_environment_capture", "scripts/p2_environment_capture.py"
)
DATASET = load_script("p2_dataset_manifest", "scripts/p2_dataset_manifest.py")


class P2CollectionToolTests(unittest.TestCase):
    def test_environment_text_redacts_url_credentials_and_user_homes(self) -> None:
        redact = ENVIRONMENT.redact_local_identity
        self.assertEqual(
            redact("pkg @ git+https://alice:token@example.test/repo"),
            "pkg @ git+https://<redacted>@example.test/repo",
        )
        self.assertEqual(
            redact("file:///C:/Users/Alice/build/pkg"),
            "file:///C:/Users/<redacted>/build/pkg",
        )
        self.assertEqual(
            redact("/home/alice/miniconda/bin/python"),
            "/home/<redacted>/miniconda/bin/python",
        )
        self.assertEqual(redact(r"D:\anaconda\envs\nanomamba"), r"D:\anaconda\envs\nanomamba")

    def test_dataset_label_container_mapping_handles_nifti_suffixes(self) -> None:
        self.assertEqual(
            DATASET.label_container_for(Path("patient001_frame01.nii.gz")),
            Path("patient001_frame01_gt.nii.gz"),
        )
        self.assertEqual(
            DATASET.label_container_for(Path("patient001_frame01.nii")),
            Path("patient001_frame01_gt.nii"),
        )

    def test_dataset_natural_sort_orders_frame_numbers_numerically(self) -> None:
        values = ["frame10", "frame2", "frame1"]
        self.assertEqual(
            sorted(values, key=DATASET.natural_sort_key),
            ["frame1", "frame2", "frame10"],
        )


if __name__ == "__main__":
    unittest.main()
