"""Regression tests for the thesis' high-risk methodology statements."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "src" / "21_rigorous_experiment_pipeline.py"
METHOD_PATH = (
    ROOT
    / "paper_write"
    / "Universiti_Malaya_Thesis_Template"
    / "sample-chap-methodology.tex"
)
LITREVIEW_PATH = (
    ROOT
    / "paper_write"
    / "Universiti_Malaya_Thesis_Template"
    / "sample-chap-litreview.tex"
)
SUMMARY_PATH = ROOT / "evidence" / "rigorous_patient_split" / "summary_metrics.csv"
CHECKPOINT_MANIFEST = (
    ROOT / "evidence" / "rigorous_patient_split" / "checkpoint_manifest.json"
)
EVIDENCE_DIR = ROOT / "evidence" / "rigorous_patient_split"
TORCH_MONAI_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("monai") is not None
)


def double_conv_parameters(in_channels: int, out_channels: int) -> int:
    # Two biased 3x3x3 convolutions plus two affine BatchNorm3d layers.
    return 27 * out_channels * (in_channels + out_channels) + 6 * out_channels


def transpose_conv_parameters(in_channels: int, out_channels: int) -> int:
    return in_channels * out_channels * 2 * 2 * 2 + out_channels


def analytical_nano_parameter_count() -> int:
    channels = 128
    state = 16
    gated_block = (
        (2 * channels * channels + 2 * channels)
        + (3 * channels + channels)
        + ((2 * state + 1) * channels + (2 * state + 1))
        + (channels * channels + channels)
    )
    return sum(
        [
            double_conv_parameters(1, 16),
            double_conv_parameters(16, 32),
            double_conv_parameters(32, 64),
            double_conv_parameters(64, 128),
            gated_block,
            transpose_conv_parameters(128, 64),
            double_conv_parameters(128, 64),
            transpose_conv_parameters(64, 32),
            double_conv_parameters(64, 32),
            transpose_conv_parameters(32, 16),
            double_conv_parameters(32, 16),
            16 * 4 + 4,
        ]
    )


class MethodologyIntegrityTests(unittest.TestCase):
    def test_seed_entry_point_contains_every_stated_control(self) -> None:
        source = PIPELINE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "set_seed"
        )
        function_source = ast.get_source_segment(source, node)
        self.assertIsNotNone(function_source)
        for required in (
            "random.seed(seed)",
            "np.random.seed(seed)",
            "torch.manual_seed(seed)",
            "torch.cuda.manual_seed_all(seed)",
            "torch.backends.cudnn.benchmark = False",
            "torch.backends.cudnn.deterministic = True",
        ):
            self.assertIn(required, function_source)
        self.assertGreaterEqual(source.count("num_workers=0"), 2)

    def test_executed_batch_sizes_match_thesis_contract(self) -> None:
        tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TRAIN_BATCH_SIZE"
                for target in node.targets
            )
        )
        batches = ast.literal_eval(assignment.value)
        self.assertEqual(batches["AttentionUNet"], 1)
        self.assertEqual(batches["SegResNet16"], 1)
        self.assertEqual(batches["NanoMambaUNet"], 2)

    def test_batch_one_models_completed_all_training_epochs(self) -> None:
        for model in ("AttentionUNet", "SegResNet16"):
            with (EVIDENCE_DIR / f"training_log_{model}.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 150)
            self.assertEqual([int(row["epoch"]) for row in rows], list(range(1, 151)))
            self.assertTrue(
                all(
                    math.isfinite(float(row["train_loss"]))
                    and math.isfinite(float(row["val_mean_dice"]))
                    for row in rows
                )
            )

    def test_exact_nano_parameter_and_buffer_counts(self) -> None:
        expected = analytical_nano_parameter_count()
        self.assertEqual(expected, 1_456_325)
        with SUMMARY_PATH.open(newline="", encoding="utf-8-sig") as handle:
            summary = {row["model_name"]: row for row in csv.DictReader(handle)}
        self.assertEqual(
            round(float(summary["NanoMambaUNet"]["params_m"]) * 1_000_000),
            expected,
        )
        manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8-sig"))
        checkpoint = next(
            row for row in manifest["checkpoints"] if row["model_name"] == "NanoMambaUNet"
        )
        # Fourteen BatchNorm3d layers each store running_mean, running_var, and
        # one num_batches_tracked scalar in the state dict.
        bn_channels = (16, 16, 32, 32, 64, 64, 128, 128, 64, 64, 32, 32, 16, 16)
        expected_buffers = sum(2 * channels + 1 for channels in bn_channels)
        self.assertEqual(expected_buffers, 1_422)
        self.assertEqual(checkpoint["state_dict_numel"] - expected, expected_buffers)

    def test_active_formula_sources_avoid_fragile_script_glyphs(self) -> None:
        methodology = METHOD_PATH.read_text(encoding="utf-8")
        literature = LITREVIEW_PATH.read_text(encoding="utf-8")
        self.assertNotIn("\\mathcal", methodology)
        self.assertNotIn("\\mathcal", literature)
        self.assertIn("\\operatorname{Attention}", literature)
        self.assertIn("\\operatorname{Concat}", literature)
        self.assertIn("L_{\\mathrm{total}}", methodology)
        self.assertIn("\\mathbf{P}_{\\mathrm{train}}", methodology)

    @unittest.skipUnless(
        TORCH_MONAI_AVAILABLE, "PyTorch and MONAI are required for model inspection"
    )
    def test_instantiated_normalization_and_parameter_contract(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        spec = importlib.util.spec_from_file_location("methodology_pipeline", PIPELINE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {PIPELINE_PATH}")
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        torch = pipeline.torch

        nano = pipeline.build_model("NanoMambaUNet")
        self.assertEqual(pipeline.count_params(nano), 1_456_325)

        attention = pipeline.build_model("AttentionUNet")
        segresnet = pipeline.build_model("SegResNet16")
        self.assertGreater(
            sum(isinstance(module, torch.nn.BatchNorm3d) for module in attention.modules()),
            0,
        )
        self.assertEqual(
            sum(isinstance(module, torch.nn.BatchNorm3d) for module in segresnet.modules()),
            0,
        )
        self.assertGreater(
            sum(isinstance(module, torch.nn.GroupNorm) for module in segresnet.modules()),
            0,
        )


if __name__ == "__main__":
    unittest.main()
