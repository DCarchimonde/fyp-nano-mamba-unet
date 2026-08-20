"""Checkpoint-compatibility and shape tests for the Nano-Mamba core.

The executable tests are skipped cleanly when PyTorch is not installed. This
keeps the default evidence audit data-free while preserving a concrete smoke
test for the experiment environment.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPO_ROOT / "src" / "nano_mamba_core.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class NanoMambaCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("nano_mamba_core_test", CORE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {CORE_PATH}")
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_legacy_aliases_preserve_class_identity(self) -> None:
        self.assertIs(
            self.module.PurePyTorchMambaBlock,
            self.module.MambaInspiredGatedSequenceBlock,
        )
        self.assertIs(
            self.module.SpatioTemporalMambaBottleneck,
            self.module.MambaInspiredBottleneck,
        )

    def test_bottleneck_preserves_shape_and_state_dict_keys(self) -> None:
        torch = self.module.torch
        torch.manual_seed(42)
        model = self.module.MambaInspiredBottleneck(channels=8)
        legacy = self.module.SpatioTemporalMambaBottleneck(channels=8)
        self.assertEqual(tuple(model.state_dict()), tuple(legacy.state_dict()))

        sample = torch.randn(2, 8, 4, 4, 2)
        output = model(sample)
        self.assertEqual(output.shape, sample.shape)

    def test_unused_projection_channels_have_zero_output_gradient(self) -> None:
        torch = self.module.torch
        torch.manual_seed(42)
        block = self.module.MambaInspiredGatedSequenceBlock(d_model=8, d_state=2)
        block(torch.randn(2, 6, 8)).sum().backward()
        gradient = block.x_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient[0].abs().sum()), 0.0)
        self.assertEqual(float(gradient[1:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
