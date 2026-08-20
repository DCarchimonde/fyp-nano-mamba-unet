"""Implementation-faithful Nano-Mamba bottleneck components.

The historical experiment used a compact, Mamba-inspired gated sequence block.
It does not implement the Mamba selective state-space algorithm or selective
scan. The aliases at the bottom preserve the class names used by the original
training and visualization scripts, and all parameter names and tensor shapes
remain checkpoint-compatible with the reported experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaInspiredGatedSequenceBlock(nn.Module):
    """Apply depthwise sequence mixing and input-dependent scalar gating.

    For an input ``x`` with shape ``[batch, sequence, channels]``, the block:

    1. projects and splits ``x`` into a content branch and a modulation branch;
    2. applies a symmetric, depthwise 1D convolution and SiLU to the content;
    3. derives one sigmoid gate per token from the first ``x_proj`` output;
    4. modulates the content by the gate and the SiLU modulation branch; and
    5. projects the result back to the model dimension.

    ``x_proj`` intentionally retains its historical ``2 * d_state + 1`` output
    width so that reported checkpoints load without modification. Only its
    first output channel was used by the executed graph; the remaining legacy
    projection channels have no path to the output. Consequently, the reported
    parameter count includes parameters that did not affect predictions.
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor with the same ``[B, L, C]`` shape as ``x``."""
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        gate = torch.sigmoid(self.x_proj(x_branch)[..., 0:1])
        gated_content = x_branch * gate * F.silu(z_branch)
        return self.out_proj(gated_content)


class MambaInspiredBottleneck(nn.Module):
    """Mix a raster-flattened 3D feature grid and add a residual connection.

    Input shape is ``[batch, channels, height, width, depth]``. The depth axis
    is anatomical slice depth, not time. Flattening ``H * W * D`` therefore
    creates a spatial token sequence; this block does not perform temporal
    tracking across cine frames.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mamba_block = MambaInspiredGatedSequenceBlock(d_model=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor with the same ``[B, C, H, W, D]`` shape as ``x``."""
        batch, channels, height, width, depth = x.shape
        flattened = x.view(batch, channels, -1).transpose(1, 2)
        mixed = self.mamba_block(flattened)
        restored = mixed.transpose(1, 2).view(
            batch,
            channels,
            height,
            width,
            depth,
        )
        return restored + x


# Backward-compatible imports used by the historical scripts. These aliases do
# not add state-dict prefixes or change any learned parameter name.
PurePyTorchMambaBlock = MambaInspiredGatedSequenceBlock
SpatioTemporalMambaBottleneck = MambaInspiredBottleneck


if __name__ == "__main__":
    torch.manual_seed(42)
    sample = torch.randn(2, 256, 16, 16, 4)
    bottleneck = MambaInspiredBottleneck(channels=256)
    result = bottleneck(sample)
    if result.shape != sample.shape:
        raise RuntimeError(
            f"Shape mismatch: expected {tuple(sample.shape)}, got {tuple(result.shape)}"
        )
    print(f"Smoke test passed with output shape {tuple(result.shape)}")
