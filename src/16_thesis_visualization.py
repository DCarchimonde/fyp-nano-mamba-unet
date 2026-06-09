"""
Generate thesis figures for the Nano-Mamba U-Net project.

This script is intentionally separated from the training pipeline. It is used only to
produce thesis-ready visualizations from either saved model outputs or the final
summary metrics. The quantitative values below must match the final rigorous
patient-level validation experiment reported in the thesis.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    ToTensord,
)

from nano_mamba_core import SpatioTemporalMambaBottleneck


PROJECT_ROOT = Path(r"D:\AI_FYP")
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Candidate checkpoints are checked in order. The first existing path is used for
# qualitative prediction. Add a new path here if your final checkpoint has a
# different filename.
CANDIDATE_NANOMAMBA_CHECKPOINTS = [
    PROJECT_ROOT / "experiment_outputs" / "rigorous_patient_split" / "checkpoints" / "NanoMambaUNet_best.pth",
    PROJECT_ROOT / "experiment_outputs" / "rigorous_patient_split" / "NanoMambaUNet_best.pth",
    PROJECT_ROOT / "models" / "best_nanomamba_unet.pth",
]

DEFAULT_PATIENT_DIR = PROJECT_ROOT / "Data" / "ACDC" / "database" / "training" / "patient005"
DEFAULT_FRAME_NAME = "patient005_frame01.nii"

FINAL_RESULTS = [
    {"model": "3D U-Net", "mean_dsc": 80.83, "params_m": 4.81, "fps": 88.17},
    {"model": "Attention U-Net", "mean_dsc": 74.78, "params_m": 5.91, "fps": 22.41},
    {"model": "SegResNet16", "mean_dsc": 86.70, "params_m": 4.70, "fps": 25.50},
    {"model": "No-Mamba", "mean_dsc": 85.64, "params_m": 2.29, "fps": 28.80},
    {"model": "Half-Mamba", "mean_dsc": 84.95, "params_m": 1.64, "fps": 29.04},
    {"model": "Nano-Mamba", "mean_dsc": 84.78, "params_m": 1.46, "fps": 29.09},
]

CLASS_DICE_RESULTS = [
    {"model": "3D U-Net", "RV": 78.35, "MYO": 74.57, "LV": 89.59},
    {"model": "Nano-Mamba", "RV": 82.11, "MYO": 80.35, "LV": 91.88},
    {"model": "No-Mamba", "RV": 83.26, "MYO": 81.30, "LV": 92.37},
    {"model": "SegResNet16", "RV": 84.30, "MYO": 82.76, "LV": 93.03},
]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NanoMambaUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(
            DoubleConv(64, 128),
            SpatioTemporalMambaBottleneck(channels=128),
        )
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)


def _annotate_with_offsets(ax, xs, ys, labels, offsets):
    """Annotate scatter points using fixed offsets to avoid label overlap."""
    for x_value, y_value, label in zip(xs, ys, labels):
        dx, dy = offsets.get(label, (8, 8))
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.65", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.45"},
        )


def generate_tradeoff_figures() -> None:
    """Generate accuracy-efficiency figures from final validation results."""
    models = [item["model"] for item in FINAL_RESULTS]
    dice = [item["mean_dsc"] for item in FINAL_RESULTS]
    params = [item["params_m"] for item in FINAL_RESULTS]
    fps = [item["fps"] for item in FINAL_RESULTS]

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=220)
    ax.scatter(params, dice, s=70, zorder=3)
    parameter_offsets = {
        "Nano-Mamba": (-82, 10),
        "Half-Mamba": (-80, -18),
        "No-Mamba": (12, 12),
        "SegResNet16": (12, 8),
        "3D U-Net": (-62, -22),
        "Attention U-Net": (-92, 10),
    }
    _annotate_with_offsets(ax, params, dice, models, parameter_offsets)
    ax.set_xlabel("Trainable Parameters (Millions)")
    ax.set_ylabel("Validation Mean DSC (%)")
    ax.set_title("Accuracy-Parameter Trade-off")
    ax.set_xlim(1.0, 6.4)
    ax.set_ylim(73.5, 87.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "accuracy_parameter_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=220)
    ax.scatter(fps, dice, s=70, zorder=3)
    speed_offsets = {
        "Nano-Mamba": (-94, -20),
        "Half-Mamba": (14, -18),
        "No-Mamba": (14, 10),
        "SegResNet16": (14, 22),
        "3D U-Net": (-62, 8),
        "Attention U-Net": (12, 8),
    }
    _annotate_with_offsets(ax, fps, dice, models, speed_offsets)
    ax.set_xlabel("Inference Speed (FPS)")
    ax.set_ylabel("Validation Mean DSC (%)")
    ax.set_title("Accuracy-Speed Trade-off")
    ax.set_xlim(18, 93)
    ax.set_ylim(73.5, 87.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "accuracy_speed_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def generate_class_dice_figure() -> None:
    """Generate grouped class-wise Dice figure with non-overlapping labels."""
    classes = ["RV", "MYO", "LV"]
    models = [item["model"] for item in CLASS_DICE_RESULTS]
    x_positions = list(range(len(classes)))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=220)
    for model_idx, item in enumerate(CLASS_DICE_RESULTS):
        values = [item[class_name] for class_name in classes]
        bar_positions = [x + offsets[model_idx] for x in x_positions]
        bars = ax.bar(bar_positions, values, width=bar_width, label=models[model_idx])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.55,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Dice Similarity Coefficient (%)")
    ax.set_title("Class-specific Validation Dice Scores")
    ax.set_ylim(68, 98)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "class_specific_dice_scores.png", bbox_inches="tight")
    plt.close(fig)


def find_first_existing_checkpoint() -> Path | None:
    """Return the first Nano-Mamba checkpoint that exists locally."""
    for path in CANDIDATE_NANOMAMBA_CHECKPOINTS:
        if path.exists():
            return path
    return None


def _find_inner_nifti(container: Path) -> Path | None:
    """Handle the local ACDC layout where .nii paths may be directories."""
    if container.is_file() and container.stat().st_size > 0:
        return container
    if container.is_dir():
        matches = sorted(container.glob("*.nii"))
        if matches:
            return matches[0]
    return None


def can_generate_qualitative_figure() -> tuple[bool, str, Path | None]:
    """Check whether qualitative prediction can be generated on this machine."""
    checkpoint = find_first_existing_checkpoint()
    if checkpoint is None:
        checked = "\n".join(str(path) for path in CANDIDATE_NANOMAMBA_CHECKPOINTS)
        return False, f"No Nano-Mamba checkpoint found. Checked:\n{checked}", None

    img_container = DEFAULT_PATIENT_DIR / DEFAULT_FRAME_NAME
    label_container = DEFAULT_PATIENT_DIR / DEFAULT_FRAME_NAME.replace(".nii", "_gt.nii")
    if _find_inner_nifti(img_container) is None:
        return False, f"Image file/container not found or empty: {img_container}", checkpoint
    if _find_inner_nifti(label_container) is None:
        return False, f"Label file/container not found or empty: {label_container}", checkpoint
    return True, "Ready", checkpoint


def generate_qualitative_figure(
    model_path: Path,
    patient_dir: Path = DEFAULT_PATIENT_DIR,
    frame_name: str = DEFAULT_FRAME_NAME,
    slice_idx: int = 8,
) -> None:
    """Generate the qualitative segmentation comparison used in the thesis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NanoMambaUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_file = _find_inner_nifti(patient_dir / frame_name)
    label_file = _find_inner_nifti(patient_dir / frame_name.replace(".nii", "_gt.nii"))
    if img_file is None or label_file is None:
        raise FileNotFoundError("Could not locate the selected image/label NIfTI files.")

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    data_dict = transform({"image": str(img_file), "label": str(label_file)})
    img_tensor = data_dict["image"].unsqueeze(0).to(device)
    label_tensor = data_dict["label"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    img_np = img_tensor.cpu().numpy()[0, 0]
    label_np = label_tensor.cpu().numpy()[0, 0]

    cmap = ListedColormap(["#00000000", "blue", "lime", "red"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    axes[0].imshow(img_np[:, :, slice_idx], cmap="gray")
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(img_np[:, :, slice_idx], cmap="gray")
    axes[1].imshow(label_np[:, :, slice_idx], cmap=cmap, interpolation="nearest", alpha=0.45)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(img_np[:, :, slice_idx], cmap="gray")
    axes[2].imshow(pred_mask[:, :, slice_idx], cmap=cmap, interpolation="nearest", alpha=0.45)
    axes[2].set_title("Nano-Mamba Prediction")
    axes[2].axis("off")

    legend_patches = [
        mpatches.Patch(color="blue", label="Right Ventricle (RV)", alpha=0.6),
        mpatches.Patch(color="lime", label="Myocardium (MYO)", alpha=0.6),
        mpatches.Patch(color="red", label="Left Ventricle (LV)", alpha=0.6),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.04))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(FIGURE_DIR / "qualitative_result.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    generate_tradeoff_figures()
    generate_class_dice_figure()
    print(f"Trade-off and class-specific figures saved to: {FIGURE_DIR}")

    ready, message, checkpoint_path = can_generate_qualitative_figure()
    if ready and checkpoint_path is not None:
        generate_qualitative_figure(model_path=checkpoint_path)
        print(f"Qualitative figure saved to: {FIGURE_DIR / 'qualitative_result.png'}")
    else:
        print("Qualitative figure was skipped.")
        print(message)
        print("If the thesis already has D:\\AI_FYP\\figures\\qualitative_result.png, recompilation is still fine.")
