"""
Generate thesis figures for the Nano-Mamba U-Net project.

This script is intentionally separated from the training pipeline. It is used only to
produce thesis-ready visualizations from either saved model outputs or the final
summary metrics. The quantitative values below must match the final rigorous
patient-level validation experiment reported in the thesis.
"""

import os
import glob
from pathlib import Path

import numpy as np
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

FINAL_RESULTS = [
    {"model": "3D U-Net", "mean_dsc": 80.83, "params_m": 4.81, "fps": 88.17},
    {"model": "Attention U-Net", "mean_dsc": 74.78, "params_m": 5.91, "fps": 22.41},
    {"model": "SegResNet16", "mean_dsc": 86.70, "params_m": 4.70, "fps": 25.50},
    {"model": "No-Mamba", "mean_dsc": 85.64, "params_m": 2.29, "fps": 28.80},
    {"model": "Half-Mamba", "mean_dsc": 84.95, "params_m": 1.64, "fps": 29.04},
    {"model": "Nano-Mamba", "mean_dsc": 84.78, "params_m": 1.46, "fps": 29.09},
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


def generate_tradeoff_figures() -> None:
    """Generate accuracy-efficiency figures from final validation results."""
    models = [item["model"] for item in FINAL_RESULTS]
    dice = [item["mean_dsc"] for item in FINAL_RESULTS]
    params = [item["params_m"] for item in FINAL_RESULTS]
    fps = [item["fps"] for item in FINAL_RESULTS]

    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(params, dice, s=80)
    for x_value, y_value, label in zip(params, dice, models):
        plt.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Trainable Parameters (Millions)")
    plt.ylabel("Validation Mean DSC (%)")
    plt.title("Accuracy-Parameter Trade-off")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_parameter_tradeoff.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(fps, dice, s=80)
    for x_value, y_value, label in zip(fps, dice, models):
        plt.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Inference Speed (FPS)")
    plt.ylabel("Validation Mean DSC (%)")
    plt.title("Accuracy-Speed Trade-off")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_speed_tradeoff.png", bbox_inches="tight")
    plt.close()


def generate_qualitative_figure(
    model_path: Path = PROJECT_ROOT / "models" / "best_nanomamba_unet.pth",
    patient_dir: Path = PROJECT_ROOT / "Data" / "ACDC" / "database" / "training" / "patient005",
    frame_name: str = "patient005_frame01.nii",
    slice_idx: int = 8,
) -> None:
    """Generate the qualitative segmentation comparison used in the thesis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NanoMambaUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_container = glob.glob(str(patient_dir / frame_name))[0]
    label_container = glob.glob(str(patient_dir / frame_name.replace(".nii", "_gt.nii")))[0]
    img_file = glob.glob(os.path.join(img_container, "*.nii"))[0]
    label_file = glob.glob(os.path.join(label_container, "*.nii"))[0]

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    data_dict = transform({"image": img_file, "label": label_file})
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
    # Uncomment the line below only when the trained checkpoint and local ACDC data are available.
    # generate_qualitative_figure()
    print(f"Figures saved to: {FIGURE_DIR}")
