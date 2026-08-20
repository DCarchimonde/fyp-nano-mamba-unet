"""Generate thesis figures from audited experiment evidence.

Quantitative figures are derived directly from ``summary_metrics.csv``. A
qualitative figure is optional and deliberately fails closed unless the caller
provides the rigorous checkpoint, audited split, validation patient, and local
ACDC data directory. No legacy checkpoint or training patient is accepted as a
fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SUMMARY_CSV = (
    REPOSITORY_ROOT / "evidence" / "rigorous_patient_split" / "summary_metrics.csv"
)
DEFAULT_SPLIT_JSON = (
    REPOSITORY_ROOT
    / "evidence"
    / "rigorous_patient_split"
    / "patient_split_seed42.json"
)
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "figures"
MAIN_RESULT_PIPELINE = REPOSITORY_ROOT / "src" / "21_rigorous_experiment_pipeline.py"

MODEL_ORDER = [
    "UNet3D",
    "AttentionUNet",
    "SegResNet16",
    "Ablation_NoMamba_UNet",
    "Ablation_HalfMamba_UNet",
    "NanoMambaUNet",
]
DISPLAY_NAMES = {
    "UNet3D": "3D U-Net",
    "AttentionUNet": "Attention U-Net",
    "SegResNet16": "SegResNet16",
    "Ablation_NoMamba_UNet": "No-Mamba",
    "Ablation_HalfMamba_UNet": "Half-Mamba",
    "NanoMambaUNet": "Nano-Mamba",
}
CLASS_FIGURE_MODELS = [
    "UNet3D",
    "NanoMambaUNet",
    "Ablation_NoMamba_UNet",
    "SegResNet16",
]
NUMERIC_FIELDS = [
    "val_dice_RV",
    "val_dice_MYO",
    "val_dice_LV",
    "val_mean_dice",
    "params_m",
    "fps",
    "latency_ms",
]
EXPECTED_PATIENTS = {f"patient{i:03d}" for i in range(1, 101)}


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary_metrics(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate the result rows used by every quantitative figure."""
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"model_name", "best_epoch", *NUMERIC_FIELDS}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Summary CSV is missing columns: " + ", ".join(sorted(missing_columns))
            )

        for raw_row in reader:
            model_name = raw_row["model_name"].strip()
            if model_name in rows:
                raise ValueError(f"Duplicate model row in summary CSV: {model_name}")
            parsed: dict[str, Any] = {
                "model_name": model_name,
                "best_epoch": int(raw_row["best_epoch"]),
            }
            for field in NUMERIC_FIELDS:
                parsed[field] = float(raw_row[field])
            rows[model_name] = parsed

    missing_models = set(MODEL_ORDER).difference(rows)
    if missing_models:
        raise ValueError(
            "Summary CSV is missing required models: "
            + ", ".join(sorted(missing_models))
        )

    for model_name in MODEL_ORDER:
        row = rows[model_name]
        class_mean = (
            row["val_dice_RV"] + row["val_dice_MYO"] + row["val_dice_LV"]
        ) / 3.0
        if abs(class_mean - row["val_mean_dice"]) > 1e-9:
            raise ValueError(f"Class mean mismatch for {model_name}")
        for field in ("val_dice_RV", "val_dice_MYO", "val_dice_LV", "val_mean_dice"):
            if not 0.0 <= row[field] <= 1.0:
                raise ValueError(f"{field} is outside [0, 1] for {model_name}")
        for field in ("params_m", "fps", "latency_ms"):
            if row[field] <= 0.0:
                raise ValueError(f"{field} must be positive for {model_name}")
        reciprocal_fps = 1000.0 / row["latency_ms"]
        if abs(reciprocal_fps - row["fps"]) > 1e-6:
            raise ValueError(f"FPS/latency mismatch for {model_name}")
    return rows


def _load_pyplot():
    """Load Matplotlib with a headless backend suitable for clean builds."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "nano_mamba_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _annotate_points(ax, xs, ys, labels, offsets) -> None:
    for x_value, y_value, label in zip(xs, ys, labels):
        dx, dy = offsets.get(label, (8, 8))
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "0.65",
                "alpha": 0.92,
            },
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.45"},
        )


def generate_quantitative_figures(
    rows: dict[str, dict[str, Any]], output_dir: Path
) -> list[Path]:
    """Generate the three quantitative thesis figures from validated rows."""
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [DISPLAY_NAMES[name] for name in MODEL_ORDER]
    dice = [rows[name]["val_mean_dice"] * 100.0 for name in MODEL_ORDER]
    params = [rows[name]["params_m"] for name in MODEL_ORDER]
    fps = [rows[name]["fps"] for name in MODEL_ORDER]
    colors = ["#4472C4" if name != "NanoMambaUNet" else "#C55A11" for name in MODEL_ORDER]
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=220)
    ax.scatter(params, dice, s=72, c=colors, edgecolors="white", linewidths=0.7, zorder=3)
    _annotate_points(
        ax,
        params,
        dice,
        labels,
        {
            "Nano-Mamba": (-82, 10),
            "Half-Mamba": (12, -26),
            "No-Mamba": (12, 12),
            "SegResNet16": (12, 8),
            "3D U-Net": (-62, -22),
            "Attention U-Net": (-92, 10),
        },
    )
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Held-out validation mean DSC (%)")
    ax.set_title("Accuracy–parameter trade-off")
    ax.set_xlim(1.0, 6.4)
    ax.set_ylim(73.5, 87.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    parameter_path = output_dir / "accuracy_parameter_tradeoff.png"
    fig.savefig(parameter_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(parameter_path)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=220)
    ax.scatter(fps, dice, s=72, c=colors, edgecolors="white", linewidths=0.7, zorder=3)
    _annotate_points(
        ax,
        fps,
        dice,
        labels,
        {
            "Nano-Mamba": (-94, -20),
            "Half-Mamba": (14, -18),
            "No-Mamba": (14, 10),
            "SegResNet16": (14, 22),
            "3D U-Net": (-62, 8),
            "Attention U-Net": (12, 8),
        },
    )
    ax.set_xlabel("Inference speed (FPS; batch size 1)")
    ax.set_ylabel("Held-out validation mean DSC (%)")
    ax.set_title("Accuracy–speed trade-off")
    ax.set_xlim(18, 93)
    ax.set_ylim(73.5, 87.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    speed_path = output_dir / "accuracy_speed_tradeoff.png"
    fig.savefig(speed_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(speed_path)

    classes = ["RV", "MYO", "LV"]
    fields = ["val_dice_RV", "val_dice_MYO", "val_dice_LV"]
    x_positions = list(range(len(classes)))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    fig, ax = plt.subplots(figsize=(9.4, 5.8), dpi=220)
    for model_index, model_name in enumerate(CLASS_FIGURE_MODELS):
        values = [rows[model_name][field] * 100.0 for field in fields]
        positions = [position + offsets[model_index] for position in x_positions]
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=DISPLAY_NAMES[model_name],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.45,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Held-out validation DSC (%)")
    ax.set_title("Class-specific validation Dice scores")
    ax.set_ylim(68, 98)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    class_path = output_dir / "class_specific_dice_scores.png"
    fig.savefig(class_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(class_path)
    return outputs


def write_quantitative_manifest(
    summary_csv: Path,
    rows: dict[str, dict[str, Any]],
    outputs: list[Path],
    output_dir: Path,
) -> Path:
    """Write hashes that tie quantitative plots to their source table and code."""
    try:
        summary_display_path = str(summary_csv.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        summary_display_path = str(summary_csv.resolve())
    manifest = {
        "artifact_type": "quantitative_thesis_figures",
        "main_result_pipeline": str(MAIN_RESULT_PIPELINE.relative_to(REPOSITORY_ROOT)),
        "main_result_pipeline_sha256": sha256_file(MAIN_RESULT_PIPELINE),
        "summary_metrics": summary_display_path,
        "summary_metrics_sha256": sha256_file(summary_csv),
        "generator": str(SCRIPT_PATH.relative_to(REPOSITORY_ROOT)),
        "generator_sha256": sha256_file(SCRIPT_PATH),
        "models": MODEL_ORDER,
        "results": [rows[name] for name in MODEL_ORDER],
        "outputs": {path.name: sha256_file(path) for path in outputs},
        "provenance_note": "Plot values are loaded from the audited CSV; no result values are embedded in the plotting code.",
    }
    manifest_path = output_dir / "quantitative_figure_provenance.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest_path


def _label_container_for(image_container: Path) -> Path:
    text = str(image_container)
    if text.endswith(".nii.gz"):
        return Path(text[:-7] + "_gt.nii.gz")
    if text.endswith(".nii"):
        return Path(text[:-4] + "_gt.nii")
    raise ValueError(f"Frame must end in .nii or .nii.gz: {image_container.name}")


def _find_nifti(container: Path, expect_label: bool) -> Path:
    candidates: list[Path] = []
    if container.is_file() and container.stat().st_size > 0:
        candidates.append(container)
    elif container.is_dir():
        candidates.extend(container.rglob("*.nii"))
        candidates.extend(container.rglob("*.nii.gz"))
    for candidate in sorted(set(candidates)):
        if not candidate.is_file() or candidate.stat().st_size == 0:
            continue
        has_gt = "_gt" in candidate.name.lower()
        if container.is_dir() or has_gt == expect_label:
            return candidate
    kind = "label" if expect_label else "image"
    raise FileNotFoundError(f"No non-empty {kind} NIfTI found in {container}")


def _load_audited_split(split_json: Path, patient_id: str) -> dict[str, Any]:
    if not split_json.is_file():
        raise FileNotFoundError(f"Split JSON not found: {split_json}")
    with split_json.open(encoding="utf-8") as handle:
        split = json.load(handle)
    train_list = list(split.get("train_patients", []))
    val_list = list(split.get("val_patients", []))
    train_patients = set(train_list)
    val_patients = set(val_list)
    errors = []
    if split.get("seed") != 42:
        errors.append("seed must be 42")
    if abs(float(split.get("val_fraction", -1)) - 0.20) > 1e-12:
        errors.append("validation fraction must be 0.20")
    if len(train_list) != 80 or len(train_patients) != 80:
        errors.append("training split must contain 80 unique patients")
    if len(val_list) != 20 or len(val_patients) != 20:
        errors.append("validation split must contain 20 unique patients")
    if train_patients.intersection(val_patients):
        errors.append("training and validation patients overlap")
    if train_patients.union(val_patients) != EXPECTED_PATIENTS:
        errors.append("split must cover patient001 through patient100 exactly")
    ordered = sorted(EXPECTED_PATIENTS)
    random.Random(42).shuffle(ordered)
    if val_patients != set(ordered[:20]):
        errors.append("validation patients do not match random.Random(42)")
    if errors:
        raise ValueError("Invalid audited split: " + "; ".join(errors))
    if patient_id in train_patients:
        raise ValueError(
            f"Refusing qualitative evidence: {patient_id} belongs to the training split"
        )
    if patient_id not in val_patients:
        raise ValueError(
            f"Refusing qualitative evidence: {patient_id} is not in the validation split"
        )
    return split


def generate_qualitative_figure(
    checkpoint_path: Path,
    summary_csv: Path,
    split_json: Path,
    data_dir: Path,
    patient_id: str,
    frame_name: str,
    slice_index: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate one explicitly identified validation-case prediction figure."""
    if not re.fullmatch(r"patient\d{3}", patient_id):
        raise ValueError("patient_id must match patientNNN")
    if Path(frame_name).name != frame_name:
        raise ValueError("frame_name must be a filename, not a path")
    if "_gt" in frame_name.lower():
        raise ValueError("frame_name must identify an image, not a label")
    split = _load_audited_split(split_json, patient_id)
    summary_row = load_summary_metrics(summary_csv)["NanoMambaUNet"]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    patient_dir = data_dir / patient_id
    image_container = patient_dir / frame_name
    label_container = _label_container_for(image_container)
    image_path = _find_nifti(image_container, expect_label=False)
    label_path = _find_nifti(label_container, expect_label=True)

    import torch
    import torch.nn as nn
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        Resized,
        ScaleIntensityd,
        ToTensord,
    )

    from nano_mamba_core import MambaInspiredBottleneck

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class NanoMambaUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1, self.pool1 = DoubleConv(1, 16), nn.MaxPool3d(2)
            self.enc2, self.pool2 = DoubleConv(16, 32), nn.MaxPool3d(2)
            self.enc3, self.pool3 = DoubleConv(32, 64), nn.MaxPool3d(2)
            self.bottleneck = nn.Sequential(
                DoubleConv(64, 128),
                MambaInspiredBottleneck(channels=128),
            )
            self.up3, self.dec3 = nn.ConvTranspose3d(128, 64, 2, 2), DoubleConv(128, 64)
            self.up2, self.dec2 = nn.ConvTranspose3d(64, 32, 2, 2), DoubleConv(64, 32)
            self.up1, self.dec1 = nn.ConvTranspose3d(32, 16, 2, 2), DoubleConv(32, 16)
            self.out_conv = nn.Conv3d(16, 4, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            bottleneck = self.bottleneck(self.pool3(e3))
            d3 = self.dec3(torch.cat([e3, self.up3(bottleneck)], dim=1))
            d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
            d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
            return self.out_conv(d1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not the rigorous pipeline dictionary format")
    if checkpoint.get("model_name") != "NanoMambaUNet":
        raise ValueError(
            "Checkpoint model_name must be NanoMambaUNet; legacy fallbacks are refused"
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no model_state_dict")
    config = checkpoint.get("config", {})
    if config.get("seed") != split.get("seed"):
        raise ValueError("Checkpoint seed does not match the audited patient split")
    if abs(float(config.get("val_fraction", -1)) - 0.20) > 1e-12:
        raise ValueError("Checkpoint validation fraction does not match the audited split")
    if int(config.get("max_epochs", -1)) != 150:
        raise ValueError("Checkpoint max_epochs must be 150")
    if int(checkpoint.get("epoch", -1)) != int(summary_row["best_epoch"]):
        raise ValueError("Checkpoint epoch does not match the audited summary")
    if abs(
        float(checkpoint.get("val_mean_dice", -1))
        - float(summary_row["val_mean_dice"])
    ) > 1e-8:
        raise ValueError("Checkpoint validation Dice does not match the audited summary")
    spatial_size = tuple(config.get("spatial_size", ()))
    if spatial_size != (256, 256, 16):
        raise ValueError(
            f"Checkpoint spatial_size must be (256, 256, 16), got {spatial_size}"
        )
    if not 0 <= slice_index < spatial_size[2]:
        raise ValueError(f"slice_index must be between 0 and {spatial_size[2] - 1}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NanoMambaUNet().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = round(summary_row["params_m"] * 1_000_000)
    if parameter_count != expected_parameter_count:
        raise ValueError(
            "Current Nano-Mamba architecture parameter count does not match the audited summary"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            ),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    transformed = transform({"image": str(image_path), "label": str(label_path)})
    image_tensor = transformed["image"].unsqueeze(0).to(device)
    label_tensor = transformed["label"].unsqueeze(0)
    with torch.no_grad():
        prediction = torch.argmax(model(image_tensor), dim=1).cpu().numpy()[0]
    image_array = image_tensor.cpu().numpy()[0, 0]
    label_array = label_tensor.cpu().numpy()[0, 0]

    plt = _load_pyplot()
    cmap = ListedColormap(["#00000000", "#0072B2", "#009E73", "#D55E00"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=180)
    panels = [
        ("Original MRI", None),
        ("Ground truth", label_array),
        ("Nano-Mamba prediction", prediction),
    ]
    for axis, (title, overlay) in zip(axes, panels):
        axis.imshow(image_array[:, :, slice_index], cmap="gray")
        if overlay is not None:
            axis.imshow(
                overlay[:, :, slice_index],
                cmap=cmap,
                interpolation="nearest",
                alpha=0.48,
                vmin=0,
                vmax=3,
            )
        axis.set_title(title)
        axis.axis("off")
    patches = [
        mpatches.Patch(color="#0072B2", label="RV", alpha=0.7),
        mpatches.Patch(color="#009E73", label="MYO", alpha=0.7),
        mpatches.Patch(color="#D55E00", label="LV", alpha=0.7),
    ]
    figure.legend(handles=patches, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        f"Validation case {patient_id}/{frame_name}, resized slice {slice_index}"
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_stem = re.sub(r"\.nii(?:\.gz)?$", "", frame_name)
    figure_path = output_dir / (
        f"qualitative_validation_{patient_id}_{frame_stem}_slice{slice_index:02d}.png"
    )
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "artifact_type": "qualitative_validation_figure",
        "patient_id": patient_id,
        "frame_name": frame_name,
        "resized_slice_index": slice_index,
        "split_role": "validation",
        "split_sha256": sha256_file(split_json),
        "summary_metrics_sha256": sha256_file(summary_csv),
        "summary_best_epoch": summary_row["best_epoch"],
        "summary_val_mean_dice": summary_row["val_mean_dice"],
        "summary_parameter_count": expected_parameter_count,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_model_name": checkpoint.get("model_name"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_mean_dice": checkpoint.get("val_mean_dice"),
        "checkpoint_config": config,
        "instantiated_parameter_count": parameter_count,
        "main_result_pipeline": str(MAIN_RESULT_PIPELINE.relative_to(REPOSITORY_ROOT)),
        "main_result_pipeline_sha256": sha256_file(MAIN_RESULT_PIPELINE),
        "image_sha256": sha256_file(image_path),
        "label_sha256": sha256_file(label_path),
        "preprocessing": [
            "LoadImaged",
            "EnsureChannelFirstd",
            "Resized(256,256,16; image=trilinear,label=nearest)",
            "ScaleIntensityd(image)",
            "ToTensord",
        ],
        "figure_sha256": sha256_file(figure_path),
        "generator": str(SCRIPT_PATH.relative_to(REPOSITORY_ROOT)),
        "generator_sha256": sha256_file(SCRIPT_PATH),
    }
    manifest_path = figure_path.with_suffix(".json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return figure_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--skip-quantitative",
        action="store_true",
        help="Do not regenerate the three CSV-derived quantitative figures.",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Also generate one rigorously checked validation-case figure.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--patient-id")
    parser.add_argument("--frame-name")
    parser.add_argument("--slice-index", type=int, default=8)
    args = parser.parse_args()
    if args.qualitative:
        missing = [
            name
            for name in ("checkpoint", "data_dir", "patient_id", "frame_name")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "--qualitative requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
    return args


def main() -> None:
    args = parse_args()
    if not args.skip_quantitative:
        rows = load_summary_metrics(args.summary_csv)
        outputs = generate_quantitative_figures(rows, args.output_dir)
        manifest = write_quantitative_manifest(
            args.summary_csv,
            rows,
            outputs,
            args.output_dir,
        )
        print("Generated quantitative figures:")
        for path in [*outputs, manifest]:
            print(f"  {path}")

    if args.qualitative:
        figure, manifest = generate_qualitative_figure(
            checkpoint_path=args.checkpoint,
            summary_csv=args.summary_csv,
            split_json=args.split_json,
            data_dir=args.data_dir,
            patient_id=args.patient_id,
            frame_name=args.frame_name,
            slice_index=args.slice_index,
            output_dir=args.output_dir,
        )
        print("Generated audited qualitative evidence:")
        print(f"  {figure}")
        print(f"  {manifest}")


if __name__ == "__main__":
    main()
