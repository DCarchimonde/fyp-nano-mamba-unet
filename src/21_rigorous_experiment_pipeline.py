"""
Rigorous patient-level experiment pipeline for the Nano-Mamba U-Net FYP.

This script fixes the main evidence-chain problem in the earlier scripts:
training and evaluation must not be performed on the same cases.

It performs:
1. deterministic patient-level train/validation split;
2. same split for all models;
3. checkpoint selection by validation mean Dice;
4. final reporting on held-out validation cases only;
5. CSV/JSON outputs for thesis evidence.

Run in PyCharm:
    python src/21_rigorous_experiment_pipeline.py
"""

import csv
import glob
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.networks.nets import AttentionUnet, SegResNet
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, ScaleIntensityd, ToTensord

from nano_mamba_core import SpatioTemporalMambaBottleneck


# =============================================================================
# 0. EDIT HERE IF NEEDED
# =============================================================================

PROJECT_ROOT = Path(r"D:\AI_FYP")
DATA_DIR = PROJECT_ROOT / "Data" / "ACDC" / "database" / "training"

OUTPUT_DIR = PROJECT_ROOT / "experiment_outputs" / "rigorous_patient_split"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SPLIT_JSON = OUTPUT_DIR / "patient_split_seed42.json"
SUMMARY_CSV = OUTPUT_DIR / "summary_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary_metrics.json"

SPATIAL_SIZE = (256, 256, 16)
NUM_CLASSES = 4
CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}

SEED = 42
VAL_FRACTION = 0.20
MAX_EPOCHS = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# SegResNet32 is the larger ~18M-parameter setting. Enable only if GPU memory allows.
MODELS_TO_RUN = [
    "NanoMambaUNet",
    "Ablation_NoMamba_UNet",
    "Ablation_HalfMamba_UNet",
    "AttentionUNet",
    "SegResNet16",
    # "SegResNet32",
]

TRAIN_BATCH_SIZE = {
    "NanoMambaUNet": 2,
    "Ablation_NoMamba_UNet": 2,
    "Ablation_HalfMamba_UNet": 2,
    "AttentionUNet": 1,
    "SegResNet16": 1,
    "SegResNet32": 1,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =============================================================================
# 2. MODELS
# =============================================================================

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
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.enc1, self.pool1 = DoubleConv(in_channels, 16), nn.MaxPool3d(2)
        self.enc2, self.pool2 = DoubleConv(16, 32), nn.MaxPool3d(2)
        self.enc3, self.pool3 = DoubleConv(32, 64), nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(DoubleConv(64, 128), SpatioTemporalMambaBottleneck(channels=128))
        self.up3, self.dec3 = nn.ConvTranspose3d(128, 64, 2, 2), DoubleConv(128, 64)
        self.up2, self.dec2 = nn.ConvTranspose3d(64, 32, 2, 2), DoubleConv(64, 32)
        self.up1, self.dec1 = nn.ConvTranspose3d(32, 16, 2, 2), DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([e3, self.up3(b)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
        return self.out_conv(d1)


class AblationNoMambaUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.enc1, self.pool1 = DoubleConv(in_channels, 16), nn.MaxPool3d(2)
        self.enc2, self.pool2 = DoubleConv(16, 32), nn.MaxPool3d(2)
        self.enc3, self.pool3 = DoubleConv(32, 64), nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(DoubleConv(64, 128), DoubleConv(128, 128))
        self.up3, self.dec3 = nn.ConvTranspose3d(128, 64, 2, 2), DoubleConv(128, 64)
        self.up2, self.dec2 = nn.ConvTranspose3d(64, 32, 2, 2), DoubleConv(64, 32)
        self.up1, self.dec1 = nn.ConvTranspose3d(32, 16, 2, 2), DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([e3, self.up3(b)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
        return self.out_conv(d1)


class AblationHalfMambaUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.enc1, self.pool1 = DoubleConv(in_channels, 16), nn.MaxPool3d(2)
        self.enc2, self.pool2 = DoubleConv(16, 32), nn.MaxPool3d(2)
        self.enc3, self.pool3 = DoubleConv(32, 64), nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(DoubleConv(64, 64), SpatioTemporalMambaBottleneck(channels=64), DoubleConv(64, 128))
        self.up3, self.dec3 = nn.ConvTranspose3d(128, 64, 2, 2), DoubleConv(128, 64)
        self.up2, self.dec2 = nn.ConvTranspose3d(64, 32, 2, 2), DoubleConv(64, 32)
        self.up1, self.dec1 = nn.ConvTranspose3d(32, 16, 2, 2), DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([e3, self.up3(b)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
        return self.out_conv(d1)


def build_model(model_name: str) -> nn.Module:
    if model_name == "NanoMambaUNet":
        return NanoMambaUNet(1, NUM_CLASSES)
    if model_name == "Ablation_NoMamba_UNet":
        return AblationNoMambaUNet(1, NUM_CLASSES)
    if model_name == "Ablation_HalfMamba_UNet":
        return AblationHalfMambaUNet(1, NUM_CLASSES)
    if model_name == "AttentionUNet":
        return AttentionUnet(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    if model_name == "SegResNet16":
        return SegResNet(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES, init_filters=16, dropout_prob=0.2)
    if model_name == "SegResNet32":
        return SegResNet(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES, init_filters=32, dropout_prob=0.2)
    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# 3. DATA DISCOVERY AND PATIENT SPLIT
# =============================================================================

def natural_sort_key(text: str) -> List[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def label_path_for_image(image_path: Path) -> Path:
    s = str(image_path)
    if s.endswith(".nii.gz"):
        return Path(s[:-7] + "_gt.nii.gz")
    if s.endswith(".nii"):
        return Path(s[:-4] + "_gt.nii")
    raise ValueError(f"Unsupported extension: {image_path}")


def discover_acdc_cases(data_dir: Path) -> List[Dict[str, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    cases = []
    patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("patient")], key=lambda p: natural_sort_key(p.name))
    for patient_dir in patient_dirs:
        image_paths = sorted([
            Path(p) for p in glob.glob(str(patient_dir / "*.nii*"))
            if "_gt" not in Path(p).name and "4d" not in Path(p).name.lower()
        ], key=lambda p: natural_sort_key(p.name))
        for image_path in image_paths:
            label_path = label_path_for_image(image_path)
            if label_path.exists():
                frame_match = re.search(r"(frame\d+)", image_path.name)
                frame_id = frame_match.group(1) if frame_match else image_path.stem
                cases.append({
                    "patient_id": patient_dir.name,
                    "case_id": f"{patient_dir.name}_{frame_id}",
                    "image": str(image_path),
                    "label": str(label_path),
                })

    if not cases:
        raise RuntimeError("No ACDC cases found. Check DATA_DIR and file structure.")
    return cases


def create_or_load_split(cases: Sequence[Dict[str, str]]) -> Dict[str, object]:
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if SPLIT_JSON.exists():
        with SPLIT_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)

    patient_ids = sorted({c["patient_id"] for c in cases}, key=natural_sort_key)
    rng = random.Random(SEED)
    shuffled = patient_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * VAL_FRACTION))

    split = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train_patients": sorted(shuffled[val_count:], key=natural_sort_key),
        "val_patients": sorted(shuffled[:val_count], key=natural_sort_key),
    }
    with SPLIT_JSON.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def split_cases(cases: Sequence[Dict[str, str]], split: Dict[str, object]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train_patients = set(split["train_patients"])
    val_patients = set(split["val_patients"])
    if train_patients & val_patients:
        raise RuntimeError("Patient leakage detected between train and validation split.")
    train_cases = [c for c in cases if c["patient_id"] in train_patients]
    val_cases = [c for c in cases if c["patient_id"] in val_patients]
    if not train_cases or not val_cases:
        raise RuntimeError("Train or validation set is empty.")
    return train_cases, val_cases


def build_transform() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode=("trilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])


# =============================================================================
# 4. METRICS AND BENCHMARKS
# =============================================================================

@torch.no_grad()
def dice_per_case(logits: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, float]]:
    pred = torch.argmax(logits, dim=1)
    true = labels.squeeze(1).long()
    rows = []
    for b in range(pred.shape[0]):
        row = {}
        scores = []
        for class_idx, class_name in CLASS_NAMES.items():
            pred_c = (pred[b] == class_idx).float()
            true_c = (true[b] == class_idx).float()
            union = pred_c.sum() + true_c.sum()
            dice = 1.0 if union.item() == 0 else (2.0 * (pred_c * true_c).sum() / union).item()
            row[f"dice_{class_name}"] = float(dice)
            scores.append(float(dice))
        row["mean_dice"] = float(np.mean(scores))
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, save_csv: Path = None) -> Dict[str, object]:
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        metrics = dice_per_case(model(images), labels)
        patient_ids = list(batch.get("patient_id", ["unknown"] * len(metrics)))
        case_ids = list(batch.get("case_id", ["unknown"] * len(metrics)))
        for i, row in enumerate(metrics):
            rows.append({"patient_id": patient_ids[i], "case_id": case_ids[i], **row})

    summary = {
        "num_cases": len(rows),
        "dice_RV": float(np.mean([r["dice_RV"] for r in rows])),
        "dice_MYO": float(np.mean([r["dice_MYO"] for r in rows])),
        "dice_LV": float(np.mean([r["dice_LV"] for r in rows])),
        "mean_dice": float(np.mean([r["mean_dice"] for r in rows])),
        "per_case": rows,
    }

    if save_csv:
        with save_csv.open("w", newline="", encoding="utf-8") as f:
            fields = ["patient_id", "case_id", "dice_RV", "dice_MYO", "dice_LV", "mean_dice"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{k: r[k] for k in fields} for r in rows])
    return summary


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def benchmark(model: nn.Module, runs: int = 30) -> Dict[str, float]:
    model.eval().to(DEVICE)
    dummy = torch.randn((1, 1, *SPATIAL_SIZE), device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
    for _ in range(5):
        _ = model(dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    else:
        peak_vram = None
    latency_ms = (time.perf_counter() - start) / runs * 1000.0
    return {"params": count_params(model), "params_m": count_params(model) / 1e6, "latency_ms": latency_ms, "fps": 1000.0 / latency_ms, "peak_vram_mb": peak_vram}


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train_one_model(model_name: str, train_cases: List[Dict[str, str]], val_cases: List[Dict[str, str]], transform: Compose) -> Dict[str, object]:
    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)
    set_seed(SEED)

    train_loader = DataLoader(Dataset(train_cases, transform), batch_size=TRAIN_BATCH_SIZE.get(model_name, 1), shuffle=True, num_workers=0)
    val_loader = DataLoader(Dataset(val_cases, transform), batch_size=1, shuffle=False, num_workers=0)

    model = build_model(model_name).to(DEVICE)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    checkpoint_path = CHECKPOINT_DIR / f"best_{model_name}.pth"
    per_case_csv = OUTPUT_DIR / f"per_case_{model_name}.csv"
    log_csv = OUTPUT_DIR / f"training_log_{model_name}.csv"

    best_val_dice = -1.0
    best_epoch = -1
    log_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_summary = evaluate(model, val_loader)
        val_mean = float(val_summary["mean_dice"])
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_mean_dice": val_mean, "val_dice_RV": val_summary["dice_RV"], "val_dice_MYO": val_summary["dice_MYO"], "val_dice_LV": val_summary["dice_LV"]})

        if epoch == 1 or epoch % 10 == 0:
            print(f"[{model_name}] epoch {epoch:03d}/{MAX_EPOCHS} | train_loss={train_loss:.4f} | val_mean_dice={val_mean * 100:.2f}%")

        if val_mean > best_val_dice:
            best_val_dice = val_mean
            best_epoch = epoch
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "val_mean_dice": best_val_dice,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": {"seed": SEED, "val_fraction": VAL_FRACTION, "spatial_size": SPATIAL_SIZE, "max_epochs": MAX_EPOCHS},
            }, checkpoint_path)

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["epoch", "train_loss", "val_mean_dice", "val_dice_RV", "val_dice_MYO", "val_dice_LV"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val = evaluate(model, val_loader, save_csv=per_case_csv)
    speed = benchmark(model)

    result = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "num_train_cases": len(train_cases),
        "num_val_cases": len(val_cases),
        "val_dice_RV": final_val["dice_RV"],
        "val_dice_MYO": final_val["dice_MYO"],
        "val_dice_LV": final_val["dice_LV"],
        "val_mean_dice": final_val["mean_dice"],
        "params": int(speed["params"]),
        "params_m": float(speed["params_m"]),
        "fps": float(speed["fps"]),
        "latency_ms": float(speed["latency_ms"]),
        "peak_vram_mb": speed["peak_vram_mb"],
        "checkpoint_path": str(checkpoint_path),
        "per_case_csv": str(per_case_csv),
        "training_log_csv": str(log_csv),
    }

    print(f"Finished {model_name}: best_epoch={best_epoch}, val_mean_dice={result['val_mean_dice'] * 100:.2f}%, params={result['params_m']:.2f}M, fps={result['fps']:.2f}")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_summary(results: List[Dict[str, object]]) -> None:
    fields = ["model_name", "best_epoch", "num_train_cases", "num_val_cases", "val_dice_RV", "val_dice_MYO", "val_dice_LV", "val_mean_dice", "params_m", "fps", "latency_ms", "peak_vram_mb", "checkpoint_path"]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k) for k in fields} for r in results])
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nCurrent validation summary:")
    for r in results:
        print(f"{r['model_name']:24s} | Mean DSC {r['val_mean_dice'] * 100:6.2f}% | RV {r['val_dice_RV'] * 100:6.2f}% | MYO {r['val_dice_MYO'] * 100:6.2f}% | LV {r['val_dice_LV'] * 100:6.2f}% | Params {r['params_m']:5.2f}M | FPS {r['fps']:6.2f}")
    print(f"\nSaved: {SUMMARY_CSV}")


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"DATA_DIR: {DATA_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    cases = discover_acdc_cases(DATA_DIR)
    split = create_or_load_split(cases)
    train_cases, val_cases = split_cases(cases, split)
    print(f"Patients: train={len(split['train_patients'])}, val={len(split['val_patients'])}")
    print(f"Cases: train={len(train_cases)}, val={len(val_cases)}")
    print(f"Split saved at: {SPLIT_JSON}")

    transform = build_transform()
    results = []
    for model_name in MODELS_TO_RUN:
        results.append(train_one_model(model_name, train_cases, val_cases, transform))
        save_summary(results)

    print("\nAll rigorous experiments finished.")


if __name__ == "__main__":
    main()
