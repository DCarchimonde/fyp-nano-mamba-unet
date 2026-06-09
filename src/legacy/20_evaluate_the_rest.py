import os
import glob
import torch
import torch.nn as nn
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SegResNet
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 组装半血 Mamba 的躯壳 =================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.net(x)


class Ablation_HalfMamba_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(
            DoubleConv(64, 64),
            SpatioTemporalMambaBottleneck(channels=64),
            DoubleConv(64, 128)
        )
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, 4, kernel_size=1)

    def forward(self, x):
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


# ================= 2. 核心 Dice 计算器 =================
def compute_dice_scores(y_pred, y_true, num_classes=4):
    pred_labels = torch.argmax(y_pred, dim=1)
    true_labels = y_true.squeeze(1)
    dices = []
    for c in range(1, num_classes):
        pred_c = (pred_labels == c).float()
        true_c = (true_labels == c).float()
        intersection = (pred_c * true_c).sum()
        union = pred_c.sum() + true_c.sum()
        if union == 0:
            dices.append(1.0)
        else:
            dice_score = (2. * intersection / union).item()
            dices.append(dice_score)
    return dices


# ================= 3. 数据拉取 (同卷统考) =================
print("📦 正在拉取 ACDC 数据集...")
data_dir = r"D:\AI_FYP\Data\ACDC\database\training"
data_dicts = []

for patient_folder in sorted(os.listdir(data_dir)):
    patient_path = os.path.join(data_dir, patient_folder)
    if not os.path.isdir(patient_path) or not patient_folder.startswith("patient"): continue
    frame_dirs = glob.glob(os.path.join(patient_path, "patient*_frame*.nii"))
    img_dirs = [d for d in frame_dirs if "_gt" not in d]
    for img_dir in img_dirs:
        label_dir = img_dir.replace(".nii", "_gt.nii")
        img_files = glob.glob(os.path.join(img_dir, "*.nii"))
        label_files = glob.glob(os.path.join(label_dir, "*.nii"))
        if img_files and label_files:
            data_dicts.append({"image": img_files[0], "label": label_files[0]})

transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])

dataloader = DataLoader(Dataset(data=data_dicts, transform=transform), batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 4. 自动化阅卷主程序 =================
def evaluate_model(model, weight_path, name):
    print(f"\n🧠 正在唤醒 [{name}] 并注入灵魂...")
    model.load_state_dict(torch.load(weight_path))
    model = model.to(device)
    model.eval()

    print(f"🚀 开始批改 [{name}] 的考卷...")
    all_rv, all_myo, all_lv = [], [], []

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            outputs = model(inputs)
            dices = compute_dice_scores(outputs, labels)
            all_rv.append(dices[0])
            all_myo.append(dices[1])
            all_lv.append(dices[2])

    avg_rv = np.mean(all_rv) * 100
    avg_myo = np.mean(all_myo) * 100
    avg_lv = np.mean(all_lv) * 100
    mean_dice = (avg_rv + avg_myo + avg_lv) / 3

    print("=" * 50)
    print(f"🏆 {name} 终极成绩单 🏆")
    print("=" * 50)
    print(f"🔵 RV Dice:  {avg_rv:.2f}%")
    print(f"🟢 MYO Dice: {avg_myo:.2f}%")
    print(f"🔴 LV Dice:  {avg_lv:.2f}%")
    print("-" * 50)
    print(f"⭐ 平均 DSC: {mean_dice:.2f}%")
    print("=" * 50)


# ================= 5. 开始打分 =================
if __name__ == "__main__":
    # 评测 半血 Mamba
    half_mamba = Ablation_HalfMamba_UNet()
    evaluate_model(half_mamba, r"D:\AI_FYP\models\best_ablation_halfmamba.pth", "Ablation (Half Mamba 64通道)")
    del half_mamba
    torch.cuda.empty_cache()

    # 评测 SegResNet
    segresnet = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=4,
        init_filters=16, dropout_prob=0.2
    )
    evaluate_model(segresnet, r"D:\AI_FYP\models\best_segresnet.pth", "SegResNet (医学巨无霸 SOTA)")