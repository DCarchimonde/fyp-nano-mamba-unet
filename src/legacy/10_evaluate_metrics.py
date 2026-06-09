import os
import glob
import torch
import torch.nn as nn
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 躯壳与灵魂组装 (复用你的大杀器) =================
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


class NanoMambaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(
            DoubleConv(64, 128),
            SpatioTemporalMambaBottleneck(channels=128)
        )
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

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


# ================= 2. 姐姐手写的硬核 Dice 计算器 =================
def compute_dice_scores(y_pred, y_true, num_classes=4):
    """
    输入:
    y_pred: 模型的原始输出 [Batch, 4, H, W, D]
    y_true: 真实标签 [Batch, 1, H, W, D]
    输出: 一个包含 RV, MYO, LV 三个部位 Dice 分数的列表
    """
    # 1. 把概率图变成确定的类别标签 (0, 1, 2, 3)
    pred_labels = torch.argmax(y_pred, dim=1)
    true_labels = y_true.squeeze(1)

    dices = []
    # 遍历类别 1(RV), 2(MYO), 3(LV)。注意：跳过 0(背景)
    for c in range(1, num_classes):
        # 创建二进制掩码 (只有当前类别是 1，其他都是 0)
        pred_c = (pred_labels == c).float()
        true_c = (true_labels == c).float()

        # 计算交集和并集
        intersection = (pred_c * true_c).sum()
        union = pred_c.sum() + true_c.sum()

        # 严谨的数学处理：如果预测和真实都没有这个类别，Dice 算作 1.0 (完美)
        if union == 0:
            dices.append(1.0)
        else:
            dice_score = (2. * intersection / union).item()
            dices.append(dice_score)

    return dices


# ================= 3. 环境与数据加载 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\AI_FYP\models\best_nanomamba_unet.pth"

print("🧠 正在唤醒沉睡的 Nano-Mamba...")
model = NanoMambaUNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

print("📦 正在拉取数据集进行全面阅卷...")
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

# 评测的时候不需要打乱顺序，并且 batch_size 可以设为 1 方便统计
dataset = Dataset(data=data_dicts, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ================= 4. 开启全面阅卷 =================
print("\n🚀 开始执行全面定量评估 (Quantitative Evaluation)...\n")

all_rv_dices = []
all_myo_dices = []
all_lv_dices = []

with torch.no_grad():  # 考试状态，绝不更新参数
    for i, batch_data in enumerate(dataloader):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        outputs = model(inputs)

        # 调用姐姐写的神器，算出这一张图的三个部位得分
        dices = compute_dice_scores(outputs, labels)

        all_rv_dices.append(dices[0])
        all_myo_dices.append(dices[1])
        all_lv_dices.append(dices[2])

        if (i + 1) % 20 == 0:
            print(f"  已批改 {i + 1}/{len(dataloader)} 份考卷...")

# ================= 5. 输出终极成绩单 =================
avg_rv = np.mean(all_rv_dices) * 100
avg_myo = np.mean(all_myo_dices) * 100
avg_lv = np.mean(all_lv_dices) * 100
mean_dice = (avg_rv + avg_myo + avg_lv) / 3

print("\n" + "=" * 50)
print("🏆 Nano-Mamba U-Net 终极性能评估报告 🏆")
print("=" * 50)
print(f"🔵 右心室 (Right Ventricle, RV) Dice: {avg_rv:.2f}%")
print(f"🟢 心肌 (Myocardium, MYO)     Dice: {avg_myo:.2f}%")
print(f"🔴 左心室 (Left Ventricle, LV)  Dice: {avg_lv:.2f}%")
print("-" * 50)
print(f"⭐ 平均整体 Dice 相似系数 (Mean DSC): {mean_dice:.2f}%")
print("=" * 50)
print("\n老弟，把这四个数字狠狠地刻在你的论文结果表里！")