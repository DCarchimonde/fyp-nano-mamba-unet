import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 唤醒你的 Nano-Mamba =================
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
            SpatioTemporalMambaBottleneck(channels=128)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\AI_FYP\models\best_nanomamba_unet.pth"
model = NanoMambaUNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ================= 2. 读取数据 (找一个典型的病人) =================
data_dir = r"D:\AI_FYP\Data\ACDC\database\training\patient005"

# 第一步：找到外层伪装成 .nii 的文件夹
img_dir = glob.glob(os.path.join(data_dir, "patient005_frame01.nii"))[0]
label_dir = glob.glob(os.path.join(data_dir, "patient005_frame01_gt.nii"))[0]

# 第二步：穿透文件夹，找到里面真正的 .nii 文件 (比如 DCM10Gate1.nii)
img_file = glob.glob(os.path.join(img_dir, "*.nii"))[0]
label_file = glob.glob(os.path.join(label_dir, "*.nii"))[0]

transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])

# 第三步：把真正的文件路径喂给 MONAI
data_dict = transform({"image": img_file, "label": label_file})
img_tensor = data_dict["image"].unsqueeze(0).to(device)
label_tensor = data_dict["label"].unsqueeze(0).to(device)

# ================= 3. 模型预测 =================
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

img_np = img_tensor.cpu().numpy()[0, 0]
label_np = label_tensor.cpu().numpy()[0, 0]

# ================= 4. 顶刊级绘图逻辑 =================
slice_idx = 8  # 选心脏最明显的一层

# 定义极其专业的半透明调色板：背景透明，RV蓝色，MYO绿色，LV红色
colors = ['#00000000', 'blue', 'lime', 'red']
custom_cmap = ListedColormap(colors)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)  # 高清输出
plt.suptitle(f"Qualitative Segmentation Analysis - Nano-Mamba U-Net (Slice {slice_idx})", fontsize=16,
             fontweight='bold')

# 图1：原始 MRI
axes[0].imshow(img_np[:, :, slice_idx], cmap='gray')
axes[0].set_title("Original MRI", fontsize=14)
axes[0].axis('off')

# 图2：医生标注 (Ground Truth)
axes[1].imshow(img_np[:, :, slice_idx], cmap='gray')
# interpolation='nearest' 保证边缘锐利，alpha=0.4 是灵魂半透明
axes[1].imshow(label_np[:, :, slice_idx], cmap=custom_cmap, interpolation='nearest', alpha=0.45)
axes[1].set_title("Ground Truth (Expert)", fontsize=14)
axes[1].axis('off')

# 图3：模型预测 (Nano-Mamba)
axes[2].imshow(img_np[:, :, slice_idx], cmap='gray')
axes[2].imshow(pred_mask[:, :, slice_idx], cmap=custom_cmap, interpolation='nearest', alpha=0.45)
axes[2].set_title("Nano-Mamba Prediction", fontsize=14)
axes[2].axis('off')

# 图例生成
import matplotlib.patches as mpatches

legend_patches = [
    mpatches.Patch(color='blue', label='Right Ventricle (RV)', alpha=0.6),
    mpatches.Patch(color='lime', label='Myocardium (MYO)', alpha=0.6),
    mpatches.Patch(color='red', label='Left Ventricle (LV)', alpha=0.6)
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # 给图例留点空间

# 直接保存为可以直接插入 LaTeX 的高清图片！
os.makedirs(r"D:\AI_FYP\figures", exist_ok=True)
save_path = r"D:\AI_FYP\figures\qualitative_result.png"
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 顶刊级别高清对比图已保存至: {save_path}")
plt.show()