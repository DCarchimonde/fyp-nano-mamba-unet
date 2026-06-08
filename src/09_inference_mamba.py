import os
import glob
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from nano_mamba_core import SpatioTemporalMambaBottleneck

# ================= 1. 组装模型躯壳 =================
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

# ================= 2. 注入灵魂 (加载权重) =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\AI_FYP\models\best_nanomamba_unet.pth"

print("🧠 正在唤醒你的时空大杀器 Nano-Mamba...")
model = NanoMambaUNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ 模型唤醒成功！准备考试。")

# ================= 3. 抓取考题并预处理 =================
data_dir = r"D:\AI_FYP\Data\ACDC\database\training"
patient_path = os.path.join(data_dir, "patient002") # 还是用 2 号病人来比对
frame_dirs = glob.glob(os.path.join(patient_path, "patient*_frame*.nii"))
img_dirs = [d for d in frame_dirs if "_gt" not in d]

img_dir = img_dirs[0]
label_dir = img_dir.replace(".nii", "_gt.nii")
img_file = glob.glob(os.path.join(img_dir, "*.nii"))[0]
label_file = glob.glob(os.path.join(label_dir, "*.nii"))[0]

transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])
data_dict = transform({"image": img_file, "label": label_file})
img_tensor = data_dict["image"].unsqueeze(0).to(device)
label_tensor = data_dict["label"].unsqueeze(0).to(device)

# ================= 4. Nano-Mamba 开始画图 =================
print("🚀 Nano-Mamba 正在读取时空特征并生成预测...")
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1)

# ================= 5. 可视化比对 =================
print("🎨 正在渲染超轻量级 Mamba 预测图...")
slice_idx = 8

img_slice = img_tensor[0, 0, :, :, slice_idx].cpu().numpy()
label_slice = label_tensor[0, 0, :, :, slice_idx].cpu().numpy()
pred_slice = pred_mask[0, :, :, slice_idx].cpu().numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_slice, cmap='gray')
plt.title(f'Raw MRI (Slice {slice_idx})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(label_slice, cmap='nipy_spectral', vmin=0, vmax=3)
plt.title('Ground Truth (Doctor)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pred_slice, cmap='nipy_spectral', vmin=0, vmax=3)
plt.title('AI Prediction (Nano-Mamba!)')
plt.axis('off')

plt.tight_layout()
plt.show()