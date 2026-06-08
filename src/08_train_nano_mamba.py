import os
import glob
import torch
import torch.nn as nn
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss

# 🚀 导入你之前写好的 Mamba 核心脏
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 定义你的专属模型 (直接写在这里，方便调用) =================
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

    def forward(self, x):
        return self.net(x)


class NanoMambaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        # 核心瓶颈层：注入 Mamba 时空灵魂
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


# ================= 2. 准备炼丹炉与数据流水线 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的计算设备: {device}")

save_dir = r"D:\AI_FYP\models"
os.makedirs(save_dir, exist_ok=True)

print("📦 正在准备数据流水线...")
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

preprocessing_pipeline = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])

# Batch Size 依然设置为 2，保护你的 4060 显存
dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ================= 3. 召唤你的专属大杀器 =================
print("🧠 正在初始化 Nano-Mamba U-Net 并移动至显卡...")
model = NanoMambaUNet().to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ================= 4. 开启时空炼丹 =================
max_epochs = 150  # 咱们直接拉满跑 150 轮，看它能有多聪明！
best_loss = float('inf')
best_model_path = os.path.join(save_dir, "best_nanomamba_unet.pth")

print(f"\n🚀 开始真实训练！预计训练 {max_epochs} 轮 (Epochs)...")

for epoch in range(max_epochs):
    print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in dataloader:
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 10 == 0:
            print(f"  Step {step}/{len(dataloader)}, 当前批次误差 (Loss): {loss.item():.4f}")

    epoch_loss /= step
    print(f"✅ Epoch {epoch + 1} 结束! 平均误差 (Average Loss): {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        print(f"🏆 Nano-Mamba 进化了！误差从 {best_loss:.4f} 降到了 {epoch_loss:.4f}")
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 专属神级模型权重已备份至: {best_model_path}")

print(f"\n🎉 训练彻底结束！Mamba 已经彻底征服了数据集！")