import os
import glob
import torch
import torch.nn as nn
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 定义你的半血版 Mamba =================
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
        # 半血 Mamba: 前后需要加升降维适配器
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


# ================= 2. 数据流水线 =================
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

train_files, val_files = data_dicts[:-20], data_dicts[-20:]

train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])

# 注意：SegResNet 极其耗显存，这里把 batch_size 强制设为 1 防止 OOM
train_loader = DataLoader(Dataset(data=train_files, transform=train_transform), batch_size=1, shuffle=True)


# ================= 3. 炼丹核心函数 =================
def train_model(model, model_name, device, max_epochs=150):
    print(f"\n" + "=" * 50)
    print(f"🚀 开始魔鬼集训: {model_name}")
    print("=" * 50)

    model = model.to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = 999.0
    save_path = f"D:\\AI_FYP\\models\\best_{model_name.replace(' ', '_').lower()}.pth"

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"[{model_name}] Epoch {epoch + 1}/{max_epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    print(f"✅ {model_name} 训练完成！最强权重已保存至: {save_path}")


# ================= 4. 开启长线流水线 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⚠️ 警告：本次训练包含 18.8M 参数巨兽 SegResNet，请确保电脑散热良好！")

    # 第一位选手：半血 Mamba (64通道)
    half_mamba = Ablation_HalfMamba_UNet()
    train_model(half_mamba, "Ablation_HalfMamba", device)
    del half_mamba
    torch.cuda.empty_cache()

    # 第二位选手：医学霸主 SegResNet
    segresnet = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=4,
        init_filters=16, dropout_prob=0.2
    )
    train_model(segresnet, "SegResNet", device)

    print("\n🎉🎉🎉 全部宇宙级对比实验，正式补齐！")