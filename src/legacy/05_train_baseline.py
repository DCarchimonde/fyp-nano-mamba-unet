import os
import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss

# ================= 0. 兵马未动，粮草先行 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的计算设备: {device}")

# 创建专门存放模型权重的文件夹 (保持项目整洁)
save_dir = r"D:\AI_FYP\models"
os.makedirs(save_dir, exist_ok=True)
print(f"📁 模型将保存在此目录: {save_dir}")

# ================= 1. 数据流水线 =================
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

dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ================= 2. 召唤大脑 (Model) =================
print("🧠 正在初始化 U-Net 并移动至显卡...")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

# ================= 3. 定义戒尺与引擎 =================
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ================= 4. 终极舞台：工业级训练循环 =================
max_epochs = 150  # 真实训练，咱们先跑 50 轮
best_loss = float('inf')  # 初始最佳误差设为无穷大
best_model_path = os.path.join(save_dir, "best_baseline_unet.pth")

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

    # 一轮结束，计算平均误差
    epoch_loss /= step
    print(f"✅ Epoch {epoch + 1} 结束! 平均误差 (Average Loss): {epoch_loss:.4f}")

    # 🌟 守门员机制：如果这轮的误差比历史最低还要低，就立刻保存模型！
    if epoch_loss < best_loss:
        print(f"🏆 发现更聪明的模型！误差从 {best_loss:.4f} 降到了 {epoch_loss:.4f}")
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 模型权重已安全备份至: {best_model_path}")

print(f"\n🎉 训练彻底结束！最聪明的模型已经保存在了 {best_model_path}，你可以随时调用它了！")