import os
import glob
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet  # 引入基础 U-Net


# ================= 1. 姐姐手写的硬核 Dice 计算器 (保持不变) =================
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


# ================= 2. 召唤传统 U-Net (480万参数的巨无霸) =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\AI_FYP\models\best_baseline_unet.pth"

print("🧠 正在唤醒传统的巨无霸 U-Net (Baseline)...")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ================= 3. 数据拉取 (保持公平，用同一套卷子) =================
print("📦 正在拉取同一套试卷进行绝对公平的对比测试...")
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

dataset = Dataset(data=data_dicts, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ================= 4. 开始基础模型的阅卷 =================
print("\n🚀 开始对基础 U-Net 执行全面评估...\n")
all_rv_dices, all_myo_dices, all_lv_dices = [], [], []

with torch.no_grad():
    for i, batch_data in enumerate(dataloader):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        outputs = model(inputs)

        dices = compute_dice_scores(outputs, labels)
        all_rv_dices.append(dices[0])
        all_myo_dices.append(dices[1])
        all_lv_dices.append(dices[2])

        if (i + 1) % 20 == 0:
            print(f"  已批改 {i + 1}/{len(dataloader)} 份考卷...")

# ================= 5. 输出基础版成绩单 =================
avg_rv = np.mean(all_rv_dices) * 100
avg_myo = np.mean(all_myo_dices) * 100
avg_lv = np.mean(all_lv_dices) * 100
mean_dice = (avg_rv + avg_myo + avg_lv) / 3

print("\n" + "=" * 50)
print("📊 传统基础版 3D U-Net 性能评估报告 📊")
print("=" * 50)
print(f"🔵 右心室 (RV) Dice: {avg_rv:.2f}%")
print(f"🟢 心肌 (MYO)     Dice: {avg_myo:.2f}%")
print(f"🔴 左心室 (LV)  Dice: {avg_lv:.2f}%")
print("-" * 50)
print(f"⭐ 平均整体 Dice (Mean DSC): {mean_dice:.2f}%")
print("=" * 50)