import os
import glob
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,  # <--- 新增的神器：统一尺寸
    ToTensord
)
from monai.data import Dataset, DataLoader

# ================= 1. 自动抓取所有数据路径 (保持不变) =================
data_dir = r"D:\AI_FYP\Data\ACDC\database\training"

data_dicts = []
print("正在扫描数据集目录...")

for patient_folder in sorted(os.listdir(data_dir)):
    patient_path = os.path.join(data_dir, patient_folder)
    if not os.path.isdir(patient_path) or not patient_folder.startswith("patient"):
        continue

    frame_dirs = glob.glob(os.path.join(patient_path, "patient*_frame*.nii"))
    img_dirs = [d for d in frame_dirs if "_gt" not in d]

    for img_dir in img_dirs:
        label_dir = img_dir.replace(".nii", "_gt.nii")
        img_files = glob.glob(os.path.join(img_dir, "*.nii"))
        label_files = glob.glob(os.path.join(label_dir, "*.nii"))

        if img_files and label_files:
            data_dicts.append({
                "image": img_files[0],
                "label": label_files[0]
            })

print(f"✅ 扫描完毕！成功找到 {len(data_dicts)} 对有效数据。")

# ================= 2. 定义数据清洗流水线 =================
preprocessing_pipeline = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # 🌟 姐姐加的核心大招：强制统一尺寸！
        # 把所有病人的心脏统统规范成 长256 x 宽256 x 深度10
        # mode: 图像用 trilinear 保持平滑，标签用 nearest 防止产生出 1.5 这种假类别
        Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),

        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ]
)

# ================= 3. 核心！批量打包引擎 (DataLoader) =================
dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ================= 4. 模拟模型训练时的抓取测试 =================
print("\n正在测试 DataLoader 批量输出...")

for batch_data in dataloader:
    imgs = batch_data["image"]
    labels = batch_data["label"]

    print(f"🚀 批量加载成功！")
    print(f"当前 Batch 图像的 Shape: {imgs.shape}")
    print(f"当前 Batch 标签的 Shape: {labels.shape}")
    print(f"图像像素最大值: {imgs.max().item():.4f}")
    print(f"图像像素最小值: {imgs.min().item():.4f}")

    break