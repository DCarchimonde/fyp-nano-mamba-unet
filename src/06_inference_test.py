import os
import glob
import torch
import matplotlib

matplotlib.use('TkAgg')  # 解决你电脑上 PyCharm 的画图报错
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)

# ================= 1. 准备环境与加载模型 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\AI_FYP\models\best_baseline_unet.pth"

print("🧠 正在唤醒沉睡的最强模型...")
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=4,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
).to(device)

# 关键动作：把存在硬盘里的“灵魂(权重)”注入到这副躯壳里
model.load_state_dict(torch.load(model_path))
model.eval()  # 极其重要：告诉模型“现在是考试，不是训练，不要乱改参数了！”
print("✅ 模型唤醒成功！进入考试模式。")

# ================= 2. 抓取一个病人作为考题 =================
# 咱们抓 patient002 出来考考它
data_dir = r"D:\AI_FYP\Data\ACDC\database\training"
patient_path = os.path.join(data_dir, "patient002")
frame_dirs = glob.glob(os.path.join(patient_path, "patient*_frame*.nii"))
img_dirs = [d for d in frame_dirs if "_gt" not in d]

img_dir = img_dirs[0]
label_dir = img_dir.replace(".nii", "_gt.nii")
img_file = glob.glob(os.path.join(img_dir, "*.nii"))[0]
label_file = glob.glob(os.path.join(label_dir, "*.nii"))[0]

# ================= 3. 数据清洗 (跟训练时一模一样) =================
print("📦 正在给考题做预处理...")
transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("trilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])

data_dict = transform({"image": img_file, "label": label_file})

# 给数据套上一个 Batch 维度 (因为模型只吃 5 维的张量: [Batch, Channel, H, W, D])
# unsqueeze(0) 就是在最前面加个 1，变成 [1, 1, 256, 256, 16]
img_tensor = data_dict["image"].unsqueeze(0).to(device)
label_tensor = data_dict["label"].unsqueeze(0).to(device)

# ================= 4. 高能时刻：让 AI 开始画图 =================
print("🚀 AI 正在读图并生成预测...")
with torch.no_grad():  # 考试时严禁计算梯度，省显存！
    output = model(img_tensor)  # output shape: [1, 4, 256, 256, 16]

    # 核心大招：由于模型吐出的是 4 张概率图，我们用 argmax 挑出每一块像素概率最大的那个类别！
    pred_mask = torch.argmax(output, dim=1)  # 瞬间变成 [1, 256, 256, 16] 的彩色标签图

# ================= 5. 将结果画出来对比 =================
print("🎨 正在渲染对比图...")
# 我们取正中间的一层切片来看 (第 8 层)
slice_idx = 8

# 把数据从显卡(GPU)拿回到内存(CPU)，转成 Numpy 才能画图
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
plt.title('AI Prediction (Your U-Net!)')
plt.axis('off')

plt.tight_layout()
plt.show()