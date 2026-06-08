import nibabel as nib
import numpy as np
import matplotlib
# 强制使用 Tkinter 独立窗口画图，绕开 PyCharm 的 Bug
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ================= 下面是你原本的代码，一字都不用改 =================

# 1. 替换成你的文件路径 (注意路径前面的 r 不要删)
img_path = r"D:\AI_FYP\Data\ACDC\database\training\patient001\patient001_frame01.nii\CMD03Gate1.nii"
label_path = r"D:\AI_FYP\Data\ACDC\database\training\patient001\patient001_frame01_gt.nii\DCM03-OH-AL_V2_1.nii"

# 2. 读取数据
img_data = nib.load(img_path).get_fdata()
label_data = nib.load(label_path).get_fdata()

print(f"原始图像 Shape: {img_data.shape}")
print(f"标签图像 Shape: {label_data.shape}")

# 3. 极其严苛的防御性断言：确保图和标签尺寸完美一致
assert img_data.shape == label_data.shape, "老弟！原图和标签的尺寸对不上，快检查路径！"

# 4. 提取中间的一层切片来可视化 (因为一共有10层，我们取第5层)
slice_idx = img_data.shape[2] // 2

img_slice = img_data[:, :, slice_idx]
label_slice = label_data[:, :, slice_idx]

# 5. 用 Matplotlib 把它们画出来
plt.figure(figsize=(12, 6))

# 画原图
plt.subplot(1, 2, 1)
plt.imshow(img_slice, cmap='gray')
plt.title(f'Raw MRI Image (Slice {slice_idx})')
plt.axis('off')

# 画标签
plt.subplot(1, 2, 2)
# 使用 nipy_spectral 色图可以让不同的类别(心室、心肌)显示出鲜艳的颜色
plt.imshow(label_slice, cmap='nipy_spectral')
plt.title(f'Ground Truth Label (Slice {slice_idx})')
plt.axis('off')

plt.tight_layout()
plt.show()