import nibabel as nib
import numpy as np

# 替换成你真实的 ACDC 图像路径
file_path = r"D:\AI_FYP\Data\ACDC\database\training\patient001\patient001_frame01.nii\CMD03Gate1.nii"

# 1. 加载医学图像
img = nib.load(file_path)

# 2. 获取图像的仿射矩阵 (以后做空间对齐时极其重要)
affine = img.affine
print(f"图像仿射矩阵 (Affine):\n{affine}")

# 3. 把图像转换为 Numpy 数组
img_data = img.get_fdata()

# 4. 打印核心信息
print(f"\n图像的三维形状 (Shape): {img_data.shape}")
print(f"图像的最大像素值: {np.max(img_data)}")
print(f"图像的最小像素值: {np.min(img_data)}")