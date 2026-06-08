import torch
from monai.networks.nets import UNet

print("正在初始化基础版 3D U-Net (Baseline)...")

# ================= 1. 搭建我们的“安全牌”基线模型 =================
model = UNet(
    spatial_dims=3,          # 这是一个处理 3D 数据的网络 (因为我们有 H, W, D)
    in_channels=1,           # 输入通道数: 1 (黑白原图)
    out_channels=4,          # 输出通道数: 4 (0=背景, 1=右心室, 2=心肌, 3=左心室)
    channels=(16, 32, 64, 128, 256), # 网络每一层提取的特征数量 (这决定了模型有多大)
    strides=(2, 2, 2, 2),    # 每次下采样的步长
    num_res_units=2          # 加上残差连接，防止网络太深变傻
)

# 打印一下模型参数量，让你直观感受下它有多大
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ 模型搭建成功！总参数量: {total_params:,}")

# ================= 2. 模拟数据输入 (Forward Pass) =================
print("\n正在模拟 DataLoader 吐出的数据...")

# 为了让你瞬间看到结果，我们不用再去读取硬盘了，直接用 PyTorch 凭空捏造一个假数据 (Dummy Input)
# 形状完全照抄你刚才跑出来的那个完美 Shape
dummy_input = torch.randn(4, 1, 256, 256, 16)
print(f"送入模型的输入形状: {dummy_input.shape}")

print("\n🚀 开始前向传播 (Forward Pass)... 模型正在疯狂计算...")

# 把数据喂给模型 (这其实就是未来训练时最核心的一行代码)
output = model(dummy_input)

# ================= 3. 验收成果 =================
print(f"🎉 模型吐出的预测结果形状: {output.shape}")