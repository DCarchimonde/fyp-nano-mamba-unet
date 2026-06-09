import torch
import torch.nn as nn

# 🚀 极其关键：把你刚刚炼制好的 Mamba 核心脏导进来！
from nano_mamba_core import SpatioTemporalMambaBottleneck


class DoubleConv(nn.Module):
    """
    U-Net 的基础砖块：连续两次卷积提取特征
    """

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
    """
    🏆 你的专属毕业设计大杀器：Spatio-Temporal Nano-Mamba U-Net
    """

    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()

        # ================== 1. 编码器 (Encoder): 提取空间特征 ==================
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        # ================== 2. 瓶颈层 (Bottleneck): 注入 Mamba 灵魂 ==================
        # 传统 U-Net 这里只是普通卷积，而咱们在这里接入了时空 Mamba 模块！
        self.bottleneck = nn.Sequential(
            DoubleConv(64, 128),
            # 就是它！让你的模型学会看“连贯视频”而不是“单张静态图片”
            SpatioTemporalMambaBottleneck(channels=128)
        )

        # ================== 3. 解码器 (Decoder): 还原图像细节 ==================
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)  # 128 是因为拼接了左边的特征 (64+64)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        # ================== 4. 预测头 ==================
        # 输出 4 张概率图 (背景、右心室、心肌、左心室)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # 走左边 (疯狂压缩提取特征)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # 走最底部的 Mamba 核心脏 (理解时空连贯性)
        b = self.bottleneck(self.pool3(e3))

        # 走右边 (疯狂放大还原，并拼接左边的特征)
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)  # 标志性的 U-Net 跳跃连接 (Skip Connection)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ================= 火力测试 =================
if __name__ == "__main__":
    print("🚀 正在组装终极武器：Nano-Mamba U-Net...")
    model = NanoMambaUNet()

    # 算一下参数量，体现我们 "Nano (轻量级)" 的优势
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型组装成功！总参数量: {total_params:,} (绝对算得上 Nano 级别！)")

    print("\n模拟 DataLoader 喂入数据...")
    # 模拟输入: Batch=2, Channel=1, 宽=256, 高=256, 深度(切片数)=16
    dummy_input = torch.randn(2, 1, 256, 256, 16)
    print(f"输入形状: {dummy_input.shape}")

    print("\n模型正在进行全链路前向传播 (Mamba 时空建模中)...")
    output = model(dummy_input)

    print(f"🎉 完美吐出预测结果！输出形状: {output.shape}")