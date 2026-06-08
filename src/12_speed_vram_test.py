import torch
import torch.nn as nn
import numpy as np
import time
from monai.networks.nets import UNet
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 把两个选手的躯壳准备好 =================
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


class NanoMambaUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)
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


# ================= 2. 姐姐写的极其严谨的压测函数 =================
def benchmark_model(model, name, device, input_shape=(1, 1, 256, 256, 16), num_runs=100):
    print(f"\n[{name}] 正在登场，准备接受极限压测...")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    # 1. 测量显存占用 (VRAM)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # 2. GPU 预热 (必须做，否则前几次运行会很慢，测试不准)
    print("  -> 正在预热 GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 3. 正式测速
    print(f"  -> 开始连续执行 {num_runs} 次推理...")
    torch.cuda.synchronize()  # 确保之前的计算全做完
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    torch.cuda.synchronize()  # 等待所有计算彻底完成
    end_time = time.time()

    # 4. 计算结果
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_runs) * 1000
    fps = 1000 / avg_time_ms

    print("-" * 40)
    print(f"🏆 {name} 性能报告:")
    print(f"   峰值显存占用 (VRAM): {max_vram_mb:.2f} MB")
    print(f"   单次推理耗时:        {avg_time_ms:.2f} ms")
    print(f"   帧率 (FPS):          {fps:.2f} 帧/秒")
    print("-" * 40)


# ================= 3. 开启双王对决 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 顶级医疗影像 AI 性能压测中心启动！")

    # 测试 1: 基础版巨无霸 U-Net
    baseline_model = UNet(
        spatial_dims=3, in_channels=1, out_channels=4,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
    )
    benchmark_model(baseline_model, "传统 3D U-Net (Baseline)", device)

    # 必须清空显存缓存，保证公平
    del baseline_model
    torch.cuda.empty_cache()

    # 测试 2: 你的专属轻量级杀器 Nano-Mamba
    nanomamba_model = NanoMambaUNet()
    benchmark_model(nanomamba_model, "Nano-Mamba U-Net (Ours)", device)

    print("\n✅ 压测全部完成！请将这些硬核数据记录到你的论文中！")