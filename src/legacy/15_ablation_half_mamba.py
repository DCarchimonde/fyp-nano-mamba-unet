import torch
import torch.nn as nn
import time
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================= 1. 基础组件 =================
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


# ================= 2. 你的第二号消融选手：半血版 Mamba =================
class Ablation_HalfMamba_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        # 🌟 核心修改区：通道数减半的 Mamba 瓶颈层
        self.bottleneck = nn.Sequential(
            DoubleConv(64, 64),  # 先不升维，保持 64
            SpatioTemporalMambaBottleneck(channels=64),  # 喂给“半血版” Mamba
            DoubleConv(64, 128)  # 出来之后再升回 128，为了跟右边的解码器对齐
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


# ================= 3. 学术测速仪 =================
def academic_benchmark(model, name, device, input_shape=(1, 1, 256, 256, 16), num_runs=30):
    print(f"\n[{name}] 踏入角斗场...")
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            _ = model(dummy_input)
        max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        with torch.no_grad():
            for _ in range(5): _ = model(dummy_input)

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs): _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / num_runs) * 1000
        fps = 1000 / avg_time_ms

        print(f"  ⭐ 参数量:   {total_params:.2f} M (百万)")
        print(f"  🔥 峰值显存: {max_vram_mb:.2f} MB")
        print(f"  ⚡ 推理速度: {fps:.2f} FPS")

    except RuntimeError as e:
        print(f"  ❌ 运行出错: {e}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 补齐最后一块拼图：Mamba 通道减半消融实验！")

    half_mamba = Ablation_HalfMamba_UNet()
    academic_benchmark(half_mamba, "Ablation: 半血版 Mamba (64 Channels)", device)

    print("\n✅ 所有的坑都填平了老弟！没有任何遗漏！")