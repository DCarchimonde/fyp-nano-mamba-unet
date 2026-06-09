import torch
import torch.nn as nn
import time
# 引入医疗影像界的两大霸主 (SOTA)
from monai.networks.nets import AttentionUnet, SwinUNETR
from nano_mamba_core import SpatioTemporalMambaBottleneck


# ================== 1. 准备咱们的消融实验选手 (Ablation Models) ==================
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


# 消融实验 1号：拔掉 Mamba 心脏，换成普通的双层卷积 (看看 Mamba 到底有没有用)
class Ablation_NoMamba_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        # 【修改处】原本的 Mamba 被替换成了普通卷积
        self.bottleneck = nn.Sequential(
            DoubleConv(64, 128),
            DoubleConv(128, 128)
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


# ================== 2. 极其严谨的学术测速仪 ==================
def academic_benchmark(model, name, device, input_shape=(1, 1, 256, 256, 16), num_runs=30):
    print(f"\n[{name}] 踏入角斗场...")
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # 转换成 M (百万)

    # 测量显存占用
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            _ = model(dummy_input)
        max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # 预热并测速
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
        if "Out of memory" in str(e):
            print(f"  ❌ 显存爆炸 (OOM)! 你的 4060 显卡撑不住这个巨无霸！")
            torch.cuda.empty_cache()
        else:
            print(f"  ❌ 运行出错: {e}")

    # 打扫战场，释放显存
    del model
    del dummy_input
    torch.cuda.empty_cache()


# ================== 3. 开启诸神之战 ==================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 医疗影像 AI 终极对比实验 (SOTA & Ablation) 启动！")

    # --- 战场 A：SOTA 巨无霸对比 ---
    print("\n" + "=" * 50)
    print("⚔️ 第一场仗：对决当前世界最强模型 (SOTA)")
    print("=" * 50)

    # 1. 著名的 Attention U-Net
    att_unet = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=4,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
    )
    academic_benchmark(att_unet, "Attention U-Net (2018 经典王者)", device)

    # 2. 算力黑洞：Swin-UNETR (Transformer 架构)
    print("  -> 尝试唤醒 Swin-UNETR...")
    try:
        # 尝试新版 MONAI 的写法
        swin_unetr = SwinUNETR(
            img_size=(256, 256, 16), in_channels=1, out_channels=4, feature_size=24
        )
    except TypeError:
        try:
            # 尝试老版 MONAI 的写法
            swin_unetr = SwinUNETR(
                image_size=(256, 256, 16), in_channels=1, out_channels=4, feature_size=24
            )
        except TypeError:
            print("  ❌ 警告: 你的 MONAI 版本可能太老了，Swin-UNETR 接口不匹配。")
            print("  -> 姐姐建议：打开终端运行 `pip install monai --upgrade` 升级一下！")
            swin_unetr = None  # 如果实在不行，就先跳过它，防止程序崩溃

    if swin_unetr is not None:
        academic_benchmark(swin_unetr, "Swin-UNETR (当前 Transformer 霸主)", device)

    # --- 战场 B：我们自己的消融实验 ---
    print("\n" + "=" * 50)
    print("🔬 第二场仗：内部消融实验 (Ablation Study)")
    print("=" * 50)

    # 3. 拔掉 Mamba 的模型
    no_mamba = Ablation_NoMamba_UNet()
    academic_benchmark(no_mamba, "Ablation: 移除 Mamba 模块 (纯卷积)", device)

    # 4. 这个位置你未来也可以填入通道数减半的 Mamba 进行测速...

    print("\n✅ 全系列硬核工业级压测结束！你的论文素材库彻底爆炸了！")