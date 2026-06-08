import torch
import time
# 直接引入医学影像挑战赛的常客：SegResNet
from monai.networks.nets import SegResNet


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
    print("🚀 召唤现代 SOTA 霸主补齐实验矩阵！")

    # 初始化 SegResNet
    # init_filters=32 保证了它具备现代 3D 卷积网络的庞大特征提取能力
    segresnet = SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=4,
        dropout_prob=0.2,
    )

    academic_benchmark(segresnet, "SegResNet (现代 3D 医学分割霸主)", device)

    print("\n✅ 第二位 SOTA 巨头数据获取完毕！")