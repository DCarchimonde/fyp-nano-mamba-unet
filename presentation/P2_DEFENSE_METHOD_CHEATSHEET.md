# P2 Method and Defense Cheat Sheet

This is a study aid, not a script that must be read word for word. The goal is
to understand the pipeline well enough to answer follow-up questions naturally.

## 先背住这两句

**关于时空分析：** 本实验没有做真正的时间建模。每位病人的 ED 和 ES
是两个独立的 3D 样本；模型里的 depth 是心脏切片深度，不是时间。当前工作是
后续心功能/运动分析之前的分割步骤。

**关于图片处理：** NIfTI 图像和标签配对读取，转成 channel-first，图像用
trilinear、标签用 nearest-neighbour 统一缩放到 `256×256×16`，图像逐体积
min--max 到 `[0,1]`，再转 tensor。没有数据增强、方向/物理间距统一、心脏裁剪、
原尺寸回映射或掩膜后处理。

## 30-second English answers

**Spatio-temporal question**

> The completed experiment does not model cardiac time. Each labelled ED or ES
> volume is an independent 3D sample, and the depth axis contains anatomical
> slices. The registered title reflects the wider cardiac-motion motivation;
> the evaluated contribution is spatial 3D segmentation as a prerequisite for
> later functional analysis.

**Image-processing question**

> I load each paired NIfTI image and label, add the channel dimension, and
> resize both to 256 by 256 by 16. The image uses trilinear interpolation and
> volume-wise min--max scaling to zero--one; the label uses nearest-neighbour
> interpolation. I then convert both to tensors. There is no augmentation,
> spacing or orientation normalization, inverse resampling, or mask
> post-processing, so Dice is measured on the resized grid.

## 从磁盘到 Dice：按代码顺序讲

1. `discover_acdc_cases` 查找 100 位病人的图像/标签配对；每人必须正好两例。
2. seed 42 做 patient-level 80/20 split；同一病人的 ED/ES 绝不会跨集合。
3. `LoadImaged`: 读取 image NIfTI 和 label NIfTI。
4. `EnsureChannelFirstd`: `H×W×D → 1×H×W×D`。
5. `Resized`: 两者都到 `256×256×16`。
   - image: trilinear，因为强度是连续量；
   - label: nearest，因为类别必须保持 `0/1/2/3`。
6. `ScaleIntensityd`: 单个 MRI 体积 min--max 到 `[0,1]`。
7. `ToTensord`: DataLoader 输入为 `B×1×256×256×16`。
8. Nano-Mamba 编码器三次池化，bottleneck 形状为
   `B×128×32×32×2`。
9. 空间网格展平为 `B×2048×128`；2048 是空间 token，不是时间点。
10. 解码器输出 `B×4×256×256×16` logits。
11. `argmax(dim=1)` 得到 `0/1/2/3` 类别图；无阈值和后处理。
12. 在 resized grid 上分别算 RV/MYO/LV Dice，再汇总 40 个 validation cases。

## 实验到底完整到什么程度

已经完成并可支持论文主表：

- 100 patients / 200 labelled ED-ES cases；
- patient-level split，无病人泄漏；
- 六个模型、统一 150 epochs、统一主要优化设置；
- 每模型 40 条 validation case metrics；
- 每模型 150 epoch curves 与 best checkpoint epoch 一致；
- 主表、类别 Dice、参数量、FPS 和图表互相一致。

真实科学限制（答辩时主动承认，不要回避）：

- validation 同时用于选 checkpoint 和报告结果，不是 independent test；
- 只有一个 split / seed；
- 没有 data augmentation；
- 没有 `Orientationd` / `Spacingd`，也没有 native-space Dice；
- batch size、normalization、模型容量不完全匹配；
- 没有 ED-vs-ES、病种分层、HD95/ASD 或正式 qualitative review；
- 完全没有 temporal tracking / optical flow / displacement / EF。

这些是实验设计限制，不是“缺少旧命令或完整历史日志”。历史取证不是论文提交门槛。

## 结果必须能脱口而出

- SegResNet16: **86.70%**，最高 Dice。
- No-Mamba: **85.64%**，比 Nano 高 **0.862 pp**，但参数更多且结构不匹配。
- Nano-Mamba: **84.78%**, **1.456M** 参数，最小模型。
- UNet3D: **80.83%**, **4.809M** 参数。
- Nano 对 UNet3D: **+3.945 pp Dice**, **-69.714% parameters**。
- Nano 最差 case: `patient049_frame11`, mean **57.86%**, RV **26.04%**。
- Attention U-Net 最明显 late-epoch drop: best 到 epoch 150 下降 **3.05 pp**。

## 老师常见追问的短答案

**为什么 resize 到 16 slices？**

为了让不同深度的体积形成固定大小 batch，并满足 3D 网络的内存限制。代价是没有
保留统一物理 spacing，因此可能发生几何变形。

**为什么标签不用 trilinear？**

会产生 1.4、2.6 这样的伪类别；nearest-neighbour 保留整数类别。

**为什么不做 augmentation？**

历史实验使用确定性预处理，所有模型保持一致；这保证比较口径一致，但削弱了鲁棒性
证据，后续应加入同一套训练增强后重训全部模型。

**Mamba 到底做什么？**

把低分辨率 3D 特征按固定 raster order 展成 spatial sequence，用 depthwise 1D
convolution 和 input-dependent scalar gate 混合，再 residual add 并恢复 3D。

**为什么不能说 gate 提高准确率？**

No-Mamba 的 Dice 更高，而且参数量、bottleneck 结构都不同；没有 capacity-matched
和 multi-seed 设计，不能做因果归因。

**你做了 motion analysis 吗？**

没有。只分割 ED/ES；没有联合输入两相、位移场、光流、时序一致性或 EF。

**为什么 validation 不是 test？**

因为每一 epoch 都看 validation Dice，并用它挑 best checkpoint；最终表又用同一组。

## 不要说的句子

- “Depth is time.”
- “The model performs spatio-temporal tracking.”
- “This is a full Mamba/selective-scan implementation.”
- “Nano-Mamba is the most accurate model.”
- “The ablation proves Mamba improves accuracy.”
- “The 20 patients are an independent or official test set.”
- “1.456M parameters means it must be the fastest.”

## 代码—论文—演示映射

| Topic | Code | Thesis | Slides / Q&A |
|---|---|---|---|
| ED/ES are independent | `discover_acdc_cases`, one case per frame | Ch. 1 scope; Ch. 5 dataset; Ch. 7 motion boundary | Slides 2, 12; Q1--4 |
| Image preprocessing | `build_transform` | Ch. 5 Preprocessing Pipeline | Slide 5; Q5--12 |
| Tensor/token shape | `NanoMambaUNet.forward`, `MambaInspiredBottleneck.forward` | Ch. 5 architecture/bottleneck | Slide 4; Q13--17 |
| Patient split | `create_or_load_split`, `validate_split` | Ch. 5 Patient-Level Split | Slide 3; Q18--19 |
| Loss and optimization | `train_one_model` | Ch. 5 Training Objective | Q20--25 |
| Dice calculation | `dice_per_case`, `evaluate` | Ch. 5 Evaluation Metrics | Q21--23 |
| Main results | `summary_metrics.csv` | Ch. 6 | Slides 6--10; Q26--32 |
| Limitations | executed omissions and comparison design | Ch. 7 Limitations | Slide 11; Q33--35 |
