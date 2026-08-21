# P2 Method and Defense Cheat Sheet

这份材料用于理解和复述，不需要逐字背诵。答辩时先给结论，再给实现细节，最后主动说明边界。

## 最先背熟的两句话

**时空分析：** 项目包含两个明确分开的阶段。训练阶段是逐个 ED/ES 体积处理的空间 3D 分割，`D` 是解剖切片深度；完成后的 full-cine 阶段把冻结的空间 checkpoint 应用于 20 位验证病人的全部 550 个心动相位，再用固定的相邻帧概率融合计算体积曲线、ED/ES、EF 和全局运动代理指标。它不是 learned temporal Mamba、光流、配准、稠密位移或应变分析。

**图像处理：** 每个 3D 相位读取为 NIfTI，增加 channel 轴，MRI 用 trilinear resize 到 `256×256×16` 并按该相位 min--max 到 `[0,1]`。网络输出四类 softmax 概率；固定融合在概率层完成，再 `argmax`。cine 预测用 nearest-neighbour 恢复到原始数组尺寸，用 header voxel spacing 计算物理体积。没有裁剪、连通域筛选、空洞填充、配准或形变估计。

## 45-second English answer: what is the spatio-temporal analysis?

> The project has two distinct stages. The trained Nano-Mamba backbone is a
> spatial 3D segmenter: ED and ES are independent labelled samples, and depth
> means anatomical slices. For the completed full-cine stage, I froze the
> selected checkpoint and applied it to all 550 phases from the 20 validation
> patients. Each phase produced four-class softmax probabilities. I compared
> independent frame-wise masks with a fixed circular probability fusion of
> 0.25 previous, 0.50 current, and 0.25 next. After nearest-neighbour restoration
> to the native array shape, I derived chamber-volume curves, ED and ES timing,
> EDV, ESV, stroke volume, EF, LV-centroid displacement, and myocardial radial
> change. This is a real segmentation-derived full-cine analysis, but it is not
> a learned temporal network, optical flow, dense correspondence, or strain.

## 30-second English answer: how are the images processed?

> I load each NIfTI volume, extract one 3D cine phase when the source is 4D,
> add the channel and batch axes, and resize the MRI to 256 by 256 by 16 with
> trilinear interpolation. Each phase is min--max scaled to zero--one. The
> network returns four-class logits, which I convert to softmax probabilities.
> The frame-wise method takes argmax directly; the temporal method fuses adjacent
> probability maps before argmax. The categorical mask is restored to the native
> array shape by nearest-neighbour interpolation. Historical Dice is on the
> resized grid, while cine physical volumes use the native mask and NIfTI header
> voxel sizes. There is no crop, registration, deformation, or mask cleanup.

## Stage A：训练和历史六模型比较

1. `discover_acdc_cases` 找到 100 位病人、每人两个已标注 ED/ES 体积，共 200 cases。
2. seed 42 做 patient-level `80/20` split；同一病人的两个相位绝不跨集合。
3. `LoadImaged` 读取 image/label NIfTI。
4. `EnsureChannelFirstd`: `H×W×D → 1×H×W×D`。
5. `Resized` 到 `256×256×16`：MRI 用 trilinear，label 用 nearest-neighbour。
6. `ScaleIntensityd` 对每个 3D MRI 单独做 min--max；`ToTensord` 转 tensor。
7. Nano-Mamba 输入为 `B×1×256×256×16`；三次池化后 bottleneck 为 `B×128×32×32×2`。
8. `32×32×2=2,048` 个 spatial tokens；它们不是 cardiac time points。
9. 解码器输出 `B×4×256×256×16` logits；`argmax(dim=1)` 得到类别 `0/1/2/3`。
10. RV、MYO、LV Dice 在 resized grid 上计算；40 个 validation cases 汇总为主表。

## Stage B：完整 full-cine 分析

1. 对 patient-level validation split 中的 20 人读取 `Info.cfg`：`NbFrame`、ED、ES、诊断组。
2. 递归发现 shape 为 `(X,Y,Z,T)` 且 `T=NbFrame` 的 4D cine，不依赖文件名或文件大小猜测。
3. 对每个 `t=1…T` 提取一个 3D phase；共处理 `550/550` 个相位。
4. 每相位做 channel-first、trilinear resize、per-frame min--max、batch-one inference。
5. logits 经 softmax 得到 `P_t`。保留两条路径：
   - frame-wise：`argmax(P_t)`；
   - temporal fusion：先算下式，再 argmax。
6. 相位索引是 circular 的，因此第一个相位的 previous 是最后一个，最后一个的 next 是第一个。
7. categorical mask 用 nearest-neighbour 恢复到原始 `(X,Y,Z)` 数组尺寸。
8. 在 native grid 上逐帧统计 LV/RV/MYO 体积、LV centroid、mean MYO radius。
9. 用三帧 circular moving average 只平滑标量 LV-volume curve，再识别 phase 和计算功能指标。
10. 保存 1,100 条 frame rows（550 帧 × 2 方法）、40 条 patient rows（20 人 × 2 方法）、图、摘要、哈希和审计报告。

## 必须能在白板上写出的公式

**固定概率融合**

\[
\widetilde P_t=0.25P_{t-1}+0.50P_t+0.25P_{t+1}.
\]

权重固定、和为 1、无可学习参数；融合对象是四类概率图，不是最终 class ID。

**物理体积**

\[
v_{voxel}=\frac{s_xs_ys_z}{1000}\ \mathrm{mL},\qquad
V_{k,t}=N_{k,t}v_{voxel}.
\]

`s_x,s_y,s_z` 是 header 中的毫米 voxel sizes，`N_{k,t}` 是 native mask 中类别 `k` 的 voxel count。

**phase 与 EF**

\[
\widehat V_t=(V_{t-1}+V_t+V_{t+1})/3,
\]

`predicted ED = argmax(\widehat V)`，`predicted ES = argmin(\widehat V)`；误差用最短 circular frame distance。

\[
SV=EDV-ESV,\qquad EF=100\frac{EDV-ESV}{EDV}.
\]

有两种 EF，不能混淆：

- **annotated-phase EF**：在 `Info.cfg` 指定的 ED/ES 上，用 manual masks 算 reference，用预测 masks 算 predicted；用于 EF MAE 和 correlation。
- **curve-derived EF**：用预测 LV 曲线的最大/最小相位；它同时包含 segmentation 和 phase-selection 误差。

**全局运动代理指标**

\[
d_{peak}=\max_t\|c_{LV,t}-c_{LV,ED(reference)}\|_2.
\]

MYO radius 是每个 MYO voxel centre 到 LV centroid 的平均欧氏距离；报告 `radius_ED − radius_ES`。这些是 global geometry summaries，不建立相邻相位间的 material-point correspondence。

**曲线平滑度**

\[
S=\frac{\operatorname{mean}_t|\widehat V_{t+1}-2\widehat V_t+\widehat V_{t-1}|}
{\max(\widehat V)-\min(\widehat V)}.
\]

数值越小表示 global volume curve 的二阶波动越小；它不能单独证明中间帧解剖更准确。

## 如何证明这次 full-cine 不是“只跑出了图”

- 20/20 validation patients、550/550 expected phases 均完成；无缺病人、缺帧或重复帧。
- 4D cine 的 ED/ES 图像与历史 standalone endpoint 在执行归一化后完全一致，最大 normalized MAE 为 0。
- frame-wise endpoint Dice 与历史 Nano-Mamba case table 最大差异仅 `0.000169`。
- checkpoint SHA、split SHA、source lineage、每个 CSV/PNG SHA 都在 manifest 中。
- 独立脚本重新计算 frame rows、patient rows、volume、EF、phase、centroid、radius、smoothness、pathology summaries 和 bootstrap CI，结果为 `PASS`。
- patient002 overlay 来自 validation patient 和 epoch-121 checkpoint；它是可追踪例子，不是总体性能证明。

## 结果必须能脱口而出

### 空间分割

- SegResNet16：**86.70%**，最高 validation mean Dice。
- No-Mamba：**85.64%**，比 Nano 高 **0.862 pp**，但参数更多且结构/批大小混杂。
- Nano-Mamba：**84.78%**，**1.456M** reported parameters，最小模型。
- UNet3D：**80.83%**，**4.809M**；Nano 相比它 **+3.945 pp**，参数少 **69.714%**。

### Full-cine：frame-wise / fixed fusion

| Metric | Frame-wise | Fusion |
|---|---:|---:|
| Resized endpoint Dice | 84.779% | 84.794% |
| Native endpoint Dice | 77.919% | 77.923% |
| Annotated-phase EF MAE | 3.514 pp | 3.329 pp |
| EF Pearson r | 0.9809 | 0.9803 |
| ED exact / within ±1 | 50% / 80% | 55% / 80% |
| ES exact / within ±1 | 55% / 85% | 55% / 85% |
| Curve smoothness | 0.032521 | 0.031392 |
| Median peak LV-centroid displacement | 6.361 mm | 6.252 mm |
| Mean MYO radius ED−ES | 4.250 mm | 4.153 mm |

### Paired fusion minus frame-wise

- Endpoint Dice：`+0.0144 pp`, 95% CI `[-0.0635,+0.0819] pp`；跨 0，不能说 accuracy improved。
- EF absolute error：`−0.1853 pp`, 95% CI `[-0.4678,+0.0909] pp`；跨 0，不能说 EF improved。
- Curve smoothness：`−0.001129`, 95% CI `[-0.001950,−0.000511]`；完整小于 0，**19/20 patients smoother**。
- 所以最强结论是：fixed fusion 让 global LV-volume trajectories 更平滑；不是 Dice/EF 优势，也不是 learned cardiac dynamics。

## 失败案例与数据边界

- `patient016` 是 phase detection 主 outlier：fusion ED error 5 frames、ES error 3 frames；不能删除。
- `patient057` 的 fusion annotated-phase EF absolute error 最大，约 **8.12 pp**。
- 中间 510 个相位没有 manual segmentation mask；只能在 ED/ES 做 overlap validation。
- 六个恢复的公开 ACDC 4D cine 与 standalone endpoints 的 raw affine metadata 不同，但 array shape、header zooms、endpoint pixel identity、endpoint image-label alignment 均通过。体积使用 header zooms；全局距离使用由 zooms 和可靠方向构造的 metric transform。不能据此声称 world-coordinate tissue tracking。
- pathology groups 仅描述：MINF 7、NOR 6、DCM 4、RV 2、HCM 1；样本太小且不平衡，不能做 subgroup significance/generalization claim。

## 老师追问时的短答

**你到底完成时空分析了吗？**

完成了界限明确的 segmentation-derived full-cine analysis：20 人、550 帧、两种推理路径、物理体积、phase、EF 和 global motion curves 均生成并独立复算。没有完成 learned temporal representation、dense motion 或 strain；论文和演示没有把它们说成已完成。

**为什么在概率上融合，不在 mask 上 majority vote？**

softmax 保留每类置信度，线性组合后再 argmax；hard mask vote 会提前丢失不确定性，而且三帧加权的当前相位偏好不如概率公式直接。

**为什么使用 circular boundary？**

cine 是周期序列；第一个和最后一个相位在心动周期上相邻。非 circular padding 会人为制造序列边界。

**为什么还要对 LV scalar curve 做三帧平均？不是重复 smoothing 吗？**

两者作用域不同。概率融合改变 segmentation path；三帧 scalar average 只用于两种方法共同的 phase detection，降低单帧峰值抖动。比较仍在同一 phase rule 下进行。

**为什么 native Dice 比 resized Dice 低？**

模型在固定 resized grid 上训练和预测；恢复到 heterogeneous native shapes 时 nearest-neighbour 显露了 geometry/resampling loss。两者回答不同问题，所以都报告，不能只保留较高者。

**centroid displacement 是心肌运动吗？**

它是 LV cavity mask 的 global positional surrogate；MYO radius 是 global radial-extent surrogate。没有配准或 correspondence，因此不能解释为局部 myocardium trajectory 或 strain。

**为何说 EF 被验证？**

reference EF 由 manual ED/ES masks 计算；predicted EF 在同一 annotated phases 上由预测 masks 计算。MAE 和 correlation 是 internal endpoint validation，不是 external clinical validation。

**为什么 fusion 没提高 Dice 还保留？**

因为预设问题不只是 endpoint accuracy。它在 19/20 病人上降低 global curve second-difference，形成透明的 temporal regularization baseline；同时负结果被完整报告。

## 不要说的句子

- “Depth is time.”
- “The Nano-Mamba bottleneck learns cardiac temporal dynamics.”
- “This is full Mamba/selective scan.”
- “Temporal fusion significantly improves Dice or EF.”
- “Centroid displacement is dense tissue tracking.”
- “MYO radius change is myocardial strain.”
- “All 550 frames have manual labels.”
- “The 20 patients are an independent test set.”
- “Nano-Mamba is the most accurate or fastest model.”
- “Affine mismatch was ignored or overwritten.”

## 代码—证据—论文—演示映射

| Topic | Code / Evidence | Thesis | Slides |
|---|---|---|---|
| Spatial preprocessing | `src/21_rigorous_experiment_pipeline.py` | Ch. 5 preprocessing | 5 |
| Spatial bottleneck | `src/nano_mamba_core.py` | Ch. 5 architecture | 4 |
| Split / six-model result | `evidence/rigorous_patient_split/` | Ch. 5–6 | 3, 6–8 |
| Cine discovery / identity | `src/23_spatiotemporal_cine_analysis.py`, `run_metadata.json` | Ch. 5 full-cine cohort | 9, 12 |
| Function / motion formulas | `src/cardiac_motion_metrics.py` | Ch. 5 full-cine method | 9, 11 |
| Full-cine rows/results | `evidence/spatiotemporal_cine/raw/` | Ch. 6 | 10–13 |
| Independent recomputation | `src/25_spatiotemporal_result_audit.py`, `INDEPENDENT_AUDIT.json` | Ch. 6 evidence boundary | 10–12 |
| Scientific limits | `SCIENTIFIC_BOUNDARIES.md` | Ch. 6–7 | 12–14 |
