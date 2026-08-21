# Methodology and Typesetting Deep Audit

Audit date: 22 August 2026

Scope: the eleven high-risk items raised before submission, plus adjacent
code--paper inconsistencies found during the same review. The verdicts below
refer to the executed Stage-A training pipeline, the sealed Stage-B full-cine
run, the active LaTeX sources, and the canonical thesis PDF.

## Executive verdict

No error was found in the executed circular frame indexing, millilitre
conversion, seed entry point, Nano-Mamba trainable-parameter count, or
native-mask axis order. The reported six-model experiment and 20-patient,
550-frame cine analysis remain numerically valid within their stated internal
validation scope.

Four groups of thesis formulas were nevertheless rewritten to remove fragile
math-font dependencies and to make operator scope unambiguous. The audit also identified
two genuine methodological qualifications that must remain visible:

1. the six-model comparison mixes batch size and normalization family, so it
   is not a controlled causal architecture ablation; and
2. the sealed cine run stored softmax probability maps in `float16` before
   fixed temporal fusion, although logits/softmax and fusion arithmetic were
   computed in `float32` and the frame-wise argmax was retained before storage.

Neither qualification invalidates the reported measurements. Both narrow the
claims that can be made from them.

## A. LaTeX and PDF findings

| Raised item | Verdict | Evidence and final correction |
|---|---|---|
| Section 5.3 train/validation sets appeared as empty-set glyphs | Viewer/font portability risk closed | The active source represented the sets with script glyphs. The equations now use robust bold `P` notation with upright `train` and `val` subscripts. Their intersection is empty and their union is the complete 100-patient set. |
| Section 5.8 loss symbols appeared as `Z`/empty-set glyphs | Viewer/font portability risk closed | Equations 5.13--5.15 now use plain semantic `L_Dice`, `L_CE`, and `L_total`, with `DSC` typeset as an operator. No active chapter source depends on `\mathcal`. |
| Operators below Equations 4.4 and 5.11 disappeared | Viewer/font portability risk closed | Convolution, concatenation, upsampling, sigmoid, projection, and activation are now written with `\operatorname{...}`. Componentwise gating is also stated by the explicit scalar equation `M_{b l c}=H_{b l c} G_{b l c} SiLU(V_{b l c})`, so meaning does not depend on a special product glyph. |
| Equation 4.8 mixed the attention score and `V` | Mathematical ambiguity closed | It is now `Attention(Q,K,V)=[softmax(QK^T/sqrt(d_k))]V`, with explicit square brackets and a sentence defining `d_k` and the softmax scope. |

The canonical PDF is rebuilt from a clean temporary directory after these
changes. PDF acceptance requires embedded fonts, no undefined references or
citations, no overfull boxes, text extraction of the corrected symbols, and
visual inspection of every rendered page.

## B. Python and experimental-method findings

### 1. Batch size one and batch normalization

**Verdict: real comparison confound; not a crash or invalid variance bug.**

- The executed Attention U-Net constructor contains `BatchNorm3d` and used
  training batch size one.
- `BatchNorm3d` computes per-channel statistics over the batch and spatial
  elements of the 5D input. A one-volume batch therefore does not by itself
  imply an empty sample set or force every variance to zero.
- The recovered Attention U-Net log contains 150 finite epochs and its selected
  checkpoint is epoch 62, confirming that the executed configuration did not
  crash.
- SegResNet16 also used batch size one, but its executed default is GroupNorm
  with eight groups, not BatchNorm.
- MONAI 3D U-Net used InstanceNorm; the three custom U-Net variants used
  `BatchNorm3d` at batch size two.

The thesis now reports this contract explicitly. Since normalization, batch
size, optimizer-step count, and capacity differ, Table 6.1 is an executed-system
comparison rather than a parameter-matched causal test of one architecture.

### 2. Circular temporal-fusion boundaries

**Verdict: implementation correct.**

The executed code uses
`(frame_index - 1) % num_frames` and
`(frame_index + 1) % num_frames`. New regression tests separately prove that
the final frame contributes to `t=0` and the first frame contributes to
`t=T-1`. This matches the periodic cardiac-cycle equation.

### 3. Physical volume in millilitres

**Verdict: implementation correct.**

The geometry function computes
`voxel_volume_ml = prod(spatial_zooms_mm) / 1000.0`; class volume is the native
mask voxel count multiplied by that value. The independent Stage-B audit
recomputes the volumes and rejects non-finite or non-positive voxel volume.

### 4. Global seed and deterministic flags

**Verdict: every flag stated in the thesis is present; bitwise determinism is
not claimed.**

The Stage-A entry point calls `random.seed`, `numpy.random.seed`,
`torch.manual_seed`, and `torch.cuda.manual_seed_all`, sets
`cudnn.deterministic=True` and `cudnn.benchmark=False`, and uses
`num_workers=0` for both loaders. It does not call
`torch.use_deterministic_algorithms`; therefore the defensible statement is
controlled seeding and deterministic CuDNN configuration, not guaranteed
bitwise identity across hardware or library versions.

### 5. Resized-grid to native-grid Dice reduction

**Verdict: no axis-order or interpolation-mode bug found; attribution solely to
the Z axis is rejected.**

The sealed implementation converts the categorical mask to shape
`[1,1,X,Y,Z]`, calls nearest-neighbour interpolation with the exact original
three-axis shape, and removes the two leading singleton dimensions. No
transpose or class-blending interpolation occurs. A new non-cubic regression
test checks output shape, corner orientation, and class preservation.

The frame-wise endpoint mean changes from 84.779% on the resized grid to
77.919% after native restoration, a 6.861 percentage-point difference. The
class-mean differences are approximately 5.917 pp for RV, 10.248 pp for MYO,
and 4.418 pp for LV. The largest change in the thin myocardium is consistent
with boundary discretization, but the available data do not support saying
that Z-axis anisotropy alone caused the entire change. Direct in-plane and
through-plane resizing, heterogeneous native dimensions, boundary thickness,
and prediction error all contribute.

`src/26_native_grid_roundtrip_audit.py` applies the exact nearest-neighbour
native-to-resized-to-native path to all 40 manual endpoint masks. It quantifies
the resampling component without retraining; its result is diagnostic rather
than a universal upper bound on restored prediction Dice.

### 6. Exact Nano-Mamba parameter count

**Verdict: exact.**

`sum(p.numel() for p in model.parameters())` equals **1,456,325**. The
checkpoint state dictionary contains 1,457,747 scalar entries because it also
stores 1,422 BatchNorm running-statistic and batch-counter buffers. Buffers are
not trainable parameters, so this is not a mismatch. The thesis now states
both numbers and their interpretation.

### 7. Six cine/endpoint affine mismatches

**Verdict: handled safely for the reported scalar measurements; unsuitable for
dense correspondence claims.**

The affected validation patients are `patient002`, `patient042`, `patient049`,
`patient066`, `patient071`, and `patient073`. The preflight requires matching
spatial shape, header voxel sizes, endpoint image content, and endpoint
image--label alignment. Those checks passed; the maximum normalized endpoint
image error was zero. The code records rather than overwrites raw affine
differences.

Volumes use header spacings. Euclidean global distances use the reliable
orthonormal direction component and spacings; no dense voxel correspondence,
local tissue path, deformation field, or strain is claimed. A raw affine
mismatch would be unacceptable for those stronger analyses, which this thesis
does not perform.

## C. Additional issue found during the audit

### Probability-map storage precision before fusion

The full-cine run disabled AMP, computed model logits and softmax in float32,
and preserved each frame-wise argmax before conversion. It then copied the
probability tensor to CPU as float16 and converted neighbouring maps back to
float32 for the weighted fusion. Thus, the reported fusion result includes
float16 storage quantization.

This does not change the already preserved frame-wise masks and does not
justify discarding the completed run. It does mean that the very small fusion
changes in endpoint Dice and EF must not be advertised as
precision-independent improvements. The thesis now says exactly this; the
supported temporal result remains smoother global LV-volume curves, not an
accuracy or EF gain.

## D. Regression and submission gates

The following checks protect the conclusions above:

- static AST checks for every seed/CuDNN/data-loader setting and executed batch
  size;
- analytical and, when PyTorch/MONAI are installed, instantiated model
  parameter/normalization checks;
- first/last-frame wrap-around tests;
- non-cubic native-mask axis and categorical-class tests;
- exact volume, EF, phase, curve, and global-motion recomputation in the
  independent cine audit;
- formula-source guards against fragile script glyphs; and
- clean LaTeX build plus full rendered-page inspection.

## Final claim boundary

The thesis supports a compact spatial 3D segmentation model and a completed,
audited, segmentation-derived full-cine analysis of 20 validation patients and
550 phases. It does not support a learned temporal Mamba, a perfectly controlled
architecture ablation, an independent test result, dense tissue tracking,
regional strain, or a precision-independent Dice/EF advantage from fixed
fusion.
