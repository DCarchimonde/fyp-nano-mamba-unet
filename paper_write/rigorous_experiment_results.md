# Rigorous Patient-Level Validation Results

These results were produced with `src/21_rigorous_experiment_pipeline.py` using a deterministic patient-level 80/20 split of the ACDC training cohort.

## Experiment setup

- Dataset root: `D:\AI_FYP\Data\ACDC\database\training`
- Valid image/label pairs found: 200
- Training patients: 80
- Validation patients: 20
- Training cases: 160
- Validation cases: 40
- Input size: 256 x 256 x 16
- Checkpoint selection: best validation mean Dice
- Final metrics: held-out validation cases only

## Main validation table

| Model | Mean DSC (%) | RV DSC (%) | MYO DSC (%) | LV DSC (%) | Params (M) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| UNet3D | 80.83 | 78.35 | 74.57 | 89.59 | 4.81 | 88.17 |
| NanoMambaUNet | 84.78 | 82.11 | 80.35 | 91.88 | 1.46 | 29.09 |
| Conv. control (historical `Ablation_NoMamba_UNet`) | 85.64 | 83.26 | 81.30 | 92.37 | 2.29 | 28.80 |
| 64-ch Mamba ablation (historical `Ablation_HalfMamba_UNet`) | 84.95 | 82.41 | 80.46 | 91.99 | 1.64 | 29.04 |
| AttentionUNet | 74.78 | 69.43 | 72.38 | 82.53 | 5.91 | 22.41 |
| SegResNet16 | 86.70 | 84.30 | 82.76 | 93.03 | 4.70 | 25.50 |

## Interpretation for thesis writing

The rigorous validation results show that SegResNet16 achieved the highest mean Dice score. NanoMambaUNet did not achieve the highest absolute Dice score, but it achieved a competitive mean Dice of 84.78% with 1.456M reported parameters. Compared with UNet3D, NanoMambaUNet improved mean Dice by 3.945 percentage points while using 69.714% fewer reported parameters. Compared with SegResNet16, NanoMambaUNet used 69.021% fewer reported parameters but had 1.918 percentage points lower mean Dice.

The historical No-Mamba control contains zero Mamba operations: its bottleneck
is two convolutional blocks. The historical Half-Mamba ablation uses the same
type of Mamba-inspired block at 64 rather than 128 channels; “Half” does not
describe half of the complete network. The control's higher Dice prevents a
claim that the present gate improves accuracy, while its 57.1% larger parameter
count and different bottleneck prevent a clean causal claim that the gate is
harmful. The 64-channel ablation's paired ordering against Nano is unresolved.
More completely, Nano uses 1.456325M reported trainable parameters versus
1.638469M for the 64-channel ablation, a reduction of 11.117%, while retaining
99.796% of its mean DSC. The paired interval crossing zero means that the
current evidence does not establish a reliable accuracy penalty for Nano; it
does not constitute a formal equivalence result.

The safest thesis claim is therefore:

> NanoMambaUNet provides a lightweight and parameter-efficient cardiac MRI segmentation model with competitive held-out validation performance, rather than the highest absolute Dice score among all compared models.

Avoid claiming that NanoMambaUNet is the best-performing model by Dice. Also avoid using the earlier training-set evaluation results as the main quantitative table.

The six per-case tables and six 150-epoch logs reproduce the aggregate values
and selected checkpoint epochs. Post-hoc patient-bootstrap intervals and paired
differences remain descriptive because the validation patients were also used
for checkpoint selection and only one split/seed was trained.

## Full-cine extension

The selected epoch-121 Nano-Mamba checkpoint was subsequently applied to all
550 phases from the same 20 validation patients. Independent frame-wise
inference was compared with fixed circular adjacent-frame softmax fusion
(`0.25 / 0.50 / 0.25`). Both paths restore categorical masks to the native
array shape before computing physical volumes from NIfTI header spacings.

| Full-cine metric | Frame-wise | Fixed fusion |
|---|---:|---:|
| Resized endpoint Dice | 84.779% | 84.794% |
| Native endpoint Dice | 77.919% | 77.923% |
| Annotated-phase EF MAE | 3.514 pp | 3.329 pp |
| EF Pearson correlation | 0.9809 | 0.9803 |
| ED exact / within one phase | 50% / 80% | 55% / 80% |
| ES exact / within one phase | 55% / 85% | 55% / 85% |
| Normalized LV-curve second difference | 0.032521 | 0.031392 |

The fixed fusion made the LV-volume curve smoother in 19/20 patients; the
paired smoothness interval was entirely below zero. Its Dice and EF-error
intervals crossed zero, so no endpoint-accuracy or EF-improvement claim is
made. The completed motion outputs are global segmentation-derived LV-centroid
and myocardial-radial trajectories, not optical flow, dense correspondence, or
strain.

The central remaining scientific gaps are independent testing, multi-seed
training, matched-capacity ablations, geometry-aware training, intermediate
manual cine labels, boundary metrics, learned temporal modeling, and dense
motion/strain validation--not reconstruction of historical shell logs.
