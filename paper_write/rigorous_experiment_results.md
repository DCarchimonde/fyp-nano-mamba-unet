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
| Ablation_NoMamba_UNet | 85.64 | 83.26 | 81.30 | 92.37 | 2.29 | 28.80 |
| Ablation_HalfMamba_UNet | 84.95 | 82.41 | 80.46 | 91.99 | 1.64 | 29.04 |
| AttentionUNet | 74.78 | 69.43 | 72.38 | 82.53 | 5.91 | 22.41 |
| SegResNet16 | 86.70 | 84.30 | 82.76 | 93.03 | 4.70 | 25.50 |

## Interpretation for thesis writing

The rigorous validation results show that SegResNet16 achieved the highest mean Dice score. NanoMambaUNet did not achieve the highest absolute Dice score, but it achieved a competitive mean Dice of 84.78% with 1.456M reported parameters. Compared with UNet3D, NanoMambaUNet improved mean Dice by 3.945 percentage points while using 69.714% fewer reported parameters. Compared with SegResNet16, NanoMambaUNet used 69.021% fewer reported parameters but had 1.918 percentage points lower mean Dice.

The safest thesis claim is therefore:

> NanoMambaUNet provides a lightweight and parameter-efficient cardiac MRI segmentation model with competitive held-out validation performance, rather than the highest absolute Dice score among all compared models.

Avoid claiming that NanoMambaUNet is the best-performing model by Dice. Also avoid using the earlier training-set evaluation results as the main quantitative table.

The six per-case tables and six 150-epoch logs reproduce the aggregate values
and selected checkpoint epochs. Post-hoc patient-bootstrap intervals and paired
differences remain descriptive because the validation patients were also used
for checkpoint selection and only one split/seed was trained. The central
scientific gaps are independent testing, multi-seed training, matched-capacity
ablations, native-space preprocessing, subgroup analysis, boundary metrics,
and true temporal modeling--not reconstruction of historical shell logs.
