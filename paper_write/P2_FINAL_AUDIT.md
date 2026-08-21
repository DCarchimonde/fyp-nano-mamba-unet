# P2 Final Scientific and Defense Audit

Audit date: 21 August 2026

Scope: experiment completeness, methodological vulnerabilities, code--paper
agreement, thesis quality, and viva readiness. Full historical forensics is
explicitly outside the P2 submission gate.

## Executive conclusion

The six-model result table is internally consistent and is supported by a
patient-level split, 40 validation-case rows per model, and 150 epoch rows per
model. The thesis can defend an internal accuracy--efficiency comparison.

The project must not be presented as true spatio-temporal modeling, a full
Mamba implementation, an independent-test result, or proof that the sequence
gate improves accuracy. Those are claim boundaries, not missing historical-log
tasks.

The final thesis and defense package now explain the exact NIfTI preprocessing,
tensor shapes, model output, loss/metric boundary, absence of augmentation and
post-processing, learning dynamics, worst cases, and study limitations. The
previous defense risk--being unable to explain temporal scope or image
processing--is covered directly in the slides, speaker notes, viva bank, and
bilingual method cheat sheet.

## What the experiment actually did

1. Discovered exactly two labelled 3D cases for each of 100 ACDC patients.
2. Split at patient level with seed 42: 80 train patients / 20 validation
   patients, equivalent to 160 / 40 cases.
3. Loaded paired NIfTI image and label volumes.
4. Added a channel dimension.
5. Resized both to `256 x 256 x 16`: trilinear for image, nearest-neighbour for
   label.
6. Scaled each image volume to `[0,1]` and converted arrays to tensors.
7. Trained six configurations for 150 epochs with AdamW and Dice-CE loss.
8. Selected each best checkpoint by validation foreground mean Dice.
9. Produced four output logits per voxel and used `argmax` without mask
   post-processing.
10. Reported RV, MYO, LV, foreground mean Dice, parameters, and batch-one FPS.

## Evidence checks completed

| Check | Result |
|---|---|
| Patient leakage | None; train/validation patient sets are disjoint and complete |
| Case coverage | 160 train and 40 validation cases; two cases per patient |
| Models | Six expected configurations present |
| Aggregate arithmetic | Every mean equals the underlying class/case values |
| Case evidence | Six files x 40 rows; same 20 validation patients |
| Training evidence | Six files x 150 contiguous epochs |
| Checkpoint selection | Every logged maximum matches the reported best epoch |
| Empty-empty Dice branch | Zero exact-unity values among 720 foreground scores |
| Figure inputs | Quantitative figures are generated from result CSV/log files |
| Code tests | 20 tests pass when PyTorch is installed; three skip cleanly without it |

## Main result interpretation

| Model | Mean Dice | Parameters | Defensible reading |
|---|---:|---:|---|
| SegResNet16 | 86.70% | 4.701M | Accuracy leader |
| No-Mamba | 85.64% | 2.288M | Higher Dice than Nano; not capacity matched |
| Half-Mamba | 84.95% | 1.638M | Close to Nano on this split |
| Nano-Mamba | 84.78% | 1.456M | Smallest; competitive trade-off |
| 3D U-Net | 80.83% | 4.809M | Nano is +3.945 pp and 69.714% smaller |
| Attention U-Net | 74.78% | 5.909M | Weakest executed configuration; not a general verdict on attention |

The correct conclusion is an accuracy--efficiency trade-off. SegResNet16 is
most accurate, No-Mamba is also higher by Dice, and Nano-Mamba has the smallest
reported parameter count.

## Learning and failure analysis

- Nano-Mamba peaks at epoch 121; its epoch-150 validation score is 0.29
  percentage points lower.
- Attention U-Net peaks at epoch 62 and finishes 3.05 points lower, the largest
  late-epoch decline.
- Training loss continues to decrease for all models while validation peaks
  earlier, supporting validation-based checkpoint selection and indicating
  some late-epoch overfitting.
- `patient049_frame11` is the worst case for Nano-Mamba and four other model
  families. Nano-Mamba records 57.86% mean Dice and 26.04% RV Dice on that
  case. The shared failure should not be assigned a pathology or imaging cause
  without inspecting the image and metadata.

## Scientific vulnerabilities and disposition

| Priority | Vulnerability | Impact | Final disposition |
|---|---|---|---|
| P0 | Registered title suggests temporal analysis, but code processes ED/ES independently | Examiner may challenge project identity | Explicitly disclosed in abstract, scope, discussion, slides 2/12, and Q1--4; never claim temporal modeling |
| P0 | Same validation set selects checkpoints and supplies final table | Optimistic internal estimate; not independent test | Reported as held-out validation only; no test/SOTA/clinical claim |
| P1 | One split and one seed | Training and split uncertainty unknown | Post-hoc patient bootstrap is descriptive only; multi-seed work stated as future |
| P1 | No augmentation | Robustness to plausible transformations untested | Exact omission stated; no robustness claim |
| P1 | Direct array resize without orientation/spacing standardization or inverse mapping | Geometry may be distorted; metrics are not native-space | Exact pipeline and resized-grid Dice stated in methodology, slide 5, and Q5--12 |
| P1 | Batch size, normalization, optimizer-update count, and model capacity differ | Comparison cannot isolate one architectural cause | Results framed as executed-system comparison; no causal gate claim |
| P1 | No-Mamba is not parameter matched and scores higher | Gate benefit is unproven | Treated as an important negative result |
| P1 | Training Dice includes background; reported Dice does not | Optimization and reporting averages differ | Explicitly documented in methodology and Q21 |
| P1 | No ED/ES or pathology subgroup table | Phase/pathology behavior unknown | Explicit limitation; no subgroup claim |
| P1 | Dice only; no HD95/ASD or formal qualitative review | Boundary quality and visual failure mode incompletely measured | Explicit limitation and future work |
| P2 | Simplified block is not selective scan; legacy projection outputs are unused | “Mamba” terminology can be overstated | Called Mamba-inspired everywhere; implementation explained exactly |
| P2 | FPS is environment/operator specific | Smallest model is not necessarily fastest | Treated as a batch-one engineering benchmark only |

None of the P1/P2 items above can be truthfully “fixed” by rewriting prose; most
would require retraining, new evaluation, or additional clinical metadata. They
are now bounded so the thesis does not claim evidence it does not have.

## Previous 31-item paper review

The earlier review document was checked against the current thesis. The prior
problems are closed as follows:

- Unsupported WHO/annotation-time/SOTA/economic claims were removed or replaced
  with cited, bounded statements.
- Missing U-Net, Attention U-Net, ResNet, SegResNet, ACDC, and software
  citations were repaired.
- Template-like and hyperbolic wording was rewritten.
- Research objectives and questions now map directly to Chapters 5--8.
- Duplicate SSM equations and duplicate V-Net bibliography entries were
  removed.
- Obsolete training-set tables and contradictory ablation values were replaced
  by the patient-level validation table.
- “Absolute optimum”, “definitively proven”, and similar overclaims were
  removed.
- The class comparison figure is now a grouped bar chart generated from the
  result CSV.

## Defense coverage map

| Examiner question | Where it is answered |
|---|---|
| Did you really perform temporal analysis? | Slides 2 and 12; thesis Chapters 1 and 7; Viva Q1--4 |
| How were the images processed? | Slide 5; thesis Chapter 5; Viva Q5--12; method cheat sheet |
| What are the tensor and sequence shapes? | Slide 4; thesis Chapter 5; Viva Q13--17 |
| How was leakage prevented? | Slide 3; thesis Chapter 5; Viva Q18--19 |
| How were loss, checkpoint, and Dice defined? | Thesis Chapter 5; Viva Q20--25 |
| Why is Nano not the best? | Slides 6--8 and 13; Viva Q26--28 |
| Where does the model fail? | Thesis Chapter 6; Viva Q29--30 |
| What is missing scientifically? | Slide 11; thesis Chapter 7; Viva Q31--34 |

## Submission decision

Scientifically suitable for P2 submission with the stated scope. The remaining
work before portal upload is administrative: fill private declaration fields on
the private machine, rebuild once, inspect the declaration page, and upload the
PDF/PPTX required by the faculty.

`src/22_p2_evidence_audit.py --strict-closure` is an optional archival
attestation mode. Its status is not a thesis, presentation, viva, or portal
submission blocker.
