# P2 Final Scientific and Defense Audit

Audit date: 23 August 2026

Scope: experiment completeness, methodological vulnerabilities, code--paper
agreement, thesis quality, and viva readiness. Historical command-forensics is
explicitly outside the P2 submission gate.

## Executive conclusion

The project is scientifically complete for the bounded claim made in the
final thesis. It now contains two connected stages:

1. a six-model, patient-level evaluation of a compact spatial 3D cardiac MRI
   segmentation model; and
2. an executed full-cine analysis of all 20 validation patients and all 550
   available phases, using independent frame-wise inference and a transparent
   fixed adjacent-frame probability-fusion baseline.

The full-cine result package was independently recomputed from its frame and
patient tables. Artifact integrity, cohort/split/checkpoint/source lineage,
endpoint identity, volume arithmetic, ejection fraction, phase timing, global
motion surrogates, paired differences, and bootstrap intervals all pass.

The supported temporal finding is narrow but positive: fixed probability
fusion produced smoother segmentation-derived LV-volume curves in 19 of 20
patients. It did **not** establish a resolved endpoint-Dice or EF improvement.
The project must not be presented as a learned temporal Mamba, optical flow,
registration, dense deformation, tissue correspondence, regional strain, an
independent-test result, or proof that the Mamba-inspired gate improves Dice.

The final thesis and defense package explain the exact NIfTI processing,
tensor shapes, spatial-versus-temporal boundary, full-cine fusion, native-grid
restoration, physical-volume calculation, phase detection, EF, global motion
surrogates, validation strategy, failure cases, and limitations. The previous
defense risk--being unable to explain temporal analysis or image processing--is
covered directly in the thesis, 14-slide deck, speaker notes, 68-question viva
bank, and bilingual method cheat sheet.

## Stage A: spatial segmentation experiment

1. Exactly two labelled 3D cases were discovered for each of 100 ACDC
   patients.
2. A seed-42 patient-level split assigned 80 patients / 160 cases to training
   and 20 patients / 40 cases to validation; ED and ES from a patient never
   crossed partitions.
3. Each NIfTI image and label was channelized and resized to
   `256 x 256 x 16`: trilinear interpolation for MRI and nearest-neighbour for
   labels.
4. MRI intensities were scaled independently to `[0,1]` per 3D volume.
5. Six configurations were trained for 150 epochs with AdamW and Dice-CE loss.
6. Each checkpoint was selected by validation foreground mean Dice.
7. Four-class logits were converted to labels by class-axis `argmax`, without
   connected-component or morphological post-processing.

### Stage-A evidence checks

| Check | Result |
|---|---|
| Patient leakage | None; train/validation patient sets are disjoint and complete |
| Case coverage | 160 train and 40 validation cases; two labelled cases per patient |
| Aggregate arithmetic | Every reported class/model mean reproduces from case rows |
| Case evidence | Six files x 40 rows; the same 20 validation patients |
| Training evidence | Six files x 150 contiguous epochs |
| Checkpoint selection | Every logged maximum matches the reported best epoch |
| Empty-empty Dice branch | Zero exact-unity values among 720 foreground scores |
| Figure inputs | Quantitative figures are generated from committed CSV/log evidence |

### Main spatial result

| Model | Mean Dice | Parameters | Defensible reading |
|---|---:|---:|---|
| SegResNet16 | 86.70% | 4.701M | Accuracy leader |
| Conv. control (historical No-Mamba) | 85.64% | 2.288M | Zero Mamba operations; higher Dice than Nano; not capacity matched |
| 64-channel Mamba ablation (historical Half-Mamba) | 84.95% | 1.638M | 0.174 pp above Nano; paired interval crosses zero |
| Nano-Mamba | 84.78% | 1.456M | 11.1% fewer parameters than Half-Mamba; no reliable accuracy penalty established |
| 3D U-Net | 80.83% | 4.809M | Nano is +3.945 pp and 69.714% smaller |
| Attention U-Net | 74.78% | 5.909M | Weakest executed configuration; not a general verdict on attention |

The correct conclusion is a lightweight accuracy--efficiency trade-off.
Nano-Mamba is not the highest-Dice model, and the unbalanced ablations do not
isolate a causal benefit from the Mamba-inspired gate. The historical
No-Mamba label denotes a purely convolutional control with zero Mamba
operations; “Half-Mamba” denotes only a 64-channel version of the gated
bottleneck, not half of the whole network.

The Half-Mamba comparison is a positive compactness result when both axes are
reported: Nano uses 11.117% fewer reported trainable parameters (1.456325M
versus 1.638469M) and retains 99.796% of the ablation's mean DSC. Its paired
interval crosses zero, so this supports a more compact operating point without
an established accuracy penalty, but not a formal claim of equivalence.

## Stage B: completed full-cine analysis

The frozen epoch-121 Nano-Mamba checkpoint was applied to every 3D phase of
the 20 validation cines: 550 phases in total. Two paths were evaluated:

- **Frame-wise:** independent softmax prediction and `argmax` for each phase.
- **Fixed temporal fusion:** circular
  `0.25 P(t-1) + 0.50 P(t) + 0.25 P(t+1)` before `argmax`.

Categorical predictions were restored to each original array shape with
nearest-neighbour interpolation. Native-grid class voxel counts were multiplied
by NIfTI header voxel volume to obtain millilitres. A circular three-point mean
was applied to the scalar LV-volume curve; ED was its global maximum and ES its
global minimum. EF was `100 x (EDV - ESV) / EDV`.

Global motion was summarized by maximum 3D LV-centroid displacement from ED
and by the ED-minus-ES change in mean myocardial radius around the LV centroid.
These are segmentation-derived global geometric surrogates, not tissue
tracking or strain.

### Stage-B evidence checks

| Check | Result |
|---|---|
| Cohort and frames | 20/20 validation patients; 550/550 cine phases |
| Output rows | 1,100 method-frame rows and 40 method-patient rows |
| Checkpoint and split | Exact checkpoint metadata and validation-patient identity reproduced |
| Endpoint identity | Annotated ED/ES images and masks map to the correct cine phases |
| Metric arithmetic | Native/resized Dice, volume, EF, phases, centroid/radius, and smoothness recomputed |
| Paired statistics | 10,000 patient-level bootstrap replicates reproduced |
| Artifact integrity | 29 tracked bundle files and 23 PNG dimensions/hashes verified |
| Independent audit | `src/25_spatiotemporal_result_audit.py`: PASS |

### Main full-cine results

| Metric | Frame-wise | Fixed fusion | Fusion minus frame-wise / interpretation |
|---|---:|---:|---|
| Resized endpoint Dice | 84.779% | 84.794% | +0.0144 pp; 95% CI crosses zero |
| Native endpoint Dice | 77.919% | 77.923% | Essentially unchanged |
| Annotated-phase EF MAE | 3.514 pp | 3.329 pp | -0.185 pp; 95% CI crosses zero |
| EF Pearson correlation | 0.9809 | 0.9803 | Strong internal agreement; not external validation |
| ED exact / within one phase | 50% / 80% | 55% / 80% | One exact ED change; within-one unchanged |
| ES exact / within one phase | 55% / 85% | 55% / 85% | Unchanged |
| LV-curve second-difference | 0.032521 | 0.031392 | -0.001129; CI entirely below zero; 19/20 smoother |
| Median peak LV-centroid displacement | 6.361 mm | 6.252 mm | Global mask-derived displacement |
| Mean MYO radial ED-minus-ES change | 4.250 mm | 4.153 mm | Global radial-extent surrogate |

The paired fusion-minus-framewise Dice interval was
`[-0.0635, +0.0819]` percentage points and the EF-error interval was
`[-0.468, +0.091]` percentage points; neither supports an accuracy or EF-gain
claim. The smoothness interval was `[-0.001950, -0.000511]`, entirely below
zero, supporting the stated smoother-curve conclusion.

Patient016 was the main phase-detection outlier (ED error five phases, ES error
three). Patient057 had the largest fused EF absolute error (8.119 percentage
points). Both are retained and discussed rather than removed.

Six recovered official ACDC full-cine files have affine metadata that differs
from their endpoint files. Shape, voxel spacing, endpoint pixel identity, and
endpoint image--label affine agreement were verified. Physical scalar metrics
use header spacings; the analysis does not use the inconsistent affine metadata
to assert world-coordinate tissue correspondence.

## Last-mile methodology and typesetting audit

An additional eleven-item review was completed against the active LaTeX,
executed model constructors, historical training entry point, sealed cine code,
and retained evidence. Its full evidence and disposition are recorded in
`paper_write/METHODOLOGY_DEEP_AUDIT_2026-08-22.md`.

| High-risk item | Final verdict |
|---|---|
| Set/loss/operator glyph corruption | Active formulas rewritten with portable bold/plain symbols and semantic operators; verified in the clean 95-page PDF |
| Attention equation scope | Scaled score is explicitly parenthesized inside softmax and multiplied by `V` afterward |
| Batch size 1 with normalization | Attention U-Net used BatchNorm3d and completed 150 epochs; SegResNet16 used GroupNorm; cross-model confounding is explicit |
| Circular fusion at first/last frame | Correct modulo indexing; both boundaries now have dedicated regression tests |
| Millilitre conversion | Exact `product(spacing_mm) / 1000` implementation and independent recomputation pass |
| Seed controls | Python, NumPy, CPU/CUDA PyTorch, both CuDNN flags, and zero-worker loaders are present; no bitwise-identity claim |
| Native-grid Dice reduction | Nearest inverse resize, original axis order, and categorical preservation confirmed; reduction is not attributed solely to Z anisotropy |
| Nano-Mamba parameters | Exactly 1,456,325 trainable parameters; 1,422 additional state-dict entries are BatchNorm buffers |
| Six affine mismatches | Shape/zoom/pixel/label checks pass; safe for reported scalar metrics, not for dense correspondence claims |

The review also found that the sealed cine run stores probability maps as
float16 before converting them back to float32 for fixed fusion. Frame-wise
argmax is preserved before that conversion. This executed precision path is now
disclosed, and the already unresolved Dice/EF changes are not described as
precision-independent improvements.

## Scientific vulnerabilities and final disposition

| Priority | Vulnerability | Impact | Final disposition |
|---|---|---|---|
| P0 closed | Registered title previously exceeded the ED/ES-only experiment | Project identity challenge | Real 20-patient full-cine function/global-motion analysis completed and audited; learned/dense-motion boundary explicit |
| P0 closed | Previous qualitative image could have used a training patient/old checkpoint | Invalid validation evidence | Replaced by traceable patient002 endpoint overlay using the selected checkpoint and committed inputs |
| P0 closed | Training curves previously lacked source logs | Unsupported learning narrative | Six 150-epoch logs recovered, validated, plotted, and discussed |
| P1 | Same validation cohort selected the checkpoint and supplies both analyses | Optimistic internal estimate | Called internal/held-out validation everywhere; no independent-test/SOTA/clinical claim |
| P1 | One split and one training seed | Split/training uncertainty unknown | Patient bootstrap is descriptive; multi-seed validation remains future work |
| P1 | Intermediate phases have no manual masks | Contour correctness between ED/ES is not directly validated | Anatomical Dice is restricted to annotated endpoints; curve/phase metrics are labelled as surrogates |
| P1 | Fusion is fixed post-hoc averaging over a spatial checkpoint | Cannot prove learned temporal reasoning | Exact formula disclosed; contribution called a transparent baseline |
| P1 | Direct resize; no orientation/spacing normalization or augmentation | Geometry/robustness limitations | Resized and native endpoint Dice both reported; physical calculations use native header spacings |
| P1 | Ablations differ in capacity and bottleneck structure | Cannot isolate the gate's causal effect | Results framed as executed-system comparison; zero-Mamba control and 64-channel ablation retained with exact definitions |
| P1 | No external cohort, HD95/ASD, calibration, or blinded visual review | Generalization/boundary quality incomplete | Explicit limitations and future work |
| P2 | Simplified block is not reference selective-scan Mamba | Terminology can be overstated | Called Mamba-inspired; kernel-3 local mixing, scalar gate, raster boundary, and lack of state recurrence/global interaction documented |
| P2 | Pathology groups are small and imbalanced (n=1--7) | Subgroup significance invalid | Reported descriptively only; no pathology-level inference |

## Defense coverage map

| Examiner question | Where it is answered |
|---|---|
| What exactly is spatial and what is temporal? | Slides 1--2, 4, 9, 12; Viva Q1--10 |
| How were images processed? | Slide 5; methodology; Viva Q11--18; method cheat sheet |
| How were all cine phases fused? | Slide 9; methodology; Viva Q19--25 |
| How were ED, ES, volume, EF, and phase error computed? | Slides 9 and 11; Viva Q26--35 |
| What does “motion” mean here? | Slides 9, 11, 12; Viva Q36--42 |
| Was the full-cine run complete and audited? | Slides 3, 10--12; Viva Q43--48 |
| Did fusion improve Dice or EF? | Slides 10--12; Viva Q49--52 |
| What are the principal failures and limits? | Slides 11--13; Viva Q53--60 |
| Why is Nano-Mamba still a contribution? | Slides 6--8 and 14; spatial-results section |

## Delivery verification

| Deliverable/check | Final status |
|---|---|
| Thesis | 95-page A4 PDF; clean four-pass build; all pages rendered and reviewed |
| Presentation | 14 slides; all rendered/reviewed; no overflow; template-fidelity pass |
| Speaker notes | Present on all 14 slides; each contains a source block |
| Viva preparation | 68 questions plus bilingual method cheat sheet |
| Automated tests | 55 discovered; 48 passed and seven PyTorch/MONAI-dependent tests skipped explicitly in the audit container |
| Candidate PyTorch environment | Earlier architecture and evidence tests passed; final full suite should also be run after pull |
| Spatial evidence audit | Scientific consistency PASS |
| Full-cine independent audit | PASS |

## Submission decision

Scientifically and defensively ready for P2 submission within the explicit
scope above. Before portal upload, the candidate must create the ignored
private-field file on the private machine, fill institution-required identity
and signature fields, rebuild once, and inspect the declaration page.

Historical command/environment attestation is optional archival forensics and
is not a thesis, presentation, viva, or portal-submission requirement.
