# Scientific Claim Boundaries

These boundaries apply to the thesis, figures, repository, presentation, and
viva answers.

## Supported claims

- The implemented task is four-class 3D cardiac MRI segmentation: background,
  right ventricle, myocardium, and left ventricle.
- Labelled ED and ES volumes are processed as separate samples.
- Tensor depth is anatomical slice depth. A fixed raster order converts a 3D
  latent grid to a one-dimensional spatial token sequence.
- Each input follows the same deterministic path: NIfTI load, channel-first,
  image/label resize to `256 x 256 x 16`, per-volume image min--max scaling,
  and tensor conversion. Image interpolation is trilinear; label interpolation
  is nearest-neighbour.
- The Nano-Mamba bottleneck receives `B x 128 x 32 x 32 x 2`, flattens the
  spatial grid to 2,048 tokens, and returns it to the decoder.
- The bottleneck is a lightweight Mamba-inspired gated sequence module. Its
  kernel-3 depthwise convolution mixes only neighbouring positions in the
  fixed raster sequence, followed by a token-wise scalar gate. It has no
  recurrent state propagation, selective scan, self-attention, or direct
  all-token interaction and is not a full state-space implementation.
- Main values come only from the rigorous patient-split pipeline and audited
  summary CSV.
- SegResNet16 achieved the highest validation mean Dice (86.70%).
- Nano-Mamba U-Net achieved 84.78% with exactly 1,456,325 trainable parameters,
  the smallest count among the evaluated configurations. Its checkpoint has
  1,422 additional BatchNorm running-statistic/counter entries that are buffers,
  not trainable parameters.
- Nano-Mamba U-Net exceeded UNet3D by 3.945 percentage points and used 69.714%
  fewer reported parameters.
- The zero-Mamba convolutional-bottleneck control (historical experiment name:
  `Ablation_NoMamba_UNet`) contains no Mamba or Mamba-inspired operation. It
  achieved 85.64%, 0.862 percentage points above Nano-Mamba, with 57.076% more
  reported parameters.
- The 64-channel Mamba-bottleneck ablation (historical experiment name:
  `Ablation_HalfMamba_UNet`) uses the same type of Mamba-inspired block at 64
  channels. “Half” describes that block width relative to Nano's 128 channels,
  not half of the network. Nano uses 1.456325M reported trainable parameters
  versus 1.638469M for this ablation, an 11.117% reduction, while retaining
  99.796% of its mean Dice. The paired interval crosses zero, so the evidence
  does not establish a reliable accuracy penalty for Nano or formal equivalence.
- Audited per-case rows support a post-hoc paired patient-bootstrap interval
  of [+2.829, +5.009] percentage points for Nano-Mamba minus UNet3D.
- The corresponding interval crosses zero for Nano-Mamba versus the 64-channel
  ablation, while it lies below zero versus the zero-Mamba convolutional
  control and SegResNet16.
- Six available 150-epoch logs and six checkpoint metadata/hash records agree
  with every selected epoch and mean Dice in the summary table.
- A separate, completed full-cine run processed every one of the 550 phases from
  the same 20 validation patients using the selected epoch-121 Nano-Mamba
  checkpoint. This run did not retrain or alter the historical six-model result.
- Complete-cine endpoint images match the historical standalone ED/ES inputs
  after the executed normalization (maximum normalized MAE 0), and reproduced
  endpoint Dice differs from the historical case table by at most 0.0001692.
- The cine workflow restores predicted masks to the original array shape and
  derives LV/RV/MYO volume curves, ED/ES timing, EDV, ESV, stroke volume, EF,
  LV-centroid displacement, and a global MYO radial-distance surrogate.
- Fixed circular probability fusion uses
  `0.25 P(t-1) + 0.50 P(t) + 0.25 P(t+1)`; it is deterministic and contains no
  learned temporal parameter.
- The sealed cine run computed logits and softmax in float32 and preserved the
  frame-wise argmax before storing probability maps as float16. Neighbour maps
  were converted back to float32 for fixed fusion. Reported fusion measurements
  describe this executed precision path.
- Frame-wise and temporal-fusion resized endpoint Dice are 84.779% and 84.794%.
  Their paired difference is +0.014 percentage points with a 95% bootstrap
  interval of [-0.063, +0.082] percentage points, which includes zero.
- Frame-wise and temporal-fusion annotated-phase EF MAE are 3.514 and 3.329
  percentage points. The paired change is -0.185 points with interval
  [-0.468, +0.091], which includes zero.
- Temporal fusion reduced normalized LV-curve second difference by 0.001129;
  the paired interval [-0.001950, -0.000511] excludes zero, and 19/20 patient
  trajectories were smoother.
- The full-cine raw artifacts, all 1,100 frame rows, all 40 patient rows, 23
  figures, source/checkpoint hashes, and aggregate arithmetic pass a separate
  independent audit implementation.

## Claims that must not be made

- Depth must never be described as cardiac time.
- The trained Nano-Mamba checkpoint must not be described as a learned temporal
  Mamba model. It processes each 3D cardiac phase independently; only the
  separate fixed probability fusion uses neighbouring phases.
- The project must not be described as optical flow, deformable registration,
  dense displacement, voxel correspondence, local myocardial strain, or a
  clinically validated motion-tracking system.
- Historical six-model Dice is measured on the resized grid. The full-cine
  workflow does perform nearest-neighbour native-shape mask restoration and
  reports the lower native-grid endpoint Dice; it does not retroactively make
  the training comparison spacing-preserving or orientation-canonicalized.
- The approximately 6.861 percentage-point resized/native frame-wise Dice
  difference must not be attributed solely to Z-axis anisotropy. Code and
  regression review found no axis permutation or interpolation-mode bug;
  in-plane/through-plane resizing, native-grid heterogeneity, boundary
  discretization, and model error can all contribute.
- Intermediate cine frames do not have manual masks. Smooth trajectories alone
  must not be presented as proof of anatomically correct intermediate masks.
- Diagnostic-group values are descriptive only because group sizes are 1--7;
  no pathology-specific significance or generalization is established.
- Nano-Mamba U-Net is not the best model by Dice in this experiment.
- The ablation does not causally prove that Mamba-inspired gating improves or
  harms accuracy because capacity, normalization, and batch-size confounds
  remain.
- Attention U-Net used BatchNorm3d at batch size one and SegResNet16 used
  GroupNorm at batch size one. Completion of all 150 epochs rules out a training
  crash, but does not remove the cross-model normalization/batch-size confound.
- The 20 validation patients are not the official ACDC test set and not an
  independent final test set; the same set was used for checkpoint selection.
- A single seed/split and post-hoc bootstrap intervals do not establish
  statistical significance, training robustness, pathology-specific
  performance, or external generalization. No p-values are claimed.
- Reported FPS is not a universal architecture property; it is an
  environment-specific batch-one random-input benchmark.
- Peak VRAM was not retained as a principal comparison metric because the
  historical configurations used different batch sizes and model-specific
  normalization. No claim of peak-memory superiority is made. In this thesis,
  “lightweight” is operationally supported by the lowest trainable-parameter
  count together with competitive recorded batch-one throughput, not by
  leadership in every resource metric.
- The legacy gate projection contains trainable channels without an output path;
  published parameter counts include those channels.
- The very small fixed-fusion Dice/EF differences must not be presented as
  precision-independent improvements because the sealed probability maps used
  float16 host-memory storage before float32 fusion arithmetic.

## Fixed-title viva defence

The registered title remains unchanged. The completed framework has two stages:
the trained network is spatial 3D segmentation with independently processed
phases, while a separate post-hoc workflow applies that checkpoint to every
frame of each validation cine and derives complete-cycle global function and
motion trajectories. Thus, a full-cine segmentation-derived spatio-temporal
analysis was achieved. What remains future work is learned temporal modeling,
dense tissue correspondence, regional strain, and independent clinical
validation.
