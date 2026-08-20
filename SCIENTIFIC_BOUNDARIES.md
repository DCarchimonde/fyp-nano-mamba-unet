# Scientific Claim Boundaries

These boundaries apply to the thesis, figures, repository, presentation, and
viva answers.

## Supported claims

- The implemented task is four-class 3D cardiac MRI segmentation: background,
  right ventricle, myocardium, and left ventricle.
- Labelled ED and ES volumes are processed as separate samples.
- Tensor depth is anatomical slice depth. A fixed raster order converts a 3D
  latent grid to a one-dimensional spatial token sequence.
- The bottleneck is a lightweight Mamba-inspired gated sequence module. It is
  not the original Mamba selective scan or a full state-space implementation.
- Main values come only from the rigorous patient-split pipeline and audited
  summary CSV.
- SegResNet16 achieved the highest validation mean Dice (86.70%).
- Nano-Mamba U-Net achieved 84.78% with the smallest reported parameter count
  (1.456M) among the evaluated configurations.
- Nano-Mamba U-Net exceeded UNet3D by 3.945 percentage points and used 69.714%
  fewer reported parameters.
- The No-Mamba ablation achieved 85.64%, 0.862 percentage points above
  Nano-Mamba, with 57.076% more reported parameters.
- Recovered per-case rows support a post-hoc paired patient-bootstrap interval
  of [+2.829, +5.009] percentage points for Nano-Mamba minus UNet3D.
- The corresponding interval crosses zero for Nano-Mamba versus Half-Mamba,
  while it lies below zero versus No-Mamba and SegResNet16.
- Six recovered 150-epoch logs and six checkpoint metadata/hash records agree
  with every selected epoch and mean Dice in the summary table.

## Claims that must not be made

- No temporal tracking, temporal registration, displacement, optical flow,
  ejection fraction, or true 4D cine sequence was implemented or evaluated.
- Depth must never be described as cardiac time.
- Nano-Mamba U-Net is not the best model by Dice in this experiment.
- The ablation does not causally prove that Mamba-inspired gating improves or
  harms accuracy because capacity, normalization, and batch-size confounds
  remain.
- The 20 validation patients are not the official ACDC test set and not an
  independent final test set; the same set was used for checkpoint selection.
- A single seed/split and post-hoc bootstrap intervals do not establish
  statistical significance, training robustness, pathology-specific
  performance, or external generalization. No p-values are claimed.
- Reported FPS is not a universal architecture property; it is an
  environment-specific batch-one random-input benchmark. A current environment
  capture exists, but its identity with the historical benchmark environment
  is unconfirmed.
- The legacy gate projection contains trainable channels without an output path;
  published parameter counts include those channels.

## Fixed-title viva defence

The registered title remains unchanged. In the completed experiment,
“spatio-temporal” describes the wider cardiac-motion motivation and intended
research trajectory, not an achieved temporal model. The evaluated network is
spatial 3D segmentation with independently processed ED/ES volumes. True 4D
temporal modelling is stated as future work.
