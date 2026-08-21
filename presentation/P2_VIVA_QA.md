# P2 Viva Questions and Defensible Answers

Give the direct answer first. Add the second paragraph only when the examiner
asks for detail. Every answer below is aligned with the executed pipeline and
the final thesis.

## Scope and the title

### 1. Your title says “Spatio-Temporal Framework.” Did you model time?

No. The completed experiment is four-class 3D cardiac MRI segmentation. ED and
ES volumes are processed independently, and tensor depth contains anatomical
slices rather than cardiac time. True temporal motion analysis is future work.

### 2. Why is cardiac motion analysis in the title?

ED and ES segmentation is a prerequisite for later phase-specific volumes,
ejection fraction, and motion analysis, which motivated the registered title.
This experiment does not estimate displacement, optical flow, temporal
consistency, registration, ejection fraction, or any full-cine motion output.

### 3. What exactly is one sample and one prediction?

One sample is one labelled 3D ED or ES volume from one patient. The model
outputs one four-class voxel map: background, right-ventricular cavity,
myocardium, and left-ventricular cavity.

### 4. Did you compare ED against ES performance?

No phase-stratified result is reported. Each validation patient contributes one
ED and one ES case, but the final table pools all 40 cases. Therefore I cannot
claim that the model is better at one phase.

## Image processing: exact executed path

### 5. How did you process each MRI image?

The exact order was: load the paired NIfTI image and label; add a channel axis;
resize both to 256 by 256 by 16; scale the image intensity to 0--1; and convert
both arrays to PyTorch tensors. This is implemented by `LoadImaged`,
`EnsureChannelFirstd`, `Resized`, `ScaleIntensityd`, and `ToTensord`.

### 6. Why are image and label resized differently?

The MRI image is continuous, so trilinear interpolation estimates intermediate
intensities smoothly. The label is categorical, so nearest-neighbour
interpolation is used to avoid creating invalid fractional class labels.

### 7. What does intensity scaling do?

For each loaded single-channel volume, the minimum intensity is mapped to 0 and
the maximum to 1. It is volume-wise min--max scaling. It is not z-score,
histogram matching, dataset-wide normalization, or scanner harmonization.

### 8. Did you use data augmentation?

No. There was no random crop, rotation, flip, elastic deformation, or intensity
augmentation. The train and validation datasets used the same deterministic
transform; only training case order was shuffled. This limits robustness and is
reported as a limitation.

### 9. Did you standardize orientation or voxel spacing?

No. The pipeline has no `Orientationd` or `Spacingd`. It resizes array dimensions
directly to a common shape, so it is a fixed-grid comparison rather than a
native physical-space evaluation. This may distort geometry.

### 10. Did you crop the heart or restore predictions to native resolution?

No. There is no heart crop and no inverse resampling. Prediction and reference
labels are compared on the common 256 by 256 by 16 resized grid.

### 11. What is the model input tensor shape?

After batching, it is `B x 1 x 256 x 256 x 16`. The one is the MRI channel. The
last dimension is the stack of anatomical slices, not time.

### 12. How is the final mask produced?

The network outputs four logits per voxel with shape
`B x 4 x 256 x 256 x 16`. `argmax` selects the class with the largest logit.
There is no threshold, connected-component filtering, hole filling, or other
mask post-processing.

## Architecture and novelty

### 13. Where does the sequence module operate?

After three pooling operations, Nano-Mamba has a feature tensor of
`B x 128 x 32 x 32 x 2`. The spatial grid is flattened to 2,048 tokens with
128 channels, processed, and reshaped back before decoding.

### 14. Is the bottleneck a full Mamba implementation?

No. It is Mamba-inspired: an input projection, depthwise one-dimensional
convolution, scalar sigmoid gate, output projection, and residual connection.
It does not implement selective scan or the original hardware-aware state-space
algorithm.

### 15. Why use the term Mamba-inspired?

The inspiration is lightweight gated processing of a serialized feature
sequence at the bottleneck. The final thesis qualifies the term everywhere and
does not claim algorithmic equivalence to Mamba.

### 16. Why place it at the bottleneck?

The bottleneck has only 2,048 tokens rather than 65,536 full-resolution voxels,
so sequence processing is cheaper. The trade-off is that fine boundary detail
may already be compressed; skip connections help but cannot guarantee recovery.

### 17. What is the defensible novelty?

The contribution is the implementation and evaluation of a compact spatial
sequence gate inside a 3D U-Net, together with an accuracy--efficiency
comparison. It is not a new state-space algorithm and not a true temporal
framework.

## Training and evaluation

### 18. How was patient leakage prevented?

The split was created at patient level with seed 42: 80 patients for training
and 20 for validation. Both labelled phases from one patient stay on the same
side, giving 160 training cases and 40 validation cases with no patient overlap.

### 19. Is the reported validation set an independent test set?

No. The same 20-patient validation cohort was checked every epoch to select the
best checkpoint and then used for the final descriptive table. I report
held-out validation performance, not independent or official ACDC test
performance.

### 20. What loss and optimizer were used?

The loss was MONAI Dice plus cross entropy with one-hot labels and softmax. The
optimizer was AdamW with learning rate `1e-4` and weight decay `1e-5` for 150
epochs. There was no scheduler or early stopping.

### 21. Does training Dice use the same classes as reported Dice?

Not exactly. The default Dice term in the executed Dice-CE loss includes the
background channel. The reported metric excludes background and averages RV,
MYO, and LV. The objectives are related, but their class averaging is not
identical.

### 22. How was reported Dice computed?

For every case, `argmax` predictions were compared with integer labels for RV,
MYO, and LV. Class Dice was averaged over the 40 cases, then the three class
means were averaged. Because every patient has two cases, the aggregate gives
equal case weight and equal total weight per patient.

### 23. What happens when a class is absent in prediction and reference?

The executed metric returns Dice 1 for that class. None of the 720 recorded
foreground class scores equals exactly 1, so that special branch did not affect
the final table.

### 24. Were all models trained under a perfectly controlled protocol?

No. They shared the split, preprocessing, epoch budget, AdamW settings,
checkpoint rule, and metric. However, batch sizes were 1 or 2, optimizer steps
per epoch therefore differed, model families used different normalization and
regularization, and the ablations were not parameter matched. This is a
comparison of executed systems, not a pure causal test of one module.

### 25. Did the models overfit?

Training loss continued to fall while validation performance peaked earlier.
The best-to-final validation drop was small for Nano-Mamba at 0.29 percentage
points, but 3.05 points for Attention U-Net. This is why the best validation
checkpoint was used, while also showing that the validation cohort is not an
independent final test.

## Results and failure analysis

### 26. Which model achieved the highest Dice?

SegResNet16 achieved the highest validation mean Dice at 86.70%. Nano-Mamba
achieved 84.78%, so it is not the most accurate model in this experiment.

### 27. What is Nano-Mamba's main advantage over 3D U-Net?

Nano-Mamba improved mean Dice by 3.945 percentage points and used 69.714% fewer
reported parameters: 1.456 million versus 4.809 million.

### 28. Why did No-Mamba outperform Nano-Mamba?

The experiment cannot identify one cause. No-Mamba scored 0.862 percentage
points higher but had 57.076% more parameters and a structurally different
convolutional bottleneck. A capacity-matched, multi-seed ablation would be
needed for a causal conclusion.

### 29. Which class was hardest?

MYO had the lowest aggregate class Dice among the stronger models. A plausible
reason is its thin boundary-sensitive structure, but the study did not report
surface distances or a blinded error review, so this explanation remains an
interpretation rather than a tested mechanism.

### 30. What was Nano-Mamba's worst validation case?

`patient049_frame11` had 57.86% mean Dice, including only 26.04% RV Dice. It was
also the worst case for four other model families, suggesting a shared difficult
case rather than a Nano-only failure. I do not assign a pathology or geometric
cause without inspecting that patient's metadata and image overlay.

### 31. Are the reported differences statistically significant?

No significance claim is made. Post-hoc paired patient-bootstrap intervals are
descriptive and use the same patients involved in checkpoint selection. They
do not replace multiple training seeds or an independent test cohort.

### 32. Why is the smallest model not the fastest?

Parameter count and runtime measure different things. Kernel efficiency,
memory movement, framework implementation, and operator launch overhead also
matter. In this benchmark, 3D U-Net was fastest; Nano-Mamba was smallest.

## Limitations and conclusion

### 33. What are the most important experimental limitations?

There is one patient split and seed, no independent test set, no augmentation,
no orientation or spacing standardization, no native-space evaluation, no
phase- or pathology-stratified analysis, no matched-capacity ablation, and no
surface-distance metric. The model also does not perform temporal analysis.

### 34. What would you improve first?

First, use train/validation/test separation or nested cross-validation and run
multiple seeds. Then add spacing- and orientation-aware preprocessing,
augmentation, matched-capacity ablations, surface metrics, pathology and ED/ES
subgroup analysis, and validation-case qualitative overlays. True 4D temporal
modeling is a separate architectural extension.

### 35. What is the one-sentence conclusion?

Nano-Mamba U-Net is a compact Mamba-inspired 3D segmentation model that
achieves competitive held-out validation Dice with the smallest reported
parameter count among the evaluated configurations, while SegResNet16 remains
the accuracy leader and the current experiment does not prove a causal benefit
from the sequence gate.

## Evidence pointers

- Executed pipeline: `src/21_rigorous_experiment_pipeline.py`
- Bottleneck implementation: `src/nano_mamba_core.py`
- Result table: `evidence/rigorous_patient_split/summary_metrics.csv`
- Patient split: `evidence/rigorous_patient_split/patient_split_seed42.json`
- Case metrics and curves: `evidence/rigorous_patient_split/per_case_*.csv` and
  `training_log_*.csv`
- Thesis methodology and results: `sample-chap-methodology.tex` and
  `sample-chap-results-p2.tex`
