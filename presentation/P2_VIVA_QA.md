# P2 Viva Questions and Defensible Answers

Use these answers as evidence-bounded speaking notes. Give the direct answer
first, then add detail only if the examiner asks for it.

## Scientific scope

### 1. Your title says “Spatio-Temporal Framework.” Did you actually model time?

No. The completed experiment is four-class 3D cardiac MRI segmentation. ED and
ES volumes are processed independently, and the tensor depth dimension contains
anatomical slices rather than cardiac time. The registered title is retained,
while true temporal motion analysis is stated as future work.

### 2. Why is cardiac motion analysis in the title if no motion field is estimated?

Segmentation at ED and ES is a prerequisite for later phase-specific volume and
functional analysis, which motivated the project. However, this study does not
perform registration, displacement estimation, optical flow, temporal
consistency modelling, or ejection-fraction calculation. I therefore present
the implemented contribution as segmentation, not a complete motion-analysis
system.

### 3. What exactly is the prediction task?

For each independently processed 3D volume, the model predicts four voxel
classes: background, right-ventricular cavity, myocardium, and left-ventricular
cavity. The reported foreground mean Dice averages RV, MYO, and LV Dice.

### 4. Is the depth axis a temporal sequence?

No. Depth is the ordered stack of anatomical slices within one 3D volume. The
model never receives a cine sequence indexed by cardiac phase.

## Architecture and novelty

### 5. Is your bottleneck a full Mamba implementation?

No. It is an implementation-faithful, Mamba-inspired gated sequence module. It
flattens the compressed spatial grid, applies a depthwise one-dimensional
convolution, a scalar sigmoid gate, an output projection, and a residual path,
then restores the 3D grid. It does not implement Mamba's selective scan or its
hardware-aware state-space algorithm.

### 6. Why call it Mamba-inspired at all?

The inspiration is the conversion of a compact feature grid into a sequence and
the use of lightweight gated sequence processing in the U-Net bottleneck. The
term is qualified throughout the final thesis so it does not imply algorithmic
equivalence to the original Mamba architecture.

### 7. Where is the sequence module placed, and why?

It is placed at the most compressed encoder representation. Sequence processing
there operates on fewer tokens than at full resolution, which keeps the added
module compact while retaining the U-Net encoder, decoder, and skip connections.

### 8. What is the defensible technical contribution?

The contribution is an implemented compact 3D U-Net variant with a spatial
sequence gate, plus a transparent comparison of its accuracy–efficiency design
point. It is not a claim of a new selective state-space algorithm or a universal
accuracy improvement.

## Experimental design

### 9. How was data leakage controlled?

The ACDC training cohort was split at patient level with seed 42: 80 patients
for training and 20 for validation. Both ED and ES cases from one patient remain
on the same side. The audited split is disjoint, contains all 100 patients
exactly once, and corresponds to 160 training cases and 40 validation cases.

### 10. Is the reported validation set an independent test set?

No. The same 20-patient validation cohort was used to select the best checkpoint
and to produce the reported table. I therefore describe the results as held-out
validation performance, not independent test performance.

### 11. Were all models trained under a perfectly matched protocol?

They share the patient split, preprocessing, validation calculation, epoch
budget, optimizer family, and checkpoint-selection rule. However, batch sizes
differed because of memory limits, the custom models used BatchNorm, and the
ablation capacities are not parameter-matched. The table compares executed
systems but does not isolate a pure causal effect of the gate.

### 12. How is Dice computed when a class is absent in both prediction and label?

The pipeline assigns Dice equal to 1.0 when both masks are empty. This convention
is implemented consistently, but it can increase class averages for truly absent
structures and should be reported explicitly. The recovered 240 case rows contain
720 foreground class scores, none exactly equal to 1.0, so that special branch did
not affect the reported table.

### 13. Why report parameter count and FPS together?

Parameter count measures model size, while FPS measures runtime under a specific
benchmark. They are related but not interchangeable. In the recorded aggregate
results, the 3D U-Net was fastest even though Nano-Mamba was smallest, showing
that parameter count alone does not predict measured latency.

### 14. Can the FPS numbers be reproduced exactly?

Not exactly. The benchmark procedure is known—batch one, random 256×256×16
input, five warm-ups, and 30 timed runs. A current-machine capture records
PyTorch 2.7.1, MONAI 1.5.2, CUDA 11.8, cuDNN 9.1, and an RTX 4060 Laptop GPU,
but it has not been confirmed as the historical benchmark environment. I
therefore treat FPS as a recorded benchmark, not a portable property.

## Results and interpretation

### 15. Which model achieved the highest Dice?

SegResNet16 achieved the highest validation mean Dice at 86.70%. Nano-Mamba
achieved 84.78%, so I do not claim that Nano-Mamba is the most accurate model.

### 16. What is Nano-Mamba's main quantitative advantage over 3D U-Net?

Nano-Mamba improved validation mean Dice from 80.83% to 84.78%, a gain of 3.945
percentage points, while using 1.456 million instead of 4.809 million reported
parameters—a 69.714% reduction.

### 17. How does Nano-Mamba compare with SegResNet16?

Nano-Mamba is 1.918 percentage points lower in validation mean Dice, but uses
69.021% fewer reported parameters. This supports an accuracy–efficiency
trade-off claim, not an accuracy leadership claim.

### 18. Why did the No-Mamba ablation outperform Nano-Mamba?

No-Mamba achieved 85.64%, which is 0.862 percentage points higher than
Nano-Mamba. The result shows that this experiment does not demonstrate that the
gate improves Dice. Because No-Mamba also has 57.076% more parameters, a
matched-capacity, multi-seed study would be required for a stronger causal
conclusion.

### 19. What did the Half-Mamba ablation show?

Half-Mamba achieved 84.95%, 0.174 percentage points above Nano-Mamba in the
aggregate table. The recovered paired patient-bootstrap interval for Nano minus
Half-Mamba is -0.91 to +0.55 percentage points, so it crosses zero. The ordering
is not resolved by this single split, and repeated seeds are still needed.

### 20. Which anatomical class was hardest?

MYO had the lowest class Dice among the principal models in the audited table.
This is consistent with the myocardium being a thinner structure with more
boundary ambiguity than the ventricular cavities, but the experiment did not
include boundary metrics or pathology-stratified analysis to test that
explanation directly.

### 21. Are the differences statistically significant?

No significance claim is made. The recovered per-case rows support post-hoc
patient-level bootstrap intervals: Nano minus UNet is +2.83 to +5.01 percentage
points; Nano minus No-Mamba is -1.48 to -0.23; and Nano minus Half-Mamba is
-0.91 to +0.55. These are descriptive intervals on the validation patients
used for checkpoint selection, not pre-registered tests or independent-test
evidence, and no p-values are reported.

## Reproducibility and limitations

### 22. What evidence has been independently audited?

The aggregate CSV/JSON, seed-42 split, six 40-case tables, six 150-epoch logs,
and six checkpoint metadata/hash records agree. Every best log row and
checkpoint epoch/Dice matches the table. The audit also verifies that the empty
class branch did not fire and regenerates patient-level intervals and figures
from the recovered records. This is stronger than aggregate arithmetic, but it
is still narrower than complete historical run attestation.

### 23. What remains missing from the original experiment evidence?

Four historical links remain open: the currently found checkpoint directory is
not confirmed unchanged from training day; the current environment is not
confirmed as historical; the exact command is unconfirmed; and the old
discovery report lacks a complete 200-case content manifest. The strict audit
therefore remains incomplete even though the numerical and checkpoint-metadata
consistency checks now pass.

### 24. How would you strengthen the study next?

I would confirm the remaining historical provenance, run multiple seeds, use a
separate test cohort, add matched-capacity ablations, report surface-distance
and pathology-stratified results, and validate externally. Only after that
would I extend the model to full cine sequences and evaluate temporal
consistency or motion-related outputs.

### 25. What is the one-sentence conclusion you can defend?

Nano-Mamba U-Net is a compact Mamba-inspired 3D segmentation model that achieves
competitive held-out validation Dice with the smallest reported parameter count
among the evaluated models, while SegResNet16 remains the most accurate and the
current evidence does not establish a causal advantage from the gate.

## Evidence pointers

- Main executed pipeline: `src/21_rigorous_experiment_pipeline.py`
- Main result table: `evidence/rigorous_patient_split/summary_metrics.csv`
- Patient split: `evidence/rigorous_patient_split/patient_split_seed42.json`
- Automated audit: `src/22_p2_evidence_audit.py`
- Scientific boundary statement: `SCIENTIFIC_BOUNDARIES.md`
- Final thesis results: `paper_write/Universiti_Malaya_Thesis_Template/sample-chap-results-p2.tex`
