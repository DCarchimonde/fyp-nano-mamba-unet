# P2 Viva Questions and Defensible Answers

Give the direct answer first. Add the detail only when the examiner follows up. These answers match the executed code, audited evidence, final thesis, and presentation.

## Scope, title, and completion

### 1. Your title says “Spatio-Temporal Framework.” Did you actually complete a spatio-temporal analysis?

Yes, within a precise segmentation-derived scope. I processed every cardiac phase for all 20 validation patients—550 3D frames—and derived time-indexed masks, chamber-volume curves, ED/ES timing, EF, LV-centroid displacement, and myocardial radial change. The result bundle passed an independent numerical and provenance audit.

The trained Nano-Mamba backbone itself is spatial, not temporal. Temporal context enters afterward through a fixed adjacent-frame probability fusion. I do not claim learned temporal Mamba, optical flow, dense tissue tracking, or strain.

### 2. What are the two stages of the framework?

Stage A is supervised spatial 3D segmentation using independently labelled ED and ES volumes. Stage B freezes the selected checkpoint, applies it to every phase of each 4D cine, optionally fuses adjacent softmax probabilities, restores masks to native array shape, and derives temporal function and global motion summaries.

Keeping the stages separate is essential: the second stage analyses time, but the first-stage network was not trained across time.

### 3. Is tensor depth a time dimension?

No. In the model input `B x 1 x 256 x 256 x 16`, the last dimension contains anatomical slices. ED and ES were independent training cases. Cardiac time is the fourth axis of the original complete cine and is iterated phase by phase only in the full-cine analysis.

### 4. Why is “cardiac motion analysis” still a defensible phrase?

Because the completed pipeline produces full-cycle global anatomical trajectories, including LV-centroid displacement and myocardial radial change, as well as functional indices. These are segmentation-derived motion surrogates across time.

I qualify them as global, not dense. There is no voxel correspondence, material trajectory, deformation field, optical flow, registration, or regional strain.

### 5. What exactly is completed, and what remains future work?

Completed: a compact spatial segmenter; 20-patient, 550-frame full-cine inference; fixed temporal probability fusion; native-grid masks; volume, phase, EF, centroid, radius, smoothness, endpoint validation, figures, and independent recomputation.

Future: learned temporal representation, independent external testing, multiple training seeds, matched ablations, dense registration/correspondence, regional strain, and manual validation of intermediate phases.

### 6. What is the one-sentence contribution?

The project delivers a compact 1.456-million-parameter spatial 3D segmentation model with 84.78% held-out validation Dice, plus an audited full-cine analysis whose clearest temporal result is smoother global LV-volume curves in 19 of 20 patients.

### 7. Did Nano-Mamba learn cardiac dynamics?

No. Its bottleneck mixes spatial tokens inside one 3D volume. The cardiac-cycle dependency is a fixed post-hoc operation over neighbouring phase probabilities, with no learned temporal parameter.

### 8. Can this be called 4D segmentation?

Operationally, the output is a segmentation sequence over the complete 4D cine. Architecturally, it is repeated 3D spatial inference plus fixed temporal regularization, not a jointly trained 4D neural network. I state both facts to avoid ambiguity.

## Dataset and exact image processing

### 9. Which data were used for the full-cine analysis?

The same 20 patients from the fixed seed-42 validation split of the ACDC training cohort. Their complete cines contain 550 phases in total. This is internal validation, not the hidden ACDC test set.

### 10. How did you find the correct 4D cine file?

The code recursively inspects candidate NIfTI files and requires a four-dimensional shape `(X,Y,Z,T)` with `T` equal to `NbFrame` in that patient's `Info.cfg`. It does not infer identity from file name or file size alone.

### 11. How is one 4D cine processed?

The `(X,Y,Z,T)` array is traversed at each phase `t`. Each `(X,Y,Z)` phase is handled as one 3D input: add channel and batch axes, resize to `256 x 256 x 16`, min--max normalize, run the frozen checkpoint, and retain the four-class softmax probability map.

### 12. Why resize every phase to `256 x 256 x 16`?

That is the geometry used to train the checkpoint, so inference must be checkpoint-compatible. It provides a fixed tensor size under the available GPU memory. The limitation is that array-size resizing is not equivalent to common physical-spacing resampling and can distort geometry.

### 13. Which interpolation methods were used?

Continuous MRI intensities use trilinear interpolation. Categorical labels and predicted masks use nearest-neighbour interpolation so class identifiers remain integers and are not blended.

### 14. How was intensity normalization performed?

Each 3D phase was independently min--max scaled to `[0,1]` after resizing, using its finite minimum and maximum. It is not z-score normalization, histogram matching, dataset-wide standardization, or scanner harmonization.

### 15. Did you use augmentation, cropping, or mask cleanup?

No. There was no random augmentation, heart crop, connected-component filtering, hole filling, threshold tuning, or morphological cleanup. This keeps the execution transparent but limits robustness.

### 16. Did you standardize orientation or physical spacing?

The historical training pipeline did not use explicit `Orientationd` or `Spacingd`; it resized array dimensions directly. The full-cine stage restores the categorical output to the native array shape and uses NIfTI header voxel sizes for physical volume and distance calculations. This does not retroactively make the training geometry native-space.

### 17. Why report both resized-grid and native-grid Dice?

Resized-grid Dice reproduces the historical experiment and tests the model on its computational grid. Native-grid Dice evaluates the restored categorical mask against the original-shape endpoint mask. The latter is lower—about 77.92% versus 84.79%—which exposes the cost of the non-native geometry rather than hiding it.

### 18. How is the final mask created?

For frame-wise inference, I apply `argmax` across the four probabilities at each voxel. For temporal fusion, I first combine adjacent probability maps, then apply the same `argmax`. The resulting categorical mask is restored to native shape with nearest-neighbour interpolation.

## Temporal probability fusion

### 19. What is the temporal fusion equation?

`P_t_fused = 0.25 P_(t-1) + 0.50 P_t + 0.25 P_(t+1)`.

The weights are fixed, sum to one, and give the current phase twice the weight of either neighbour. There is no trained temporal parameter.

### 20. Why fuse probabilities instead of hard masks?

Softmax probabilities retain class uncertainty and allow a weighted continuous combination before the irreversible `argmax`. Hard-mask voting would discard that information and make the current-frame weighting less direct.

### 21. What happens at the first and last phase?

Indices wrap circularly: the previous phase of frame 1 is the final frame, and the next phase of the final frame is frame 1. This models the cine as a periodic cardiac cycle and avoids an artificial boundary.

### 22. Why use these exact weights?

They form a simple, pre-specified symmetric low-pass baseline: half current evidence and half divided equally between immediate neighbours. I did not tune them on the reported outcomes, and I do not claim they are optimal.

### 23. Is fixed fusion a temporal model?

It is a temporal regularization operation, but not a learned temporal neural model. It uses neighbouring phases and therefore introduces temporal context; it has no hidden state, attention over time, learned kernel, selective scan, or sequence training.

### 24. Why also smooth the scalar LV-volume curve with a three-frame mean?

The two operations serve different levels. Probability fusion modifies segmentation before mask creation. The circular three-frame mean is applied afterward to each method's scalar LV-volume curve only for stable phase and smoothness analysis. Both methods use the same scalar-curve rule, so the comparison remains consistent.

## Physical volumes, phase, and ejection fraction

### 25. How was chamber volume calculated?

For each native-grid mask, I count voxels in each anatomical class. A voxel represents `sx * sy * sz / 1000` millilitres, where the header spacings are in millimetres. Class volume is voxel count multiplied by voxel volume.

### 26. How were predicted ED and ES selected?

I first apply a circular three-frame average to the predicted LV-volume curve. Predicted ED is the global maximum and predicted ES is the global minimum of that smoothed curve. Phase error is the shortest circular frame distance to the `Info.cfg` reference.

### 27. How were EDV, ESV, stroke volume, and EF calculated?

EDV and ESV are the smoothed LV volumes at predicted ED and ES. `SV = EDV - ESV`, and `EF = 100 * (EDV - ESV) / EDV`.

### 28. What is the difference between annotated-phase EF and curve-derived EF?

Annotated-phase EF uses the manual reference ED/ES indices. Reference EF comes from manual masks, and predicted EF comes from model masks at those same phases; this isolates endpoint segmentation error. Curve-derived EF uses predicted maximum and minimum phases, so it also contains phase-selection error.

### 29. How well did EF agree with the manual-mask reference?

For temporal fusion, annotated-phase EF MAE was 3.329 percentage points and Pearson `r = 0.9803`. Frame-wise inference gave 3.514 points and `r = 0.9809`. These are strong internal endpoint agreements, not external clinical validation.

### 30. How accurate was phase detection?

Temporal fusion detected ED exactly in 55% of patients and within one frame in 80%. ES was exact in 55% and within one frame in 85%. Frame-wise ED exact was 50%, while its within-one and both ES results were the same.

### 31. Why is a high EF correlation not enough by itself?

Correlation measures linear association, not absolute agreement or calibration. That is why I also report MAE in percentage points and the patient scatter. The same validation cohort selected the checkpoint, so external generalization is still unknown.

## Global motion and temporal consistency

### 32. How was LV-centroid displacement computed?

For every native-grid phase mask, I compute the three-dimensional LV-cavity centroid in millimetre coordinates. I then measure Euclidean displacement from the centroid at the annotated reference ED and report the maximum over the cycle. The temporal-fusion median was 6.252 mm.

### 33. How was myocardial radial change computed?

For each phase, I calculate the mean Euclidean distance from myocardial voxel centres to the LV-cavity centroid. I report the annotated ED radius minus the annotated ES radius. Temporal fusion produced a cohort mean of 4.153 mm.

### 34. Are centroid displacement and radial change dense cardiac motion?

No. They are global segmentation-derived geometric surrogates. Because there is no registration or voxel correspondence, they cannot identify local tissue trajectories, regional deformation, or strain.

### 35. How was curve smoothness measured?

I calculate the mean absolute circular second difference of the smoothed LV-volume curve and divide by its peak-to-peak amplitude. Smaller values mean less frame-to-frame curvature in the global trajectory.

Smoothness is not anatomical accuracy. A wrong but slowly varying contour can be smooth, so endpoint Dice and manual-mask validation remain separate.

### 36. What was the strongest temporal result?

Fusion reduced mean smoothness by `0.001129`; the paired 95% bootstrap interval was `[-0.001950, -0.000511]`, entirely below zero, and 19 of 20 patients had smoother LV curves.

### 37. Did fusion improve Dice or EF?

Not conclusively. Resized endpoint Dice changed by only `+0.0144` percentage points with CI `[-0.0635, +0.0819]`. EF absolute error changed by `-0.1853` points with CI `[-0.4678, +0.0909]`. Both intervals include zero.

The correct claim is smoother global trajectories, not resolved Dice or EF improvement.

## Validation, failure analysis, and data limitations

### 38. How do you know all expected frames were processed?

The run metadata records 20 of 20 selected patients and 550 of 550 expected phases. The frame table contains exactly 1,100 unique method-patient-frame rows and the patient table exactly 40 method-patient rows. The independent audit rejects missing, duplicate, non-finite, or inconsistent rows.

### 39. How did you verify the 4D cines correspond to the historical endpoint data?

At the annotated ED and ES indices, the full-cine image content was compared with the standalone endpoint image after the executed normalization. The maximum normalized mean absolute error was zero. Frame-wise resized endpoint Dice also reproduced the historical Nano-Mamba case table within `0.000169`.

### 40. Do all 550 phases have manual masks?

No. ACDC provides manual masks only at ED and ES. Therefore overlap accuracy is validated at 40 labelled endpoints; the remaining 510 phases support complete trajectory analysis but not direct Dice validation.

### 41. What is the main phase-detection failure?

Patient016 is the main temporal-fusion outlier: ED error is five frames and ES error three. Its predicted LV curve contains a competing late-cycle maximum. I retained this case to show that smoothing cannot guarantee correct phase selection when spatial predictions create the wrong peak.

### 42. What is the largest EF error?

Patient057 has the largest temporal-fusion annotated-phase EF absolute error, approximately 8.12 percentage points. Reporting it prevents the high overall correlation from concealing a clinically relevant individual error.

### 43. What was the issue with six recovered cine files?

Their official 4D cine affine metadata differs from the standalone endpoint containers. The pipeline did not overwrite or ignore that fact. It required matching array shape, header voxel sizes, endpoint pixel content, and endpoint image-label alignment; all passed.

Volumes use header voxel sizes. Global distances use a metric transform built from those spacings and reliable orthonormal direction information when available. I therefore do not use the mismatched raw affine as evidence of tissue correspondence.

### 44. Did you perform pathology subgroup analysis?

Only descriptive summaries. The validation composition is MINF 7, NOR 6, DCM 4, RV abnormality 2, and HCM 1. These groups are too small and imbalanced for a defensible significance or generalization claim.

### 45. What does the patient002 overlay prove?

It proves that one validation patient was processed with the correct selected checkpoint and shows the native-grid ED/ES output for both methods against manual contours. It is a traceable qualitative example, not population-level evidence.

### 46. How were uncertainty intervals calculated?

Paired patient-level percentile bootstrap with 10,000 replicates and seed 20260821. Each replicate resamples 20 patients with replacement and recomputes the mean fusion-minus-frame-wise difference. These are descriptive internal-validation intervals, not pre-registered hypothesis tests.

## Spatial architecture, training, and six-model comparison

### 47. Where does the Mamba-inspired block operate?

After three pooling stages, the feature tensor is `B x 128 x 32 x 32 x 2`. The spatial grid is flattened to 2,048 tokens with 128 channels, mixed, reshaped, and sent through the decoder.

### 48. Is this a full Mamba implementation?

No. It is a Mamba-inspired gated spatial sequence block with input projection, depthwise 1D mixing, a sigmoid gate, output projection, residual addition, and normalization. It does not implement the reference selective scan or hardware-aware Mamba algorithm.

### 49. How was patient leakage prevented?

The seed-42 split was generated at patient level: 80 training patients and 20 validation patients. Both ED and ES cases from one person stay in the same partition, yielding 160 training and 40 validation cases without patient overlap.

### 50. Is the validation cohort an independent test set?

No. It was evaluated every epoch and used to select the best checkpoint, then reused for the descriptive results and full-cine analysis. I report held-out validation performance, not independent or official ACDC test performance.

### 51. What loss and optimizer were used?

MONAI Dice plus cross entropy, with one-hot labels and softmax, optimized by AdamW at learning rate `1e-4` and weight decay `1e-5` for 150 epochs. There was no scheduler or early stopping.

### 52. How was reported segmentation Dice computed?

After `argmax`, RV, MYO, and LV Dice were computed per case on the resized grid. Each class was averaged over 40 validation cases, and the three class means were averaged for the final mean. Background was excluded from reporting.

### 53. Was the architecture comparison perfectly controlled?

No. The models shared split, preprocessing, epoch budget, main optimizer settings, checkpoint rule, and evaluation metric, but batch size, optimizer updates per epoch, normalization, capacity, and regularization differed. The ablations were not parameter matched.

### 54. Which model was most accurate?

SegResNet16 at 86.70% validation mean Dice. No-Mamba reached 85.64%; Nano-Mamba reached 84.78%. Nano-Mamba is therefore not the accuracy winner.

### 55. Why can you not claim the sequence gate improves accuracy?

No-Mamba scored 0.862 percentage points higher than Nano-Mamba. It also had 57.076% more reported parameters and structural confounds, so the result neither supports a gate advantage nor cleanly proves the gate is harmful. A parameter-matched, multi-seed ablation is needed.

### 56. What is Nano-Mamba's defensible advantage?

Its compact accuracy-efficiency operating point: 84.78% validation Dice with 1.456 million reported parameters. Versus UNet3D it gains 3.945 percentage points while using 69.714% fewer parameters.

### 57. Why is the smallest model not necessarily the fastest?

Runtime depends on kernels, memory movement, tensor layout, launch overhead, and implementation—not only parameter count. The measured FPS is hardware-specific engineering evidence, not a universal property.

## Final limitations and conclusion

### 58. What are the three most important limitations?

First, one split and seed with no independent test. Second, intermediate cine phases lack manual masks, so smoother curves do not prove every contour is correct. Third, temporal processing is fixed and global; there is no learned temporal representation or dense tissue-motion validation.

Other limits include non-native training geometry, no augmentation, non-matched ablations, small pathology subgroups, and no external clinical cohort.

### 59. What would you improve first?

Create a true train/validation/test or cross-validation design and run multiple seeds. Then use orientation- and spacing-aware preprocessing with augmentation and surface metrics. For temporal science, train a learned cine model and validate it against dense registration, correspondence, or strain references rather than only global mask trajectories.

### 60. What is your final conclusion?

Nano-Mamba U-Net is a compact spatial 3D segmenter rather than the most accurate model. The completed full-cine extension produces audited functional and global motion trajectories for all 550 phases. Fixed temporal fusion consistently smooths LV curves, but it does not establish a Dice or EF improvement and does not constitute dense cardiac motion tracking.

## Evidence pointers

- Spatial experiment: `src/21_rigorous_experiment_pipeline.py`
- Spatial bottleneck: `src/nano_mamba_core.py`
- Full-cine execution: `src/23_spatiotemporal_cine_analysis.py`
- Function/motion formulas: `src/cardiac_motion_metrics.py`
- Independent recomputation: `src/25_spatiotemporal_result_audit.py`
- Spatial evidence: `evidence/rigorous_patient_split/`
- Full-cine raw evidence: `evidence/spatiotemporal_cine/raw/`
- Full-cine audit: `evidence/spatiotemporal_cine/INDEPENDENT_AUDIT.json`
- Exact method/results boundary: `SPATIOTEMPORAL_ANALYSIS.md` and `SCIENTIFIC_BOUNDARIES.md`
