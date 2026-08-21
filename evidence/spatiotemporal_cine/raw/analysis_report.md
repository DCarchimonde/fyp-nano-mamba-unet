# Full-Cine Spatio-Temporal Analysis Report

Status: **complete_validation_analysis**

Patients: 20 held-out validation patients.
Frames: 550 complete cine frames.

## What was implemented

The audited 3D checkpoint was applied to every cine frame. A fixed circular 
probability fusion used weights 0.25/0.50/0.25 for previous/current/next 
frames. Full-cycle segmentation trajectories were converted into chamber 
volume curves, ED/ES phase estimates, EDV, ESV, SV, EF, LV-centroid 
displacement, and a global myocardial radial-distance surrogate.

## Main validation metrics

| Metric | Frame-wise 3D | Temporal fusion |
|---|---:|---:|
| Endpoint resized-grid mean Dice | 84.78% | 84.79% |
| ED phase exact | 50.0% | 55.0% |
| ES phase exact | 55.0% | 55.0% |
| ED phase within ±1 frame | 80.0% | 80.0% |
| ES phase within ±1 frame | 85.0% | 85.0% |
| EF MAE at reference ED/ES | 3.51 pp | 3.33 pp |
| EF Pearson r | 0.981 | 0.980 |
| Normalized curve second difference | 0.0325 | 0.0314 |

## Temporal-fusion change

- Endpoint Dice change: +0.014 percentage points.
- EF MAE change: -0.185 percentage points (negative is better).
- Curve smoothness change: -0.00113 (negative is smoother).

## Scientific boundary

This is a real full-cine temporal analysis, but it remains segmentation-derived 
global motion/function analysis. The 3D backbone was not retrained as a learned 
temporal network, and the outputs are not optical flow, dense displacement, 
local strain, or externally validated clinical measurements.
Raw endpoint/4D affine equality is audited but is not used as the grid 
registration criterion because valid ACDC files can encode identical endpoint 
voxel arrays with different qform/sform affines. Shape, voxel size, and ED/ES 
image content must match; physical magnitudes use cine header voxel sizes.
