# Audited Full-Cine Spatio-Temporal Evidence

This directory seals the completed 20-patient ACDC validation analysis generated
from Git commit `dace079c0a3d4025aff36d159b0c732947516393` on an NVIDIA RTX
4060 Laptop GPU. The received ZIP was
`spatiotemporal_cine_20260822_011130.zip` with SHA-256
`6d9a153a0414972a07157ff3783d0740c8311dc4dbebe4daf134538e18bc1ab1`.

## Contents

- `raw/` contains the immutable analysis artifacts exactly as supplied: the
  per-frame and per-patient CSVs, input and run provenance, aggregate summary,
  analysis report, artifact manifest, and all 23 generated figures.
- `INDEPENDENT_AUDIT.json` is a second implementation's recomputation of the
  artifact hashes, source lineage, cohort coverage, segmentation arithmetic,
  physical volumes, EF, phase detection, curve smoothness, motion surrogates,
  pathology summaries, and patient-level bootstrap intervals.
- `figures/` contains exact copies of the five figures selected for thesis and
  defense use. The copies are convenience views; `raw/` is authoritative.

Run the independent audit from the repository root:

```powershell
python src\25_spatiotemporal_result_audit.py `
  --result-dir evidence\spatiotemporal_cine\raw `
  --repo-root . `
  --output evidence\spatiotemporal_cine\INDEPENDENT_AUDIT.json
```

## Verified scope

- 20 validation patients and 550 complete cine frames were processed.
- The frame table contains 1,100 rows: 550 frame-wise and 550 temporally fused.
- The patient table contains 40 rows: two methods for each patient.
- Standalone ED/ES images were numerically identical to their corresponding
  frames in the complete cine after normalization (maximum normalized MAE 0).
- Recomputed historical endpoint Dice differed by at most `0.0001691055`, which
  is consistent with the recorded post-processing path.
- For six recovered public ACDC cines, the raw 4D affine metadata differs from
  the standalone endpoint affine. Shape, voxel sizes, and endpoint pixel content
  match; endpoint image and label geometry also match. Physical measurements use
  the NIfTI header voxel sizes rather than the inconsistent affine scale.

## Main results

| Quantity | Frame-wise | Temporal fusion |
|---|---:|---:|
| Endpoint mean Dice on historical resized grid | 84.779% | 84.794% |
| Endpoint mean Dice after native-grid mask restoration | 77.919% | 77.923% |
| Annotated ED/ES EF MAE | 3.514 percentage points | 3.329 percentage points |
| Annotated ED/ES EF Pearson correlation | 0.9809 | 0.9803 |
| ED phase exact / within one frame | 50% / 80% | 55% / 80% |
| ES phase exact / within one frame | 55% / 85% | 55% / 85% |
| Normalized LV-curve second difference | 0.03252 | 0.03139 |
| Median peak LV-centroid displacement | 6.361 mm | 6.252 mm |
| Mean MYO radial ED-minus-ES change | 4.250 mm | 4.153 mm |

Paired temporal-fusion-minus-frame-wise results were:

- endpoint Dice `+0.0001441`, 95% paired bootstrap interval
  `[-0.0006350, +0.0008193]`;
- EF absolute error `-0.1853` percentage points, interval
  `[-0.4678, +0.0909]`;
- normalized curve second difference `-0.0011294`, interval
  `[-0.0019497, -0.0005109]`, with smoother trajectories for 19 of 20
  patients.

The first two intervals include zero; they do not support a resolved Dice or EF
improvement. The smoothness interval is entirely below zero and supports the
narrow claim that the fixed temporal fusion produced smoother
segmentation-derived LV-volume trajectories on this cohort.

## Claim boundary

This is a real full-cine, segmentation-derived analysis, but the temporal step
is fixed probability fusion
`0.25 P(t-1) + 0.50 P(t) + 0.25 P(t+1)`. It is not a learned temporal Mamba
network, optical flow, dense deformation, regional strain, or clinical motion
tracking. Manual masks exist only at ED and ES, and the same validation cohort
was used historically for checkpoint selection. Therefore these are internal
validation results, not independent test or deployment evidence.
