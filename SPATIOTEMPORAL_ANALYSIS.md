# Full-Cine Spatio-Temporal Analysis

## Completion status

The real-data run completed on 2026-08-22 from Git commit
`dace079c0a3d4025aff36d159b0c732947516393`. It processed all 20 validation
patients and all 550 cine frames on an NVIDIA RTX 4060 Laptop GPU. The sealed
raw results and an independent recomputation are committed under
`evidence/spatiotemporal_cine/`.

## Why this experiment is needed

The historical six-model experiment used ED and ES as independent 3D samples.
Its depth axis contains anatomical slices, not cardiac time. Therefore, that
experiment supports 3D segmentation but cannot by itself support a completed
spatio-temporal cardiac-motion claim.

This workflow adds a new, explicitly separate analysis of the complete ACDC 4D
cine sequences. It does **not** retrain the six historical models or alter
their reported comparison.

## Implemented analysis

For each of the 20 existing validation patients:

1. recursively inspect NIfTI headers to locate the complete 4D cine (including
   unpacked `.nii` directory layouts), then verify `NbFrame`, ED, and ES against
   `Info.cfg`;
2. prove that the cine ED/ES image frames match the standalone 3D endpoint
   images used by the original experiment;
3. run the audited Nano-Mamba checkpoint on every 3D cardiac frame;
4. reproduce the historical endpoint Dice within a fixed tolerance;
5. compare raw frame-wise masks with a pre-specified temporal probability
   fusion:

   \[
   \widetilde{P}_t = 0.25P_{t-1} + 0.50P_t + 0.25P_{t+1},
   \]

   using circular boundaries because the cine sequence is a cardiac cycle;
6. derive complete LV, RV, and myocardium volume trajectories;
7. detect ED/ES from the LV curve and compare them with `Info.cfg`;
8. calculate EDV, ESV, stroke volume, ejection fraction, LV-centroid
   displacement, and a global myocardial radial-distance surrogate;
9. validate ED/ES segmentation and EF against the labelled endpoint masks;
10. export patient/frame CSVs, figures, provenance, hashes, and a concise report.

## One-command Windows run

Activate the existing `nanomamba` environment, switch to
`p2-final-audit-2026-08-20`, pull the new commit, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_spatiotemporal_cine_analysis.ps1 `
  -ProjectRoot D:\AI_FYP `
  -RepairMissingCine
```

The script uses CUDA, batch size 1, the existing Nano-Mamba checkpoint, the
audited validation split, and the complete local ACDC dataset. It does not
train any network. Progress is printed patient by patient.

`-RepairMissingCine` first searches the local ACDC tree for a misplaced cine.
For the six cines found missing during the 2026-08-21 Windows preflight, it can
download only the required files (91.6 MiB total) from the public
`msepulvedagodoy/acdc` mirror pinned at revision
`067262d5b40f9c976f7139c13416ace5a3314f42`. Each download has a fixed byte
count and SHA-256, and it is installed only after its ED and ES image content,
spatial shape, header voxel sizes, and `NbFrame` agree with the existing local
patient files. Raw qform/sform affine equality is recorded but is not an
identity gate: the public ACDC release contains matching 4D and standalone
endpoint arrays with different affine metadata. Physical distances and volumes
use the cine header voxel sizes. The source and checks are written to a
recovery manifest under `experiment_outputs`. A byte-identical fallback mirror
is also pinned for transient repository failures. A full official ACDC archive
remains an alternative.

On success it creates a timestamped directory and ZIP under
`D:\AI_FYP\experiment_outputs`, for example:

```text
D:\AI_FYP\experiment_outputs\spatiotemporal_cine_YYYYMMDD_HHMMSS\
D:\AI_FYP\experiment_outputs\spatiotemporal_cine_YYYYMMDD_HHMMSS.zip
```

The successful ZIP `spatiotemporal_cine_20260822_011130.zip` was independently
audited and integrated into the thesis and defence material. Its SHA-256 is
`6d9a153a0414972a07157ff3783d0740c8311dc4dbebe4daf134538e18bc1ab1`.

## Audited results

| Quantity | Frame-wise | Temporal fusion |
|---|---:|---:|
| Resized-grid endpoint mean Dice | 84.779% | 84.794% |
| Native-grid endpoint mean Dice | 77.919% | 77.923% |
| Annotated-phase EF MAE | 3.514 pp | 3.329 pp |
| Annotated-phase EF correlation | 0.9809 | 0.9803 |
| ED exact / within one frame | 50% / 80% | 55% / 80% |
| ES exact / within one frame | 55% / 85% | 55% / 85% |
| Normalized LV-curve second difference | 0.03252 | 0.03139 |
| Median peak LV-centroid displacement | 6.361 mm | 6.252 mm |
| Mean MYO radial ED-minus-ES change | 4.250 mm | 4.153 mm |

The paired temporal-fusion-minus-frame-wise endpoint-Dice interval is
[-0.063, +0.082] percentage points and the EF-error interval is
[-0.468, +0.091] percentage points; both include zero. The curve-smoothness
change is -0.001129 with interval [-0.001950, -0.000511], and 19/20 curves are
smoother. Therefore, claim smoother global trajectories, not a resolved Dice or
EF improvement.

Re-run the independent result audit with:

```powershell
python src\25_spatiotemporal_result_audit.py `
  --result-dir evidence\spatiotemporal_cine\raw `
  --repo-root . `
  --output evidence\spatiotemporal_cine\INDEPENDENT_AUDIT.json
```

To isolate categorical resampling from model error, run the endpoint-label
round-trip diagnostic on the local ACDC data:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_native_grid_roundtrip_audit.ps1 `
  -ProjectRoot D:\AI_FYP
```

This applies the exact nearest-neighbour
`native -> 256 x 256 x 16 -> native` path to the 40 manual ED/ES masks and
writes a per-endpoint CSV plus `native_label_roundtrip_summary.json`. It does
not train or run a model. The result measures label-boundary fidelity and is
not a strict mathematical upper bound on a particular prediction's restored
Dice.

Source review and a non-cubic regression test confirm that the executed inverse
resize uses the original `(X,Y,Z)` shape with no axis permutation and preserves
categorical class identifiers. The observed 6.861 percentage-point frame-wise
grid difference must not be attributed solely to the Z axis: direct in-plane
and through-plane resizing, heterogeneous native shapes, thin-structure
boundary discretization, and model error can all contribute.

## Fail-closed checks

Before the first model inference, all 20 patient directories are scanned and
their cine is identified by a four-dimensional NIfTI header whose time axis
equals `Info.cfg` `NbFrame`. This avoids assuming a particular archive or
filename layout. The run stops instead of emitting reportable results when any
of these occur:

- a validation patient, `Info.cfg`, 4D cine, endpoint image, or label is missing;
- the NIfTI time dimension differs from `NbFrame`;
- cine and endpoint spatial shape or header voxel sizes differ;
- the cine ED/ES image does not match the original standalone image;
- the checkpoint hash/model/state dictionary differs from the audited record;
- frame-wise endpoint Dice cannot reproduce the recovered historical case row;
- the model produces non-finite probabilities;
- fewer than all 20 validation patients are processed in the final run.

A raw affine mismatch between the 4D cine and its standalone endpoint does not
stop the run. It is preserved in the input and recovery manifests, while array
registration is established by matching spatial dimensions, voxel spacing, and
endpoint image content. The endpoint image and its manual label must still
match each other's affine. This handles the documented ACDC container-header
inconsistency without weakening checks that can detect a wrong patient, frame,
voxel grid, image payload, or label alignment.

## Exact scientific boundary

The completed thesis can claim a full-cine, segmentation-derived
spatio-temporal motion/function framework. It must still state that:

- the checkpoint-compatible Nano-Mamba backbone is a spatial 3D network;
- temporal context is introduced by fixed adjacent-frame probability fusion,
  not a learned temporal Mamba block;
- the sealed run stores probability maps as float16 before converting them
  back to float32 for fixed fusion; frame-wise argmax is retained before that
  storage conversion, and no precision-independent fusion gain is claimed;
- ACDC provides manual masks only at ED and ES, so intermediate masks are not
  directly ground-truthed;
- centroid displacement and myocardial radial-distance change are global
  segmentation-derived motion surrogates;
- the workflow is not optical flow, dense registration, regional strain, or
  external clinical validation;
- the same validation cohort was used historically for checkpoint selection.

These boundaries do not erase the temporal analysis. They define exactly what
was completed and prevent the stronger, unsupported claim of dense cardiac
motion tracking.
