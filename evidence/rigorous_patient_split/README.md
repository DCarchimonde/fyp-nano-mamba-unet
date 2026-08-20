# Rigorous Patient-Split Evidence

This directory is the public, metadata-only evidence record for the reported
experiment. It contains the original four aggregate artefacts plus records
recovered from `p2_closure_20260821_020939.zip` on 2026-08-21.

## Recovered evidence

- six `per_case_*.csv` files: 40 cases and 20 patients per model;
- six `training_log_*.csv` files: contiguous epochs 1--150 per model;
- `checkpoint_manifest.json`: six readable checkpoint metadata records and six
  unique full-file SHA-256 hashes (weights are not committed);
- `environment.json`: a current-machine capture whose historical identity is
  not yet confirmed;
- `run_transcript.txt`: explicitly unconfirmed rather than reconstructed; and
- the collector report, receipt, and raw-byte checksum list.

`RECOVERED_CLOSURE_SHA256SUMS.txt` preserves the hashes supplied inside the ZIP.
The path-specific `.gitattributes` rules prevent CRLF conversion from changing
these recovered bytes. `RECOVERED_CLOSURE_RECEIPT.json` records the source ZIP
SHA-256, archive timeline, and confirmation boundaries.

## Verified numerical chain

The validator independently establishes all of the following:

- each model covers the same 40 validation cases, exactly two per patient;
- every case mean is the arithmetic mean of RV, MYO, and LV Dice;
- each per-case aggregate equals `summary_metrics.csv` and its JSON mirror;
- each log contains 150 finite, contiguous epoch rows and its best row equals
  the summary;
- each checkpoint record has the matching model, epoch, Dice, split seed,
  validation fraction, input size, and epoch budget; and
- none of the 720 recovered foreground class scores equals 1.0, so the
  executed empty-prediction/empty-reference branch did not affect the table.

The audit also computes post-hoc patient-level percentile-bootstrap intervals
with 10,000 replicates and seed 20260820. The paired Nano-Mamba differences in
percentage points are:

| Reference | Difference | 95% interval |
|---|---:|---:|
| UNet3D | +3.945 | [+2.829, +5.009] |
| Attention U-Net | +10.001 | [+3.508, +18.294] |
| SegResNet16 | -1.918 | [-3.010, -0.884] |
| No-Mamba | -0.862 | [-1.482, -0.230] |
| Half-Mamba | -0.174 | [-0.909, +0.551] |

These are descriptive intervals on the validation patients used for checkpoint
selection, not pre-registered significance tests or independent test results.

## Remaining historical-provenance gaps

Strict closure truthfully remains incomplete for four reasons:

1. the currently found checkpoint set has not been confirmed unchanged from
   the original run;
2. the current environment capture has not been confirmed as the historical
   training/benchmark environment;
3. the exact historical command and working directory remain unconfirmed; and
4. the historical discovery report contains only five example paths rather
   than a full 200-case content manifest.

`scripts/p2_dataset_manifest.py` can now audit all 200 current NIfTI pairs,
including content hashes, geometry, finite values, and labels 0--3. It binds the
historical experiment only if the candidate can truthfully confirm that the
current dataset tree is the same snapshot used for training.
