# P2 Reproducibility Guide

This guide reproduces every public, data-free submission artefact and separates
those checks from GPU/data-dependent inference or training.

## Full-cine spatio-temporal completion

The historical ED/ES experiment remains unchanged. The separate full-cine
analysis and its one-command Windows runner are documented in
[`SPATIOTEMPORAL_ANALYSIS.md`](SPATIOTEMPORAL_ANALYSIS.md). Do not copy its
metrics into the thesis until a complete 20-patient real-data bundle has been
generated and audited.

## 1. Data-free verification

From the repository root:

```bash
python src/22_p2_evidence_audit.py \
  --output evidence/rigorous_patient_split/evidence_audit_report.json
python -m unittest discover -s tests -v
python -m py_compile \
  src/16_thesis_visualization.py \
  src/21_rigorous_experiment_pipeline.py \
  src/22_p2_evidence_audit.py \
  src/23_spatiotemporal_cine_analysis.py \
  src/cardiac_motion_metrics.py \
  src/nano_mamba_core.py
```

Expected result: the aggregate-consistency audit passes; twenty-nine data-free
audit, visualization, and cine-analysis tests pass; four PyTorch-dependent
architecture tests pass
when PyTorch is installed and otherwise skip explicitly. This verifies the
split, result arithmetic, case tables, curves, figure inputs, and core model
behavior used by the thesis.

The repository enforces LF for audited text through `.gitattributes`. The
lineage validator also canonicalizes CRLF to LF for current source files, so a
Windows `core.autocrlf` checkout cannot produce a false hash mismatch. The
regression test verifies both CRLF acceptance and rejection of a real content
change.

## 2. Regenerate quantitative figures

```bash
python src/16_thesis_visualization.py \
  --summary-csv evidence/rigorous_patient_split/summary_metrics.csv \
  --output-dir figures
```

This writes four PNG files and `quantitative_figure_provenance.json`. Three
figures read the audited summary CSV; the fourth reads the six recovered
150-epoch logs. The manifest binds every source and output to SHA-256 hashes.

Qualitative inference is deliberately optional and fail-closed. It requires an
explicit rigorous checkpoint, audited split, ACDC path, validation patient, and
frame. Training patients and patients absent from the validation split are
rejected.

## 3. Build the thesis PDF

```bash
bash scripts/build_thesis.sh
```

The script builds in a fresh temporary directory, rejects undefined
citations/references and overfull boxes, and copies the accepted PDF to the
canonical `thesis.pdf` path. The repository includes a minimal
`bahasam.ldf` fallback and conditional font/bibliography fallbacks for small TeX
Live installations. A full submission installation should include Babel
language support, New TX fonts, Inconsolata, `apacite`, `multibib`, and
`tracklang`, which retain the UM template's preferred formatting.

Before producing a signed private submission copy, copy
`submission-private.example.tex` to the ignored `submission-private.tex` and
fill the matric and identity fields locally. Never commit the private file.

Clean-build acceptance criteria:

- exit status 0;
- A4 output;
- no missing figures;
- no undefined citations or references;
- no overfull boxes;
- title exactly matches the registered title; and
- all 73 pages visually inspected after rendering.

## 4. Scientific scope of reproduction

The retained case tables and epoch logs reproduce the main table,
checkpoint-selection epochs, training curves, empty-class-rule impact,
patient-level intervals, and paired differences. This supports the internal
six-model comparison. It does not supply an independent test cohort, multiple
training seeds, matched-capacity ablations, native-space evaluation, or true
temporal modeling; those are scientific limitations rather than file-recovery
tasks.

To audit the current 200-case dataset without training:

```powershell
python scripts\p2_dataset_manifest.py `
  --data-dir D:\AI_FYP\Data\ACDC\database\training `
  --output experiment_outputs\rigorous_patient_split\posthoc_dataset_manifest.json
```

The content-manifest command is an optional current-data integrity check. It is
not required to build or submit the thesis.

The validator also retains an optional `--strict-closure` archival mode for a
full historical attestation workflow. That mode is deliberately outside the
P2 submission gate and is not needed for the viva.

## 5. Main result interpretation

SegResNet16 has the highest validation mean Dice. The No-Mamba ablation also
exceeds Nano-Mamba U-Net. Nano-Mamba U-Net's defensible contribution is its
compact accuracy--efficiency trade-off: 84.78% validation mean Dice with 1.456M
reported parameters, not best-in-class Dice and not proof that its gated module
causes an accuracy improvement.
