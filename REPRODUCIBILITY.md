# P2 Reproducibility Guide

This guide reproduces every public, data-free submission artefact and separates
those checks from GPU/data-dependent inference or training.

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
  src/nano_mamba_core.py
```

Expected result: the aggregate-consistency audit passes; the nine data-free
audit tests pass; PyTorch-dependent architecture tests pass when PyTorch is
installed and otherwise skip explicitly. This default pass is not an
end-to-end provenance claim. Strict closure remains incomplete until the
original experiment artefacts are added.

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

This writes three PNG files and `quantitative_figure_provenance.json`. The
manifest binds the figures to the audited CSV and plotting-script hashes.

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
- all 71 pages visually inspected after rendering.

## 4. What cannot be reproduced from the public evidence alone

The supplied final-audit bundle omitted original per-case CSVs, epoch logs,
checkpoint files/metadata, saved predictions, and confirmed runtime metadata.
Therefore, the repository does not claim that it can independently reconstruct
the reported weights, learning curves, speed environment, confidence intervals,
or a qualitative validation image. `historical_source_lineage.json` records the
limited source/ZIP timeline that Git and archive hashes can establish, but it
does not close those run-level gaps. Use the Windows closure collector described
in `RIGOROUS_EXPERIMENTS.md` on the original machine.

## 5. Main result interpretation

SegResNet16 has the highest validation mean Dice. The No-Mamba ablation also
exceeds Nano-Mamba U-Net. Nano-Mamba U-Net's defensible contribution is its
compact accuracy--efficiency trade-off: 84.78% validation mean Dice with 1.456M
reported parameters, not best-in-class Dice and not proof that its gated module
causes an accuracy improvement.
