# Rigorous Experiment and Evidence Workflow

## Authority of results

The main thesis results are restricted to:

- training/evaluation code: `src/21_rigorous_experiment_pipeline.py`;
- numerical source of truth: `evidence/rigorous_patient_split/summary_metrics.csv`;
- audited split: `evidence/rigorous_patient_split/patient_split_seed42.json`.

Earlier scripts, P1 documents, training-set evaluations, and copied numbers are
not valid sources for the final comparison table.

`evidence/rigorous_patient_split/historical_source_lineage.json` binds the
supplied aggregate-file hashes to source versions that existed before the ZIP's
summary timestamp. This is useful temporal context, but it is explicitly not a
substitute for checkpoints, logs, environment metadata, or a command transcript.

## Audit the supplied evidence (no dataset or GPU required)

```powershell
python src\22_p2_evidence_audit.py `
  --output evidence\rigorous_patient_split\evidence_audit_report.json
python -m unittest discover -s tests -v
```

The first command must print `AGGREGATE CONSISTENCY PASS`. The recovered six
per-case tables, six logs, and six checkpoint records are now validated by this
command. `Strict closure status: incomplete` remains the expected truthful
result until checkpoint-set identity, the historical environment, the exact
command, and the complete historical dataset snapshot are confirmed. To make
those remaining provenance gaps fail a CI/submission gate:

```powershell
python src\22_p2_evidence_audit.py --strict-closure
```

## Run a future rigorous experiment

The historical Windows layout remains the compatibility default, but source
editing is no longer required:

```powershell
python src\21_rigorous_experiment_pipeline.py `
  --data-dir D:\AI_FYP\Data\ACDC\database\training `
  --output-dir D:\AI_FYP\experiment_outputs\rigorous_patient_split
```

The pipeline now rejects a cached split if its seed, fraction, unique patient
sets, or current-patient coverage differ. A future run also records a complete
de-identified case manifest, source hashes, command, package versions, batch
sizes, benchmark protocol, Git commit (when available), and device details.
Those additions describe future runs only; they do not reconstruct missing
metadata for the already reported experiment.

## Collect the original closure evidence on Windows

Run this first without confirmation switches. It collects the current facts and
also audits all 200 NIfTI image/label pairs when `-DataDir` exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\collect_p2_closure_bundle.ps1 `
  -ProjectRoot D:\AI_FYP `
  -ExperimentOutput D:\AI_FYP\experiment_outputs\rigorous_patient_split `
  -DataDir D:\AI_FYP\Data\ACDC\database\training
```

Inspect checkpoint timestamps, Conda history, and terminal/PyCharm history
before adding any of `-ConfirmCheckpointSet`,
`-ConfirmHistoricalEnvironment`, `-TrainingCommand`, or
`-ConfirmHistoricalDatasetSnapshot`. The collector copies only small
CSV/JSON/metadata records, hashes checkpoints without placing weights in the
ZIP, and never copies ACDC data.

The ZIP is written even when historical confirmation remains incomplete, but
the collector then returns exit code 2 so automation cannot mistake a partial
bundle for strict closure. Exit code 1 or another unexpected audit failure is
treated as invalid evidence and stops collection.

## Statistical scope

All six original `per_case_*.csv` files have been recovered. The validator now
computes deterministic patient-level percentile-bootstrap 95% confidence
intervals and paired Nano-Mamba differences (10,000 replicates, seed 20260820).
These are explicitly post-hoc descriptive analyses. They do not turn the
single 80/20 split into an independent test set and do not replace multi-seed
or external validation.

## Historical reporting language

> All models were trained using the same deterministic patient-level 80/20
> split of the ACDC training cohort. The best checkpoint for each model was
> selected using validation mean Dice. Final quantitative results were reported
> on the same held-out validation patients. This is not an independent test-set
> estimate.
