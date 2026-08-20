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

The first command must print `AGGREGATE CONSISTENCY PASS`. `Strict closure
status: incomplete` is the expected truthful result until the original
experiment machine supplies the full discovery manifest, per-case CSVs,
training logs, checkpoint manifest, confirmed environment, and confirmed
historical command. The default pass establishes internal consistency only. To
make the provenance gaps fail a CI/submission gate:

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

Run this from the original experiment environment. Confirm the switches only
if the current environment and checkpoint directory are the historical ones:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\collect_p2_closure_bundle.ps1 `
  -ProjectRoot D:\AI_FYP `
  -TrainingCommand "python src\21_rigorous_experiment_pipeline.py" `
  -ConfirmHistoricalEnvironment `
  -ConfirmCheckpointSet
```

The collector copies small CSV/JSON records, hashes checkpoints without placing
weights in the ZIP, captures software/hardware details, runs the strict audit,
and writes a closure ZIP under the rigorous output directory. A normal Git
commit should not include ACDC data or checkpoint weights.

## Statistical scope

If all six original `per_case_*.csv` files are recovered, the validator computes
deterministic patient-level percentile-bootstrap 95% confidence intervals and
paired Nano-Mamba differences. These are explicitly post-hoc descriptive
analyses. They do not turn the single 80/20 split into an independent test set
and do not replace multi-seed or external validation.

## Historical reporting language

> All models were trained using the same deterministic patient-level 80/20
> split of the ACDC training cohort. The best checkpoint for each model was
> selected using validation mean Dice. Final quantitative results were reported
> on the same held-out validation patients. This is not an independent test-set
> estimate.
