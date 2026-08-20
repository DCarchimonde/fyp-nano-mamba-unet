# Rigorous Patient-Split Evidence

These four files are byte-for-byte copies of the result artefacts supplied in the P2 final-audit bundle. They are the only quantitative source for the main result table and CSV-derived thesis figures.

## Source of truth

- Experiment pipeline: `src/21_rigorous_experiment_pipeline.py`
- Main table: `summary_metrics.csv`
- Machine-readable equivalent: `summary_metrics.json`
- Patient allocation: `patient_split_seed42.json`
- Discovery summary: `data_discovery_report.json`

The split was independently checked during the 2026-08-20 audit: it contains 80 train patients and 20 validation patients, has no overlap, covers patient IDs 001–100 exactly once, and is reproduced by Python's `random.Random(42)` logic in the main pipeline. The result rows were also checked for arithmetic consistency between the three class Dice values and the reported mean, and between FPS and latency.

## Known evidence gaps

The supplied bundle did **not** contain the `per_case_*.csv` files, `training_log_*.csv` files, checkpoint metadata/hashes, saved predictions, or software/hardware environment capture referenced by the result JSON. Consequently:

- no confidence intervals or paired patient-level comparisons can yet be computed;
- exact intermediate training-curve claims cannot be verified;
- the metric convention for classes absent from both prediction and ground truth cannot be audited case by case;
- checkpoint identity and speed-test environment cannot be independently reconstructed; and
- no qualitative figure is accepted as rigorous evidence until it is regenerated from `best_NanoMambaUNet.pth` on a patient listed in the validation split.

These gaps do not authorize invented replacement values. See `paper_write/P2_FINAL_AUDIT.md` for the closure plan.

## SHA-256

```text
51122867753c24fa29edb4a906d44b1d6e0a9097f52e29d107e0cf6fb1885585  data_discovery_report.json
e79f1c238d73389900364ea32a92d954458e17e461e1c81a2e595ba653456280  patient_split_seed42.json
db5d2109f5195ad0ae7ea82d970dd57edaa9767f3b9a915e038a412a0e9cce5a  summary_metrics.csv
04d5023d659b9688de72cc198d32d43c66af6108eb329a302648117c3bd8aaa6  summary_metrics.json
```
