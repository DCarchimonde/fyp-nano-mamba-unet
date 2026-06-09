# Rigorous Experiment Pipeline

Use `src/21_rigorous_experiment_pipeline.py` for the thesis result table.

This pipeline fixes the earlier research-risk where training and evaluation could happen on the same ACDC folder. It uses a deterministic patient-level train/validation split, saves the best checkpoint by validation mean Dice, and reports final metrics on held-out validation patients only.

Run in PyCharm Terminal:

```bash
cd /d D:\AI_FYP
python src\21_rigorous_experiment_pipeline.py
```

Main output:

```text
D:\AI_FYP\experiment_outputs\rigorous_patient_split\summary_metrics.csv
```

Report wording:

> All models were trained using the same deterministic patient-level 80/20 split of the ACDC training cohort. The best checkpoint for each model was selected using validation mean Dice. Final quantitative results were reported only on held-out validation patients.
