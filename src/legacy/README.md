# Legacy exploratory scripts

The files in this directory are retained only as historical implementation
records. They are **not** part of the P2 evidence chain, are not imported by
the rigorous experiment pipeline, and must not be used to support thesis
claims.

Several scripts contain informal console text and superseded terminology such
as “spatio-temporal” or “Mamba” for a module that processes a flattened
**spatial** grid. In the evaluated implementation:

- ED and ES volumes are independent 3D cases;
- the depth axis is anatomical slice depth, not time;
- the bottleneck is a lightweight Mamba-inspired gated sequence module, not a
  full selective state-space scan; and
- the only accepted main-result source is `src/21_rigorous_experiment_pipeline.py`
  together with `evidence/rigorous_patient_split/summary_metrics.csv`.

Use `src/21_rigorous_experiment_pipeline.py` for a new rigorous run and
`src/22_p2_evidence_audit.py` to audit the resulting evidence.
