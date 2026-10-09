# NYC EMS external experiment

This experiment profiles the official NYC EMS Incident Dispatch Data and evaluates only targets the source actually records. It keeps the NYC data and artifacts separate from the synthetic ResQ data.

## Run

From `ResQ_AI`:

```bash
.venv/bin/python experiments/nyc_ems/run_experiment.py
```

The script needs internet access to NYC Open Data. It performs full-dataset aggregate inspection, downloads a fixed month-stratified 2024–2025 sample, uses 2024 for training, the first half of 2025 for validation, and the second half of 2025 as the test set. Each run creates a timestamped directory under `artifacts/` containing the report, source profile, leakage audit, exact model sample, metrics, and test predictions.

## What the data can support

- Directly: prediction of NYC's initial EMS severity **code** from intake-time structured fields, plus dispatch/incident response-time regression.
- Not directly: ResQ's narrative-text category/severity labels, required skills, responder types, volunteer availability/acceptance, or dispatch assignment ranking. The source has no narrative text or dispatched unit/resource type.

The experiment therefore does not replace a ResQ production model. NYC severity codes are not translated into ResQ's four-level scale. See the timestamped `artifacts/*/report.md` for the source profile, complete field audit, methodology, metrics, and limitations.
