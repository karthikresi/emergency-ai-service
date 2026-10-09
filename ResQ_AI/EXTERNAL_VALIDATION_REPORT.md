# External Validation Report

Two public datasets are now evaluated separately. Neither is a substitute for the missing real ResQ hybrid dataset, and their scores are not directly comparable because they use different inputs, labels, and prediction tasks.

## Measured held-out results

| Dataset and task | Test metric | Result | 99%+? |
|---|---|---:|:---:|
| NYC EMS initial-severity code classification | Accuracy | 98.76% | No |
| NYC EMS initial-severity code classification | Macro F1 | 99.00% | Not an accuracy score |
| Disaster Response Messages binary tags | Label-wise accuracy | 95.14% | No |
| Disaster Response Messages binary tags | Exact-match/subset accuracy | 34.45% | No |
| Disaster Response Messages binary tags | Micro F1 / trainable-label macro F1 | 64.96% / 52.84% | Not accuracy |
| Disaster Response Messages `related` classification | Accuracy / macro F1 | 79.60% / 62.30% | No |

The NYC numbers come from the recorded 30,000-row temporal test split in [`experiments/nyc_ems/artifacts/20261008T144715Z/experiment_metrics.json`](./experiments/nyc_ems/artifacts/20261008T144715Z/experiment_metrics.json). The selected model's macro F1 rounds to 99.00%, but its accuracy is 98.76%.

The disaster-message numbers come from [`experiments/disaster_response/artifacts/20261008T172547Z/experiment_metrics.json`](./experiments/disaster_response/artifacts/20261008T172547Z/experiment_metrics.json). It uses the publisher's test split and removes exact normalized message duplicates already present in earlier splits. There are 36 binary tags, three constant in training, and a separate 3-class `related` target. A tag-set prediction counts as exact only when every tag for a message is correct; that is why exact-match accuracy is much lower than label-wise accuracy. Three constant tags are restored at their training value, and only the 33 trainable tags contribute to macro F1. The Haiti-only `actionable_haiti` target is not included in this general multi-label task.

## Interpretation and limitations

- The requested 99%+ accuracy has **not** been demonstrated on both real-world datasets. Do not claim it from these results.
- An exploratory unweighted NYC logistic-regression variant scored 99.09% accuracy on the same 2025 second-half test sample, but it was not selected by the predeclared validation metric. Since multiple variants were subsequently compared on that test sample, this score is exploratory, not independent confirmation; use a fresh future-period holdout before making a 99% claim for an accuracy-selected model.
- NYC EMS is structured intake-field prediction of local EMS severity codes. It does not have incident narrative text or ResQ skill/responder labels.
- Disaster Response Messages is text classification for disaster communication categories. It does not label NYC EMS severity codes or responder dispatch outcomes.
- Disaster-message accuracy depends on the metric: per-label binary decisions, full exact-match tag sets, and the separate `related` classification are distinct tasks. Reporting only elementwise accuracy would hide the multi-label errors.
- The Disaster Response Messages official split includes the same disaster events in train and test; this is a held-out message split, not an unseen-disaster generalization test.
- Both test results are benchmarks, not operational guarantees. Achieving a target score is not assured; changing thresholds or labels to inflate accuracy would make the result less meaningful.
- ResQ production endpoints still use a synthetic prototype and rule-based logic. Real ResQ labels and a governed, representative prospective evaluation set are needed before making production accuracy claims.

## Reproduction

From `ResQ_AI`:

```bash
.venv/bin/python experiments/nyc_ems/run_experiment.py
.venv/bin/python experiments/disaster_response/run_experiment.py
```

Each run needs internet access and creates a timestamped metrics artifact. See the [NYC EMS experiment guide](./experiments/nyc_ems/README.md) and [Disaster Response Messages experiment guide](./experiments/disaster_response/README.md) for task and source details.
