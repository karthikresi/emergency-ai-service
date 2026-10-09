# Disaster Response Messages experiment

This experiment evaluates the public [Disaster Response Messages dataset](https://github.com/rmunro/disaster_response_messages) on the publisher-provided training, validation, and test splits. The script downloads a pinned source revision and writes timestamped metrics under `artifacts/`.

Run from `ResQ_AI`:

```bash
.venv/bin/python experiments/disaster_response/run_experiment.py
```

The model uses message text only. It excludes identifiers, split names, genre, event names, and Haiti-only outcome/date fields. Exact normalized duplicate messages are kept only in their earliest split to prevent train/validation/test contamination.

The dataset contains a three-class `related` target and multiple binary category tags. The experiment keeps those tasks separate, tunes on validation data, and reports the test accuracy alongside macro/micro F1. For the tag task it reports both label-wise accuracy and strict subset (exact-match) accuracy; these are different metrics and must not be presented interchangeably. Labels with no positive examples in training are predicted at their observed constant value and excluded from trainable-label macro F1.

The NYC EMS experiment is a separate structured severity-code task with different input fields and labels. Its accuracy is not directly comparable to this message-classification experiment. Neither benchmark justifies a claim that ResQ's synthetic-data endpoint is production-accurate.

The source README states a Creative Commons Attribution license. Cite Robert Munro, *Processing short message communications in low-resource languages* (Stanford University, 2012) when redistributing or reporting results.
