# Research Results

This project implements a prototype emergency response AI pipeline based on a synthetic prototype dataset. The primary focus is on reproducible orchestration of emergency category detection, severity estimation, required-skill prediction, and dispatch optimization.

## Methodology
- Synthetic prototype dataset created for local validation and backend integration.
- Text features normalized and mapped to deterministic safety rules.
- Ranking logic uses skill compatibility, availability, distance, ETA, reliability, and acceptance probability.

## Validation
- Health endpoint validated.
- Handling for missing optional context and invalid text is implemented.
- Dispatch ranking is configured via YAML and returns deterministic scores.

## Limitations
- The dataset is synthetic and should not be treated as real-world emergency data.
- The responder pool is a simulation used for local service testing.
