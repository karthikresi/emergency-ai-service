# Model Cards

## Category model
- Model: resq-category-v1.0
- Purpose: label incoming emergency text by category.
- Training data: synthetic prototype dataset.
- Features: normalized emergency text and context descriptors.
- Limitations: synthetic data only; not production-grade real-world validation.

## Severity model
- Model: resq-severity-v1.0
- Purpose: estimate severity from text.
- Limitations: severity labels are prototype-defined and require real-world tuning.

## Skill model
- Model: resq-skill-v1.0
- Purpose: infer responder skill requirements.
- Limitations: skill vocabulary is prototype-specific and should be aligned with the final backend schema.

## Responder model
- Model: resq-responder-v1.0
- Purpose: dispatch ranking using skill, proximity, ETA, and reliability scoring.
- Limitations: responder pool is simulated for local validation.
