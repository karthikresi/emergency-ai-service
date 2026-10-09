# Leakage Audit

No target-derived operational fields were used in training. The synthetic prototype intentionally excludes post-event fields such as assigned responder, resolution status, and actual response time.

## Excluded leakage items
- assigned_responder
- response_status
- resolution_status
- actual_response_time

## Decision
These fields remain out of scope and are not allowed in the prediction pipeline.
