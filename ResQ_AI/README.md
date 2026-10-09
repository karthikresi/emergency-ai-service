# ResQ AI Subsystem

This package provides a production-style AI subsystem for emergency response intelligence and dispatch. It includes a lightweight synthetic prototype dataset, safety rules, dispatch ranking, and a FastAPI service.

## Important note
The original project hybrid dataset was not present in the workspace. This package ships with a synthetic prototype dataset for reproducible local testing and backend integration. The project documentation clearly labels it as such.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Core endpoints
- GET /api/ai/health
- POST /api/ai/process-emergency
- POST /api/ai/predict-category
- POST /api/ai/predict-severity
- POST /api/ai/predict-skills
- POST /api/ai/predict-responders
- POST /api/ai/dispatch

## Public-data validation

The official NYC EMS Incident Dispatch Data and the public Disaster Response Messages dataset are evaluated separately from the synthetic ResQ data. They measure different tasks: structured NYC severity-code prediction and text-based disaster-tag classification. Neither provides ResQ responder/skill labels or validates the current API's production accuracy. See the [NYC EMS experiment guide](experiments/nyc_ems/README.md), the [Disaster Response Messages experiment guide](experiments/disaster_response/README.md), and the [external validation report](EXTERNAL_VALIDATION_REPORT.md).

## Example request

```json
{
  "emergency_text": "Fire at warehouse with smoke and injured worker",
  "latitude": 23.8103,
  "longitude": 90.4125,
  "location": "Dhaka",
  "timestamp": "2026-10-08T12:00:00Z",
  "context": {"priority": "high"}
}
```
