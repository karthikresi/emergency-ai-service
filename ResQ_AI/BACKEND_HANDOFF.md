# Backend Handoff

## AI service URL
http://localhost:8000

## Endpoints
- GET /api/ai/health
- POST /api/ai/process-emergency
- POST /api/ai/predict-category
- POST /api/ai/predict-severity
- POST /api/ai/predict-skills
- POST /api/ai/predict-responders
- POST /api/ai/dispatch

## Startup command
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

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
