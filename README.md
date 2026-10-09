# Emergency AI Service (ResQ AI)

ResQ AI is a FastAPI prototype for processing emergency descriptions. It returns a predicted incident category and severity, suggests required responder skills, applies safety rules, and ranks simulated responders for dispatch.

> **Prototype only:** The ResQ examples are based on a small synthetic dataset and keyword/rule logic. A score measured on those same examples is not an independent test of accuracy. This project has not been validated for real emergency dispatch and must not be used as a substitute for emergency services or trained dispatchers.

## Features

- Emergency category and severity predictions
- Suggested responder skills and roles
- Rule-based safety overrides
- Prototype responder ranking and dispatch-radius expansion
- FastAPI endpoints with interactive OpenAPI documentation
- Separate experiments for public NYC EMS and disaster-response message datasets

## Run locally

Requirements: Python 3.10 or later.

```bash
cd ResQ_AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Once running, open:

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/ai/health

Alternatively, from `ResQ_AI`, run `docker compose up --build`.

## Example request

```bash
curl -X POST http://localhost:8000/api/ai/process-emergency \
  -H "Content-Type: application/json" \
  -d '{
    "emergency_text": "There is a road accident and one person is in critical stage",
    "latitude": 23.8103,
    "longitude": 90.4125,
    "location": "Dhaka",
    "context": {"priority": "critical"}
  }'
```

The response includes the category, severity, required skills, triggered safety rules, ranked simulated responders, dispatch status, and model version identifiers. The API also accepts an optional `timestamp`.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/ai/health` | Service health |
| `POST` | `/api/ai/process-emergency` | Run the full emergency pipeline |
| `POST` | `/api/ai/predict-category` | Predict an incident category |
| `POST` | `/api/ai/predict-severity` | Predict severity |
| `POST` | `/api/ai/predict-skills` | Suggest required skills |
| `POST` | `/api/ai/predict-responders` | List responder roles |
| `POST` | `/api/ai/dispatch` | Rank simulated responders and dispatch |

All prediction and processing endpoints accept JSON containing a required `emergency_text` string. Optional fields are `latitude`, `longitude`, `location`, `timestamp`, and a `context` object.

## Project layout

```text
ResQ_AI/
├── api/                  # FastAPI application and request schemas
├── configs/              # Model, feature, threshold, and dispatch settings
├── data/hybrid/           # Synthetic prototype data
├── docs/                 # API, architecture, dataset, and model notes
├── experiments/           # Separate external-dataset experiments
├── src/                   # Prediction, safety, preprocessing, and dispatch logic
└── tests/                 # Automated tests
```

## Run tests

From `ResQ_AI`, with the virtual environment active:

```bash
pytest -q
```

## Data and experiments

The ResQ prototype dataset is synthetic and intentionally small. The experiments under `ResQ_AI/experiments/` evaluate different tasks and datasets; their metrics are not directly comparable to ResQ category, severity, skills, or responder predictions. NYC EMS data supports structured NYC severity-code and response-time research, while the disaster-response messages data supports text tagging. See the experiment-specific READMEs and [the external validation report](ResQ_AI/EXTERNAL_VALIDATION_REPORT.md) for methodologies and limitations.

The repository also tracks `Datasets/emergency_dataset_v2.csv` with Git LFS. Install Git LFS to download its contents when cloning; it is not required to run the API.
