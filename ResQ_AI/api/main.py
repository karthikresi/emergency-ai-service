import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from api.schemas import EmergencyRequest
from src.dispatch.dispatch_engine import DispatchEngine
from src.prediction.pipeline import EmergencyPredictionPipeline
from src.safety.safety_rules import evaluate_safety_rules

app = FastAPI(title='ResQ AI Service', version='1.0.0')

pipeline = EmergencyPredictionPipeline()
dispatch_engine = DispatchEngine({
    'initial_radius_km': 1,
    'second_radius_km': 5,
    'third_radius_km': 10,
    'max_radius_km': 20,
    'max_notifications': 2,
    'escalation_threshold': 1,
})


@app.get('/api/ai/health')
def health():
    return {'status': 'ok', 'service': 'resq-ai'}


@app.post('/api/ai/predict-category')
def predict_category(payload: EmergencyRequest):
    return pipeline.predict_category(payload.emergency_text)


@app.post('/api/ai/predict-severity')
def predict_severity(payload: EmergencyRequest):
    return pipeline.predict_severity(payload.emergency_text)


@app.post('/api/ai/predict-skills')
def predict_skills(payload: EmergencyRequest):
    return {'required_skills': pipeline.predict_skills(payload.emergency_text)}


@app.post('/api/ai/predict-responders')
def predict_responders(payload: EmergencyRequest):
    return {'responders': pipeline.predict_responders(payload.emergency_text)}


@app.post('/api/ai/dispatch')
def dispatch(payload: EmergencyRequest):
    required = [item['skill'] for item in pipeline.predict_skills(payload.emergency_text)]
    candidate_responders = [
        {'id': 'r1', 'skill': 'Ambulance', 'available': True, 'distance_km': 1.2, 'eta_min': 4, 'reliability': 0.82, 'experience_years': 6, 'acceptance_probability': 0.88},
        {'id': 'r2', 'skill': 'Fire Fighter', 'available': True, 'distance_km': 3.0, 'eta_min': 8, 'reliability': 0.9, 'experience_years': 9, 'acceptance_probability': 0.91},
        {'id': 'r3', 'skill': 'Police', 'available': False, 'distance_km': 0.7, 'eta_min': 2, 'reliability': 0.76, 'experience_years': 5, 'acceptance_probability': 0.75},
    ]
    ranked = dispatch_engine.rank_responders(candidate_responders, required)
    return {'dispatch': dispatch_engine.dispatch(candidate_responders, required, 1.0), 'ranked': ranked}


@app.post('/api/ai/process-emergency')
def process_emergency(payload: EmergencyRequest):
    if not payload.emergency_text or not payload.emergency_text.strip():
        raise HTTPException(status_code=400, detail='emergency_text is required')

    results = pipeline.process_emergency(
        payload.emergency_text,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=payload.location,
        timestamp=payload.timestamp,
        context=payload.context,
    )
    safety_rules = evaluate_safety_rules(payload.emergency_text)
    required_skills = results['required_skills']
    skills = [item['skill'] for item in required_skills]
    candidate_responders = [
        {'id': 'r1', 'skill': 'Ambulance', 'available': True, 'distance_km': 1.2, 'eta_min': 4, 'reliability': 0.82, 'experience_years': 6, 'acceptance_probability': 0.88},
        {'id': 'r2', 'skill': 'Fire Fighter', 'available': True, 'distance_km': 3.0, 'eta_min': 8, 'reliability': 0.9, 'experience_years': 9, 'acceptance_probability': 0.91},
        {'id': 'r3', 'skill': 'Police', 'available': True, 'distance_km': 0.7, 'eta_min': 2, 'reliability': 0.76, 'experience_years': 5, 'acceptance_probability': 0.75},
        {'id': 'r4', 'skill': 'Doctor', 'available': True, 'distance_km': 2.5, 'eta_min': 12, 'reliability': 0.88, 'experience_years': 7, 'acceptance_probability': 0.9},
    ]
    ranked = dispatch_engine.rank_responders(candidate_responders, skills)
    dispatch_summary = dispatch_engine.dispatch(candidate_responders, skills, 1.0)
    response = {
        'request_id': str(uuid.uuid4()),
        'category': results['category'],
        'severity': results['severity'],
        'required_skills': required_skills,
        'safety_rules': {
            'triggered': safety_rules['triggered'],
            'overrides': safety_rules['overrides'],
        },
        'responders': ranked,
        'dispatch': {
            'initial_radius_km': 1.0,
            'current_radius_km': dispatch_summary['current_radius_km'],
            'status': dispatch_summary['status'],
        },
        'escalation': {
            'enabled': True,
            'stage': 0,
        },
        'model_versions': {
            'category': 'resq-category-v1.0',
            'severity': 'resq-severity-v1.0',
            'skill': 'resq-skill-v1.0',
            'responder': 'resq-responder-v1.0',
        },
    }
    return response
