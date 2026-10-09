import os

from fastapi.testclient import TestClient

from api.main import app
from src.dispatch.dispatch_engine import DispatchEngine
from src.preprocessing.pipeline import preprocess_text
from src.safety.safety_rules import evaluate_safety_rules


def test_preprocess_text_normalizes_input():
    result = preprocess_text(" FIRE!!! Medical emergency near school ")
    assert isinstance(result, str)
    assert "fire" in result.lower()


def test_dispatch_engine_rank_and_expand():
    engine = DispatchEngine({
        'initial_radius_km': 1,
        'second_radius_km': 5,
        'third_radius_km': 10,
        'max_radius_km': 10,
        'max_notifications': 2,
        'escalation_threshold': 1,
    })
    responders = [
        {'id': 'r1', 'skill': 'Ambulance', 'available': True, 'distance_km': 0.5, 'eta_min': 3, 'reliability': 0.8, 'experience_years': 5, 'acceptance_probability': 0.9},
        {'id': 'r2', 'skill': 'Doctor', 'available': True, 'distance_km': 2.0, 'eta_min': 10, 'reliability': 0.7, 'experience_years': 2, 'acceptance_probability': 0.7},
    ]
    result = engine.rank_responders(responders, ['Ambulance'])
    assert len(result) == 2
    assert result[0]['score'] >= result[1]['score']
    assert engine.expand_radius(1) == 5


def test_safety_rule_trigger():
    decision = evaluate_safety_rules('fire in a warehouse with smoke')
    assert decision['triggered']
    assert 'Fire Fighter' in decision['overrides'][0]['action']


def test_health_endpoint():
    client = TestClient(app)
    response = client.get('/api/ai/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
