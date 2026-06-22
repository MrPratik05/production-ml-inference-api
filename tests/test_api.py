from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "model_name" in body


def test_prediction_endpoint():
    response = client.post("/predict", json={"text": "I love this project"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "score" in body
    assert "latency_ms" in body


def test_prediction_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
