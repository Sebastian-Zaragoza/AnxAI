from fastapi.testclient import TestClient
from src.deployment import app as app_module


class DummyModel:
    def __init__(self, predicted_value: int):
        self.predicted_value = predicted_value

    def predict(self, _):
        return [self.predicted_value]


def docs_headers():
    return {"referer": "http://testserver/docs"}


def test_prediction_endpoint_returns_low_anxiety(monkeypatch):
    monkeypatch.setattr(app_module, "model_pipeline", DummyModel(predicted_value=1))
    client = TestClient(app_module.app)

    payload = {
        "daily_screen_time_min": 150.0,
        "notification_count": 120,
        "social_media_time_min": 90.0,
    }
    response = client.post("/predictions", json=payload, headers=docs_headers())

    assert response.status_code == 200
    assert response.json()["prediction"] == "Your anxiety levels are low"


def test_prediction_endpoint_returns_high_anxiety(monkeypatch):
    monkeypatch.setattr(app_module, "model_pipeline", DummyModel(predicted_value=0))
    client = TestClient(app_module.app)

    payload = {
        "daily_screen_time_min": 240.0,
        "notification_count": 320,
        "social_media_time_min": 180.0,
    }
    response = client.post("/predictions", json=payload, headers=docs_headers())

    assert response.status_code == 200
    assert response.json()["prediction"] == "Your anxiety levels are high"


def test_prediction_returns_503_if_model_not_loaded(monkeypatch):
    monkeypatch.setattr(app_module, "model_pipeline", None)
    client = TestClient(app_module.app)

    payload = {
        "daily_screen_time_min": 140.0,
        "notification_count": 90,
        "social_media_time_min": 70.0,
    }
    response = client.post("/predictions", json=payload, headers=docs_headers())

    assert response.status_code == 503
    assert "Model unavailable" in response.json()["detail"]


def test_prediction_rejected_without_docs_referer(monkeypatch):
    monkeypatch.setattr(app_module, "model_pipeline", DummyModel(predicted_value=1))
    client = TestClient(app_module.app)

    payload = {
        "daily_screen_time_min": 140.0,
        "notification_count": 90,
        "social_media_time_min": 70.0,
    }
    response = client.post("/predictions", json=payload)

    assert response.status_code == 403
    assert "Use Swagger Docs at /docs" in response.json()["detail"]