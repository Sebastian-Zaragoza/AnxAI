from fastapi.testclient import TestClient

from src.deployment.app import app


def test_root_endpoint_available():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "AnxAI API is running" in response.json()["message"]


def test_health_endpoint_available():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload
    assert payload["docs_url"] == "/docs"


def test_swagger_docs_and_openapi_available():
    client = TestClient(app)

    docs_response = client.get("/docs")
    schema_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert schema_response.status_code == 200
    assert "paths" in schema_response.json()