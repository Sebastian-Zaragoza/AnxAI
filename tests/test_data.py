from fastapi.testclient import TestClient

from src.deployment.app import app


def test_root_endpoint_removed():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 404


def test_health_endpoint_removed():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 404


def test_swagger_docs_and_openapi_available():
    client = TestClient(app)

    docs_response = client.get("/docs")
    schema_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert "paths" in schema
    assert "/predictions" in schema["paths"]
    assert "/" not in schema["paths"]
    assert "/health" not in schema["paths"]