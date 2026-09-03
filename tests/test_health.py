from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_available"] is True
    assert data["metadata_available"] is True
    assert data["database_available"] is True