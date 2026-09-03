from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_get_predictions():

    response = client.get(
        "/predictions?limit=10"
    )

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "predictions" in data

    assert isinstance(
        data["predictions"],
        list
    )


def test_get_incident_predictions():

    response = client.get(
        "/predictions?limit=10&incident_only=true"
    )

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "predictions" in data

    for prediction in data["predictions"]:

        assert prediction["prediction"] == 1

        assert (
            prediction["status"]
            == "INCIDENT RISK"
        )


def test_get_statistics():

    response = client.get(
        "/statistics"
    )

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "incident_predictions" in data
    assert "normal_predictions" in data
    assert "incident_rate" in data
    assert (
        "average_incident_probability"
        in data
    )

    assert (
        data["total_predictions"]
        >= 0
    )