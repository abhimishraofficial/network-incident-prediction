from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_incident_prediction():

    payload = {
        "metrics_history": [
            {
                "cpu_usage": 45,
                "memory_usage": 50,
                "latency_ms": 30,
                "packet_loss": 0.1,
                "throughput_mbps": 150
            },
            {
                "cpu_usage": 60,
                "memory_usage": 65,
                "latency_ms": 55,
                "packet_loss": 0.5,
                "throughput_mbps": 120
            },
            {
                "cpu_usage": 95,
                "memory_usage": 92,
                "latency_ms": 350,
                "packet_loss": 8,
                "throughput_mbps": 20
            }
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "incident_probability" in data
    assert data["production_threshold"] == 0.05
    assert data["prediction"] == 1
    assert data["status"] == "INCIDENT RISK"


def test_normal_prediction():

    payload = {
        "metrics_history": [
            {
                "cpu_usage": 35,
                "memory_usage": 40,
                "latency_ms": 20,
                "packet_loss": 0.05,
                "throughput_mbps": 200
            },
            {
                "cpu_usage": 38,
                "memory_usage": 45,
                "latency_ms": 22,
                "packet_loss": 0.08,
                "throughput_mbps": 195
            },
            {
                "cpu_usage": 40,
                "memory_usage": 50,
                "latency_ms": 25,
                "packet_loss": 0.1,
                "throughput_mbps": 190
            }
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["production_threshold"] == 0.05
    assert data["prediction"] == 0
    assert data["status"] == "NORMAL"


def test_prediction_requires_three_readings():

    payload = {
        "metrics_history": [
            {
                "cpu_usage": 45,
                "memory_usage": 50,
                "latency_ms": 30,
                "packet_loss": 0.1,
                "throughput_mbps": 150
            },
            {
                "cpu_usage": 60,
                "memory_usage": 65,
                "latency_ms": 55,
                "packet_loss": 0.5,
                "throughput_mbps": 120
            }
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_invalid_cpu_usage():

    payload = {
        "metrics_history": [
            {
                "cpu_usage": 45,
                "memory_usage": 50,
                "latency_ms": 30,
                "packet_loss": 0.1,
                "throughput_mbps": 150
            },
            {
                "cpu_usage": 60,
                "memory_usage": 65,
                "latency_ms": 55,
                "packet_loss": 0.5,
                "throughput_mbps": 120
            },
            {
                "cpu_usage": 150,
                "memory_usage": 92,
                "latency_ms": 350,
                "packet_loss": 8,
                "throughput_mbps": 20
            }
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422