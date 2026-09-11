import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.inference import Prediction

KEY = "backend-integration-test-key"
PATIENT = {"full_name": "Bệnh nhân kiểm thử", "age": 65, "sex": "female", "height_cm": 160, "weight_kg": 55}


class StubPredictor:
    """Deterministic test double, never selected by application configuration."""

    def __init__(self):
        self.values = {"drawing": 82.0, "gait": 76.0, "voice": 27.6}
        self.version = "test-model-v1"

    def predict_features(self, modality, schema, values):
        return Prediction(self.values[modality], self.version, schema)

    def predict_file(self, modality, data):
        return self.predict_features(modality, "test.file.v1", [])

    def status(self):
        return {"test_double": True}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", api_key=KEY)


@pytest.fixture
def predictor():
    return StubPredictor()


@pytest.fixture
def client(settings, predictor):
    with TestClient(create_app(settings, predictor)) as connection:
        connection.headers["Authorization"] = "Bearer " + KEY
        yield connection


@pytest.fixture
def patient(client):
    response = client.post("/api/v1/patients", json=PATIENT)
    assert response.status_code == 201, response.text
    return response.json()


def confirmed(client, patient):
    response = client.post(
        f"/api/v1/patients/{patient['id']}/diagnoses",
        json={
            "status": "confirmed",
            "clinician_name": "Bác sĩ kiểm thử",
            "version": patient["version"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def completed(client, patient):
    response = client.post(f"/api/v1/patients/{patient['id']}/screenings", json={})
    assert response.status_code == 201, response.text
    screening = response.json()
    for modality in ("drawing", "gait"):
        response = client.post(
            f"/api/v1/screenings/{screening['id']}/{modality}/features",
            json={
                "feature_schema": "test.v1",
                "values": [1],
            },
        )
        assert response.status_code == 201, response.text
    return response.json()
