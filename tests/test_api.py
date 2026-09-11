from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from conftest import KEY, PATIENT, completed, confirmed
from fastapi.testclient import TestClient

from backend.app import create_app


def test_auth_validation_and_patient_version(client, patient):
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/patients", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/v1/patients", json={**PATIENT, "full_name": "  "}).status_code == 422
    assert client.post("/api/v1/patients", json={**PATIENT, "age": -1}).status_code == 422
    assert (
        client.post("/api/v1/patients", json={**PATIENT, "diagnosis_status": "confirmed"}).status_code == 422
    )
    url = f"/api/v1/patients/{patient['id']}"
    payload = {**PATIENT, "age": 66, "version": 1}
    assert client.put(url, json=payload).json()["version"] == 2
    assert client.put(url, json=payload).status_code == 409
    assert client.get("/api/v1/patients?q=%25").json()["total"] == 0
    assert client.get("/api/v1/patients?q=" + patient["code"]).json()["total"] == 1
    assert client.get("/api/v1/patients/missing").status_code == 404


def test_fusion_pending_partial_complete_and_immutable(client, patient):
    screening = client.post(f"/api/v1/patients/{patient['id']}/screenings", json={}).json()
    assert screening["status"] == "pending"
    url = f"/api/v1/screenings/{screening['id']}"
    features = {"feature_schema": "test.v1", "values": [1]}
    response = client.post(url + "/drawing/features", json=features)
    assert response.status_code == 201
    assert response.json()["status"] == "partial"
    assert response.json()["overall_risk"] is None
    assert response.json()["missing_modalities"] == ["gait"]
    assert client.post(url + "/drawing/features", json=features).status_code == 409
    result = client.post(url + "/gait/features", json=features).json()
    assert result["overall_risk"] == 79
    assert result["status"] == "completed"
    assert result["risk_level"] == "high"
    assert result["missing_modalities"] == []
    assert client.post(url + "/gait/features", json=features).status_code == 409
    assert client.get(f"/api/v1/patients/{patient['id']}").json()["monitoring_eligibility"]["eligible"]


def test_simultaneous_modalities_complete_once(client, patient):
    screening = client.post(f"/api/v1/patients/{patient['id']}/screenings", json={}).json()

    def send(modality):
        return client.post(
            f"/api/v1/screenings/{screening['id']}/{modality}/features",
            json={"feature_schema": "test.v1", "values": [1]},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(send, ["drawing", "gait"]))
    assert [r.status_code for r in responses] == [201, 201]
    result = client.get(f"/api/v1/screenings/{screening['id']}").json()
    assert result["overall_risk"] == 79
    assert len(result["results"]) == 2


def test_screening_threshold_snapshot(settings, predictor):
    with TestClient(create_app(settings, predictor)) as first:
        first.headers["Authorization"] = "Bearer " + KEY
        patient = first.post("/api/v1/patients", json=PATIENT).json()
        screening = first.post(f"/api/v1/patients/{patient['id']}/screenings", json={}).json()
    changed = replace(settings, drawing_weight=0.8, referral_threshold=95)
    with TestClient(create_app(changed, predictor)) as second:
        second.headers["Authorization"] = "Bearer " + KEY
        for modality in ("drawing", "gait"):
            response = second.post(
                f"/api/v1/screenings/{screening['id']}/{modality}/features",
                json={"feature_schema": "test.v1", "values": [1]},
            )
        assert response.json()["overall_risk"] == 79
        assert response.json()["referral_threshold"] == 70
        assert second.get(f"/api/v1/patients/{patient['id']}").json()["monitoring_eligibility"]["eligible"]


def voice_payload(time="2025-01-01T00:00:00Z"):
    return {"feature_schema": "voice.uci16.v1", "values": [0.1] * 16, "recorded_at": time}


def test_voice_eligibility_trend_backdated_duplicates_and_isolation(client, patient, predictor):
    url = f"/api/v1/patients/{patient['id']}/voice"
    assert client.post(url + "/features", json=voice_payload()).status_code == 409
    confirmed(client, patient)
    predictor.values["voice"] = 30
    assert client.post(url + "/features", json=voice_payload("2025-02-01T00:00:00Z")).status_code == 201
    predictor.values["voice"] = 20
    first = client.post(url + "/features", json=voice_payload()).json()
    assert first["trend"] == "baseline"
    rows = client.get(url).json()["items"]
    assert [r["total_updrs"] for r in rows] == [20, 30]
    assert rows[1]["trend"] == "increased"
    assert rows[1]["delta_updrs"] == 10
    assert rows[1]["voice_score"] == 70
    assert client.post(url + "/features", json=voice_payload()).status_code == 409
    # Equivalent timezone denotes the same instant.
    assert client.post(url + "/features", json=voice_payload("2025-01-01T07:00:00+07:00")).status_code == 409
    assert client.post(url + "/features", json=voice_payload("2025-01-01T00:00:00")).status_code == 422
    assert client.post(url + "/features", json=voice_payload("2099-01-01T00:00:00Z")).status_code == 422
    other = client.post("/api/v1/patients", json={**PATIENT, "full_name": "Người khác"}).json()
    assert client.get(f"/api/v1/patients/{other['id']}/voice").json()["items"] == []
    predictor.version = "new-model"
    new = client.post(url + "/features", json=voice_payload("2025-03-01T00:00:00Z")).json()
    assert new["trend"] == "not_comparable"
    assert new["delta_updrs"] is None


def test_confirmed_only_policy(settings, predictor):
    with TestClient(create_app(replace(settings, monitoring_policy="confirmed_only"), predictor)) as client:
        client.headers["Authorization"] = "Bearer " + KEY
        patient = client.post("/api/v1/patients", json=PATIENT).json()
        completed(client, patient)
        url = f"/api/v1/patients/{patient['id']}/voice/features"
        assert client.post(url, json=voice_payload()).status_code == 409
        patient = confirmed(client, patient)
        assert client.post(url, json=voice_payload()).status_code == 201
        response = client.post(
            f"/api/v1/patients/{patient['id']}/diagnoses",
            json={"status": "ruled_out", "clinician_name": "Dr Test", "version": patient["version"]},
        )
        assert response.status_code == 201
        assert client.post(url, json=voice_payload("2025-02-01T00:00:00Z")).status_code == 409


def test_real_missing_models_no_fake_results(settings):
    with TestClient(create_app(settings)) as client:
        client.headers["Authorization"] = "Bearer " + KEY
        patient = client.post("/api/v1/patients", json=PATIENT).json()
        screening = client.post(f"/api/v1/patients/{patient['id']}/screenings", json={}).json()
        response = client.post(
            f"/api/v1/screenings/{screening['id']}/drawing/features",
            json={"feature_schema": "drawing.hog.v1", "values": [1]},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        result = client.get(f"/api/v1/screenings/{screening['id']}").json()
        assert result["results"] == [] and result["overall_risk"] is None
        assert all(not m["ready"] for m in client.get("/api/v1/models").json().values())


def test_upload_limits_and_nonfinite(client, patient, settings, predictor):
    screening = client.post(f"/api/v1/patients/{patient['id']}/screenings", json={}).json()
    path = f"/api/v1/screenings/{screening['id']}/drawing"
    assert client.post(path + "/file", files={"file": ("empty.png", b"")}).status_code == 422
    assert (
        client.post(
            path + "/features",
            content='{"feature_schema":"x","values":[NaN]}',
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )
    predictor.values["drawing"] = float("nan")
    assert client.post(path + "/features", json={"feature_schema": "x", "values": [1]}).status_code == 503
    with TestClient(create_app(replace(settings, max_upload_bytes=16), predictor)) as limited:
        limited.headers["Authorization"] = "Bearer " + KEY
        assert limited.post(path + "/file", files={"file": ("large.png", b"x" * 17)}).status_code == 413
        assert limited.post(path + "/file", content=b"x" * 70000).status_code == 413


def test_reports_persistence_and_html_escaping(client, patient, settings, predictor):
    client.put(
        f"/api/v1/patients/{patient['id']}",
        json={**PATIENT, "full_name": "<script>alert(1)</script>", "version": 1},
    )
    completed(client, patient)
    client.post(f"/api/v1/patients/{patient['id']}/voice/features", json=voice_payload())
    path = f"/api/v1/patients/{patient['id']}"
    report = client.get(path + "/report").json()
    assert report["screenings"][0]["overall_risk"] == 79
    assert len(report["voice_history"]) == 1
    html = client.get(path + "/report.html")
    assert "<script>" not in html.text and "&lt;script&gt;" in html.text
    assert html.headers["cache-control"] == "no-store"
    assert "recorded_at,total_updrs" in client.get(path + "/voice.csv").text
    with TestClient(create_app(settings, predictor)) as reopened:
        reopened.headers["Authorization"] = "Bearer " + KEY
        assert reopened.get(path + "/report").json()["screenings"][0]["overall_risk"] == 79


def test_generated_key_persists(settings):
    settings = replace(settings, api_key=None)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/patients").status_code == 401
        key = (settings.data_dir / "api-key.txt").read_text()
        assert len(key) >= 32
        assert client.get("/api/v1/patients", headers={"Authorization": "Bearer " + key}).status_code == 200
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/patients", headers={"Authorization": "Bearer " + key}).status_code == 200


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "low"),
        (30, "low"),
        (30.1, "medium"),
        (60, "medium"),
        (60.1, "high"),
        (80, "high"),
        (80.1, "very_high"),
        (100, "very_high"),
    ],
)
def test_risk_boundaries(score, expected):
    from backend.policy import risk_level

    assert risk_level(score) == expected
