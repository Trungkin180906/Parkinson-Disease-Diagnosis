import io
import json
import wave
from pathlib import Path

import joblib
import numpy as np
import pytest
from conftest import KEY, PATIENT, confirmed
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from backend.app import create_app
from backend.errors import DomainError
from backend.inference import ModelRegistry, gait_features, image_hog, validate_wav


def model_fixture(directory, modality, schema, dimension):
    """Real sklearn models fitted to synthetic test data, only under tmp_path."""
    directory = directory / modality
    directory.mkdir(parents=True)
    x = np.random.default_rng(1).normal(size=(20, dimension))
    y = np.arange(20) % 2 if modality != "voice" else np.arange(20) + 10
    scaler = StandardScaler().fit(x)
    model_type = RandomForestRegressor if modality == "voice" else RandomForestClassifier
    model = model_type(n_estimators=3, random_state=1).fit(scaler.transform(x), y)
    joblib.dump(model, directory / "model.pkl")
    joblib.dump(scaler, directory / "scaler.pkl")
    manifest = {
        "version": "synthetic-test-only",
        "feature_schema": schema,
        "model_file": "model.pkl",
        "scaler_file": "scaler.pkl",
    }
    (directory / "backend-model.json").write_text(json.dumps(manifest))
    return manifest


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)
    return buffer.getvalue()


def test_gait_correct_columns_and_invalid_data():
    matrix = np.zeros((10, 19))
    matrix[:, 0] = np.arange(10) / 100
    matrix[:, 16] = 999
    matrix[:, 17] = np.arange(10) + 100
    matrix[:, 18] = np.arange(10) + 200
    buffer = io.StringIO()
    np.savetxt(buffer, matrix, delimiter="\t")
    result = gait_features(buffer.getvalue().encode())
    assert result[0] == 104.5 and result[1] == 204.5 and result[4] == 100
    assert result[2] == pytest.approx(np.std(np.arange(10), ddof=1))
    for data in [b"video-not-supported", b"1 2 3\n", b"\xff\xfe"]:
        with pytest.raises(DomainError) as exc:
            gait_features(data)
        assert exc.value.status == 422


def test_hog_matches_existing_pipeline_and_rejects_bad_image():
    import cv2

    from ai.drawing.predict import extract_hog_from_image

    rgb = np.random.default_rng(1).integers(0, 255, size=(150, 150, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    result = image_hog(buffer.getvalue())
    assert len(result) == 8100
    np.testing.assert_allclose(result, extract_hog_from_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
    with pytest.raises(DomainError):
        image_hog(b"not-an-image")


def test_registry_schema_dimension_output_and_version(settings):
    model_fixture(settings.model_dir, "drawing", "drawing.handpd.v1", 9)
    registry = ModelRegistry(settings.model_dir)
    result = registry.predict_features("drawing", "drawing.handpd.v1", [1] * 9)
    assert 0 <= result.value <= 100
    assert result.model_version.startswith("synthetic-test-only:")
    assert registry.status()["drawing"]["file_input"] is False
    with pytest.raises(DomainError) as exc:
        registry.predict_file("drawing", b"any image")
    assert exc.value.code == "unsupported_model_input"
    for schema, values in [
        ("drawing.hog.v1", [1] * 9),
        ("drawing.handpd.v1", [1] * 8),
        ("drawing.handpd.v1", [float("inf")] * 9),
    ]:
        with pytest.raises(DomainError) as exc:
            registry.predict_features("drawing", schema, values)
        assert exc.value.status == 422
    manifest_path = settings.model_dir / "drawing/backend-model.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_schema"] = "drawing.hog.v1"
    manifest_path.write_text(json.dumps(manifest))
    assert registry.status()["drawing"]["code"] == "model_incompatible"


def test_raw_wav_requires_real_extractor(settings):
    model_fixture(settings.model_dir, "voice", "voice.uci16.v1", 16)
    registry = ModelRegistry(settings.model_dir)
    assert registry.predict_features("voice", "voice.uci16.v1", [0.1] * 16).value >= 0
    validate_wav(wav_bytes())
    with pytest.raises(DomainError) as exc:
        registry.predict_file("voice", wav_bytes())
    assert exc.value.code == "extractor_unavailable"
    for data in [b"not a wav", wav_bytes()[:-10]]:
        with pytest.raises(DomainError):
            validate_wav(data)


def test_real_sklearn_http_drawing_gait_voice_and_report(settings):
    model_fixture(settings.model_dir, "drawing", "drawing.hog.v1", 8100)
    model_fixture(settings.model_dir, "gait", "gait.vgrf.v1", 5)
    model_fixture(settings.model_dir, "voice", "voice.uci16.v1", 16)
    with TestClient(create_app(settings)) as client:
        client.headers["Authorization"] = "Bearer " + KEY
        patient = client.post("/api/v1/patients", json=PATIENT).json()
        confirmed(client, patient)
        screening = client.post(f"/api/v1/patients/{patient['id']}/screenings", json={}).json()
        buffer = io.BytesIO()
        Image.new("RGB", (128, 128), "white").save(buffer, format="PNG")
        drawing = client.post(
            f"/api/v1/screenings/{screening['id']}/drawing/file",
            files={"file": ("test.png", buffer.getvalue())},
        )
        assert drawing.status_code == 201, drawing.text
        gait_path = Path(__file__).resolve().parents[1] / "dataset/gait/control/GaCo01_01.txt"
        gait = client.post(
            f"/api/v1/screenings/{screening['id']}/gait/file",
            files={"file": ("test.txt", gait_path.read_bytes())},
        )
        assert gait.status_code == 201, gait.text
        assert gait.json()["status"] == "completed"
        voice = client.post(
            f"/api/v1/patients/{patient['id']}/voice/features",
            json={
                "recorded_at": "2025-01-01T00:00:00Z",
                "feature_schema": "voice.uci16.v1",
                "values": [0.1] * 16,
            },
        )
        assert voice.status_code == 201, voice.text
        assert voice.json()["total_updrs"] >= 0
        report = client.get(f"/api/v1/patients/{patient['id']}/report").json()
        assert len(report["screenings"][0]["results"]) == 2
        assert len(report["voice_history"]) == 1


def test_wav_extractor_contract_and_http_metadata(settings, monkeypatch):
    import sys
    import types

    from backend.inference import VOICE_FEATURES

    manifest = model_fixture(settings.model_dir, "voice", "voice.uci16.v1", 16)
    module = types.ModuleType("test_voice_extractor")
    # Verify adapter plumbing independently of the team's acoustic algorithms.
    module.extract = lambda data: dict.fromkeys(VOICE_FEATURES, 0.1)
    monkeypatch.setitem(sys.modules, "test_voice_extractor", module)
    manifest["wav_extractor"] = "test_voice_extractor:extract"
    (settings.model_dir / "voice/backend-model.json").write_text(json.dumps(manifest))
    with TestClient(create_app(settings)) as client:
        client.headers["Authorization"] = "Bearer " + KEY
        patient = client.post("/api/v1/patients", json=PATIENT).json()
        confirmed(client, patient)
        url = f"/api/v1/patients/{patient['id']}/voice/file"
        response = client.post(
            url, files={"file": ("voice.wav", wav_bytes())}, data={"recorded_at": "2025-01-01T07:00:00+07:00"}
        )
        assert response.status_code == 201, response.text
        assert response.json()["recorded_at"].startswith("2025-01-01T00:00:00")
        assert response.json()["input_kind"] == "file"
        invalid = client.post(
            url, files={"file": ("voice.wav", wav_bytes())}, data={"recorded_at": "2025-01-02T07:00:00"}
        )
        assert invalid.status_code == 422
