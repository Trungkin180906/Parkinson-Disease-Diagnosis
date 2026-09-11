import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from conftest import KEY, PATIENT

from backend.client import BackendClient, BackendError


def test_live_http_client_swagger_and_error_contract(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        "PD_DATA_DIR": str(tmp_path / "data"),
        "PD_MODEL_DIR": str(tmp_path / "models"),
        "PD_API_KEY": KEY,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=1) as response:
                    assert response.status == 200
                break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError("Uvicorn failed to start")
                time.sleep(0.05)
        with urllib.request.urlopen(base + "/openapi.json") as response:
            assert b"ScreeningOut" in response.read()
        client = BackendClient(KEY, base)
        patient = client.request("POST", "/api/v1/patients", PATIENT)
        assert client.request("GET", "/api/v1/patients")["total"] == 1
        screening = client.request("POST", f"/api/v1/patients/{patient['id']}/screenings", {})
        try:
            client.upload(f"/api/v1/screenings/{screening['id']}/drawing/file", b"no-model")
            raise AssertionError("Missing model must not produce a result")
        except BackendError as exc:
            assert exc.status == 503
            assert exc.payload["error"]["code"] == "model_unavailable"
        assert b"B\xc3\xa1o c\xc3\xa1o" in client.request(
            "GET", f"/api/v1/patients/{patient['id']}/report.html"
        )
    finally:
        process.terminate()
        process.communicate(timeout=10)
