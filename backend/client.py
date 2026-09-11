"""Synchronous client usable from Streamlit without importing models or SQL."""

import json
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4


class BackendError(RuntimeError):
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        super().__init__(payload.get("error", {}).get("message", str(payload)))


class BackendClient:
    def __init__(self, api_key, base_url="http://127.0.0.1:8000", timeout=120):
        self.api_key, self.base_url, self.timeout = api_key, base_url.rstrip("/"), timeout

    def request(self, method, path, payload=None):
        body = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
        )
        return self._send(method, path, body, "application/json")

    def _send(self, method, path, body, content_type):
        if not path.startswith("/api/v1/") or "?" in path.split("/api/v1/", 1)[0]:
            raise ValueError("Use an API path beginning with /api/v1/")
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if "application/json" in response.headers.get("Content-Type", ""):
                    return json.load(response)
                return response.read()
        except urllib.error.HTTPError as exc:
            try:
                payload = json.load(exc)
            except (ValueError, UnicodeError):
                payload = {"error": {"message": "Backend trả về lỗi HTTP."}}
            raise BackendError(exc.code, payload) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(
                503,
                {
                    "error": {
                        "code": "connection_failed",
                        "message": "Không kết nối được backend hoặc yêu cầu đã hết thời gian chờ.",
                    }
                },
            ) from exc

    def upload(self, path, data, recorded_at=None, notes=""):
        boundary = uuid4().hex
        chunks = []
        for key, value in {"recorded_at": recorded_at, "notes": notes}.items():
            if value is not None:
                chunks.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
                )
        chunks.extend(
            [
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="upload.bin"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),
                data,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        return self._send("POST", path, b"".join(chunks), "multipart/form-data; boundary=" + boundary)
