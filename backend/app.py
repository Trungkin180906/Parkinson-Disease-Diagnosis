import csv
import hmac
import io
import logging
import secrets
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from html import escape
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from backend.config import Settings
from backend.db import Database
from backend.errors import DomainError
from backend.inference import ModelRegistry
from backend.outputs import PatientList, PatientOut, ReportOut, ScreeningOut, VoiceHistory, VoiceOut
from backend.schemas import (
    DiagnosisInput,
    FeatureInput,
    Modality,
    PatientCreate,
    PatientUpdate,
    ScreeningCreate,
    VoiceInput,
)
from backend.service import Service

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)


class BodyLimitMiddleware:
    """Limit chunked requests as well as Content-Length before multipart parsing."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                response = JSONResponse(
                    {"error": {"code": "request_too_large", "message": "Dữ liệu gửi lên vượt giới hạn."}},
                    status_code=413,
                )
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            return messages.pop(0) if messages else await receive()

        return await self.app(scope, replay, send)


def local_key(settings):
    if settings.api_key:
        return settings.api_key
    path = settings.data_dir / "api-key.txt"
    try:
        with path.open("x", encoding="utf-8") as handle:
            key = secrets.token_urlsafe(32)
            handle.write(key)
        path.chmod(0o600)
    except FileExistsError:
        key = path.read_text(encoding="utf-8").strip()
    if len(key) < 16:
        raise RuntimeError("Invalid API key file; configure PD_API_KEY")
    return key


def create_app(settings=None, predictor=None):
    settings = settings or Settings()
    db = Database(settings.data_dir / "parkinson.sqlite3")
    registry = predictor if predictor is not None else ModelRegistry(settings.model_dir)
    service = Service(db, settings, registry)

    @asynccontextmanager
    async def lifespan(app):
        db.initialize()
        app.state.api_key = local_key(settings)
        yield

    app = FastAPI(
        title="Parkinson Integration API",
        version="1.0.0",
        lifespan=lifespan,
        description="Hồ sơ, sàng lọc Drawing/Gait, theo dõi Voice và báo cáo. Các endpoint /api/v1 dùng Bearer API key.",
    )
    app.state.service = service
    app.state.settings = settings
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + 64 * 1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(DomainError)
    async def domain_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(
            {
                "error": {
                    "code": "validation_error",
                    "message": "Dữ liệu đầu vào không hợp lệ.",
                    "details": details,
                }
            },
            status_code=422,
        )

    @app.exception_handler(sqlite3.OperationalError)
    async def database_error(request, exc):
        logger.error("Database operation failed: %s", type(exc).__name__)
        return JSONResponse(
            {"error": {"code": "database_unavailable", "message": "Cơ sở dữ liệu tạm thời không sẵn sàng."}},
            status_code=503,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error("Unhandled backend error: %s", type(exc).__name__)
        return JSONResponse(
            {"error": {"code": "internal_error", "message": "Không thể hoàn tất yêu cầu."}}, status_code=500
        )

    def authorize(
        request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
    ):
        if credentials is None or not hmac.compare_digest(credentials.credentials, request.app.state.api_key):
            raise DomainError(401, "unauthorized", "Thiếu hoặc sai Bearer API key.")

    router = APIRouter(prefix="/api/v1", dependencies=[Depends(authorize)])

    @app.get("/health", tags=["system"])
    def health():
        with db.transaction() as conn:
            conn.execute("SELECT 1").fetchone()
        return {"status": "ok", "version": "1.0.0"}

    @router.get("/models", tags=["system"])
    def models():
        return registry.status()

    @router.get("/config", tags=["system"])
    def configuration():
        return {
            "drawing_weight": settings.drawing_weight,
            "gait_weight": 1 - settings.drawing_weight,
            "referral_threshold": settings.referral_threshold,
            "monitoring_policy": settings.monitoring_policy,
            "voice_reference_max": settings.voice_reference_max,
            "voice_change_threshold": settings.voice_change_threshold,
            "voice_score_formula": "clamp(100 * (1 - total_updrs / voice_reference_max), 0, 100)",
            "voice_score_meaning": "Chỉ số hiển thị thử nghiệm của dự án, không phải thang đo lâm sàng.",
            "risk_bands": [
                {"max_inclusive": 30, "label": "low"},
                {"max_inclusive": 60, "label": "medium"},
                {"max_inclusive": 80, "label": "high"},
                {"max_inclusive": 100, "label": "very_high"},
            ],
            "max_upload_bytes": settings.max_upload_bytes,
        }

    @router.post("/patients", status_code=201, tags=["patients"], response_model=PatientOut)
    def create_patient(payload: PatientCreate):
        return service.create_patient(payload)

    @router.get("/patients", tags=["patients"], response_model=PatientList)
    def patients(
        q: str = Query(default="", max_length=150),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        return service.list_patients(q, limit, offset)

    @router.get("/patients/{patient_id}", tags=["patients"], response_model=PatientOut)
    def patient(patient_id: str):
        return service.patient(patient_id)

    @router.put("/patients/{patient_id}", tags=["patients"], response_model=PatientOut)
    def update_patient(patient_id: str, payload: PatientUpdate):
        return service.update_patient(patient_id, payload)

    @router.post(
        "/patients/{patient_id}/diagnoses", status_code=201, tags=["patients"], response_model=PatientOut
    )
    def diagnosis(patient_id: str, payload: DiagnosisInput):
        return service.record_diagnosis(patient_id, payload)

    @router.post(
        "/patients/{patient_id}/screenings", status_code=201, tags=["screenings"], response_model=ScreeningOut
    )
    def create_screening(patient_id: str, payload: ScreeningCreate):
        return service.create_screening(patient_id, payload)

    @router.get("/patients/{patient_id}/screenings", tags=["screenings"], response_model=list[ScreeningOut])
    def patient_screenings(patient_id: str):
        return service.list_screenings(patient_id)

    @router.get("/screenings/{screening_id}", tags=["screenings"], response_model=ScreeningOut)
    def screening(screening_id: str):
        return service.screening(screening_id)

    @router.post(
        "/screenings/{screening_id}/{modality}/features",
        status_code=201,
        tags=["screenings"],
        response_model=ScreeningOut,
    )
    def screening_features(screening_id: str, modality: Modality, payload: FeatureInput):
        return service.add_screening_result(screening_id, modality.value, features=payload)

    def upload_bytes(file):
        try:
            data = file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise DomainError(413, "file_too_large", "File vượt giới hạn tải lên.")
            if not data:
                raise DomainError(422, "empty_file", "File tải lên rỗng.")
            return data
        finally:
            file.file.close()

    @router.post(
        "/screenings/{screening_id}/{modality}/file",
        status_code=201,
        tags=["screenings"],
        response_model=ScreeningOut,
    )
    def screening_file(screening_id: str, modality: Modality, file: Annotated[UploadFile, File()]):
        return service.add_screening_result(screening_id, modality.value, data=upload_bytes(file))

    @router.post(
        "/patients/{patient_id}/voice/features", status_code=201, tags=["voice"], response_model=VoiceOut
    )
    def voice_features(patient_id: str, payload: VoiceInput):
        return service.add_voice(patient_id, payload.recorded_at, payload.notes, features=payload)

    @router.post(
        "/patients/{patient_id}/voice/file", status_code=201, tags=["voice"], response_model=VoiceOut
    )
    def voice_file(
        patient_id: str,
        file: Annotated[UploadFile, File()],
        recorded_at: Annotated[datetime, Form()],
        notes: Annotated[str, Form(max_length=2000)] = "",
    ):
        try:
            metadata = VoiceInput(
                feature_schema="voice.uci16.v1", values=[0], recorded_at=recorded_at, notes=notes
            )
        except ValidationError as exc:
            raise DomainError(
                422, "invalid_recording_time", "Thời điểm ghi âm cần có múi giờ và không được ở tương lai."
            ) from exc
        return service.add_voice(patient_id, metadata.recorded_at, metadata.notes, data=upload_bytes(file))

    @router.get("/patients/{patient_id}/voice", tags=["voice"], response_model=VoiceHistory)
    def history(patient_id: str):
        rows = service.history(patient_id)
        return {"items": rows, "total": len(rows)}

    @router.get("/patients/{patient_id}/voice.csv", tags=["reports"])
    def history_csv(patient_id: str):
        rows = service.history(patient_id)
        stream = io.StringIO(newline="")
        columns = ["recorded_at", "total_updrs", "voice_score", "delta_updrs", "delta_voice_score", "trend"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            stream.getvalue().encode("utf-8-sig"),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="voice-history.csv"'},
        )

    @router.get("/patients/{patient_id}/report", tags=["reports"], response_model=ReportOut)
    def report(patient_id: str):
        return service.report(patient_id)

    @router.get("/patients/{patient_id}/report.html", response_class=HTMLResponse, tags=["reports"])
    def report_html(patient_id: str):
        report = service.report(patient_id)
        patient = report["patient"]

        def table(columns, rows):
            header = "".join("<th>" + escape(label) + "</th>" for _, label in columns)
            body = "".join(
                "<tr>"
                + "".join(
                    "<td>" + escape(str(row.get(key) if row.get(key) is not None else "—")) + "</td>"
                    for key, _ in columns
                )
                + "</tr>"
                for row in rows
            )
            return "<table><thead><tr>" + header + "</tr></thead><tbody>" + body + "</tbody></table>"

        screening_rows = []
        for screening in report["screenings"]:
            screening_rows.append(
                {**screening, **{r["modality"]: round(r["risk_score"], 2) for r in screening["results"]}}
            )
        content = "<!doctype html><html lang='vi'><meta charset='utf-8'><title>Báo cáo Parkinson</title>"
        content += "<style>body{font:16px sans-serif;margin:40px;line-height:1.5}table{border-collapse:collapse;width:100%;margin-bottom:24px}th,td{border:1px solid #ccc;padding:8px;text-align:left}th{background:#eee}@media print{body{margin:0}tr{break-inside:avoid}}</style>"
        content += "<h1>Báo cáo sàng lọc và theo dõi Parkinson</h1><p>" + escape(report["notice"]) + "</p>"
        content += "<p>Ngày xuất: " + escape(report["generated_at"]) + "</p><h2>Thông tin bệnh nhân</h2>"
        content += table(
            [
                ("code", "Mã"),
                ("full_name", "Họ tên"),
                ("age", "Tuổi"),
                ("sex", "Giới tính"),
                ("height_cm", "Chiều cao cm"),
                ("weight_kg", "Cân nặng kg"),
                ("phone", "Điện thoại"),
                ("diagnosis_status", "Trạng thái chẩn đoán"),
            ],
            [patient],
        )
        content += "<h2>Sàng lọc</h2>" + table(
            [
                ("performed_on", "Ngày"),
                ("status", "Trạng thái"),
                ("drawing", "Drawing %"),
                ("gait", "Gait %"),
                ("overall_risk", "Overall %"),
                ("risk_level", "Mức nguy cơ"),
                ("recommendation", "Khuyến nghị"),
            ],
            screening_rows,
        )
        content += "<h2>Theo dõi giọng nói</h2>" + table(
            [
                ("recorded_at", "Thời điểm"),
                ("total_updrs", "UPDRS dự đoán"),
                ("voice_score", "Voice Score"),
                ("delta_updrs", "Thay đổi UPDRS"),
                ("trend", "Xu hướng"),
            ],
            report["voice_history"],
        )
        content += "<h2>Ghi nhận từ bác sĩ</h2>" + table(
            [
                ("assessed_on", "Ngày đánh giá"),
                ("clinician_name", "Người đánh giá"),
                ("status", "Trạng thái"),
                ("notes", "Ghi chú"),
            ],
            report["diagnoses"],
        )
        return HTMLResponse(
            content + "</html>",
            headers={
                "Content-Disposition": 'attachment; filename="patient-report.html"',
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            },
        )

    app.include_router(router)

    @app.middleware("http")
    async def no_patient_caching(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()
