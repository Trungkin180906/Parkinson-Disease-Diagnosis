"""Response contracts used by OpenAPI and frontend clients."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

from backend.schemas import PatientCreate


class Eligibility(BaseModel):
    eligible: bool
    basis: str | None
    policy: Literal["confirmed_only", "at_risk_or_confirmed"]


class PatientOut(PatientCreate):
    id: str
    code: str
    diagnosis_status: Literal["unknown", "confirmed", "ruled_out"]
    version: int
    created_at: datetime
    updated_at: datetime
    monitoring_eligibility: Eligibility | None = None


class PatientList(BaseModel):
    items: list[PatientOut]
    total: int
    limit: int
    offset: int


class ScreeningResult(BaseModel):
    id: str
    screening_id: str
    modality: Literal["drawing", "gait"]
    risk_score: float
    model_version: str
    feature_schema: str
    input_kind: Literal["file", "features"]
    input_sha256: str
    created_at: datetime


class ScreeningOut(BaseModel):
    id: str
    patient_id: str
    performed_on: date
    notes: str
    status: Literal["pending", "partial", "completed"]
    created_at: datetime
    drawing_weight: float
    referral_threshold: float
    overall_risk: float | None
    risk_level: Literal["low", "medium", "high", "very_high"] | None
    recommendation: str | None
    completed_at: datetime | None
    results: list[ScreeningResult]
    missing_modalities: list[Literal["drawing", "gait"]]
    notice: str


class VoiceOut(BaseModel):
    id: str
    patient_id: str
    recorded_at: datetime
    notes: str
    total_updrs: float
    voice_score: float
    reference_max: float
    change_threshold: float
    model_version: str
    feature_schema: str
    input_kind: Literal["file", "features"]
    input_sha256: str
    eligibility_basis: str
    created_at: datetime
    delta_updrs: float | None
    delta_voice_score: float | None
    trend: Literal["baseline", "stable", "increased", "decreased", "not_comparable"]
    comparison_record_id: str | None


class VoiceHistory(BaseModel):
    items: list[VoiceOut]
    total: int


class DiagnosisOut(BaseModel):
    id: str
    patient_id: str
    status: Literal["unknown", "confirmed", "ruled_out"]
    clinician_name: str
    assessed_on: date
    notes: str
    created_at: datetime


class ReportOut(BaseModel):
    generated_at: datetime
    patient: PatientOut
    diagnoses: list[DiagnosisOut]
    screenings: list[ScreeningOut]
    voice_history: list[VoiceOut]
    monitoring_eligibility: Eligibility
    notice: str
