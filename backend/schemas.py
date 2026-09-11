from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator


def today():
    return date.today()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PatientCreate(Input):
    full_name: str = Field(min_length=1, max_length=150)
    age: int = Field(ge=0, le=120, strict=True)
    sex: Literal["male", "female", "other", "unspecified"]
    height_cm: Annotated[FiniteFloat, Field(gt=0, le=300)]
    weight_kg: Annotated[FiniteFloat, Field(gt=0, le=500)]
    phone: str | None = Field(default=None, pattern=r"^\+?[0-9 ()\-]{6,25}$")


class PatientUpdate(PatientCreate):
    version: int = Field(ge=1)


class DiagnosisInput(Input):
    status: Literal["unknown", "confirmed", "ruled_out"]
    clinician_name: str = Field(min_length=1, max_length=150)
    assessed_on: date = Field(default_factory=today)
    notes: str = Field(default="", max_length=2000)
    version: int = Field(ge=1)

    @field_validator("assessed_on")
    @classmethod
    def not_future(cls, value):
        if value > today():
            raise ValueError("Assessment date cannot be in the future")
        return value


class ScreeningCreate(Input):
    performed_on: date = Field(default_factory=today)
    notes: str = Field(default="", max_length=2000)

    @field_validator("performed_on")
    @classmethod
    def not_future(cls, value):
        if value > today():
            raise ValueError("Screening date cannot be in the future")
        return value


class Modality(str, Enum):
    drawing = "drawing"
    gait = "gait"


class FeatureInput(Input):
    feature_schema: str = Field(min_length=1, max_length=100)
    values: list[FiniteFloat] = Field(min_length=1, max_length=10000)


class VoiceInput(FeatureInput):
    recorded_at: datetime
    notes: str = Field(default="", max_length=2000)

    @field_validator("recorded_at")
    @classmethod
    def valid_recording_time(cls, value):
        if value.tzinfo is None:
            raise ValueError("recorded_at must include a timezone")
        value = value.astimezone(timezone.utc)
        if value > datetime.now(timezone.utc):
            raise ValueError("recorded_at cannot be in the future")
        return value
