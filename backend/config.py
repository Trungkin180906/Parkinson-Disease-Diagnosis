import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PD_DATA_DIR", str(ROOT / ".data"))).resolve()
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PD_MODEL_DIR", str(ROOT / "ai"))).resolve()
    )
    api_key: str | None = field(default_factory=lambda: os.getenv("PD_API_KEY"))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip()
            for s in os.getenv("PD_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
            if s.strip()
        )
    )
    drawing_weight: float = field(default_factory=lambda: float(os.getenv("PD_DRAWING_WEIGHT", "0.5")))
    referral_threshold: float = field(default_factory=lambda: float(os.getenv("PD_REFERRAL_THRESHOLD", "70")))
    monitoring_policy: str = field(
        default_factory=lambda: os.getenv("PD_MONITORING_POLICY", "at_risk_or_confirmed")
    )
    voice_reference_max: float = field(
        default_factory=lambda: float(os.getenv("PD_VOICE_REFERENCE_MAX", "100"))
    )
    voice_change_threshold: float = field(
        default_factory=lambda: float(os.getenv("PD_VOICE_CHANGE_THRESHOLD", "5"))
    )
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("PD_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    )

    def __post_init__(self):
        import math

        for value in (
            self.drawing_weight,
            self.referral_threshold,
            self.voice_reference_max,
            self.voice_change_threshold,
        ):
            if not math.isfinite(value):
                raise ValueError("Numeric settings must be finite")
        if not 0 < self.drawing_weight < 1:
            raise ValueError("PD_DRAWING_WEIGHT must be between 0 and 1, exclusively")
        if not 0 <= self.referral_threshold <= 100:
            raise ValueError("PD_REFERRAL_THRESHOLD must be between 0 and 100")
        if self.monitoring_policy not in {"confirmed_only", "at_risk_or_confirmed"}:
            raise ValueError("Unsupported PD_MONITORING_POLICY")
        if self.voice_reference_max <= 0 or self.voice_change_threshold <= 0 or self.max_upload_bytes <= 0:
            raise ValueError("Voice reference, change threshold and upload limit must be positive")
        if self.api_key is not None and len(self.api_key) < 16:
            raise ValueError("PD_API_KEY must contain at least 16 characters")
