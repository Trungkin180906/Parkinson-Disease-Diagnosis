"""Trusted, versioned model adapters. This module never imports a Streamlit page."""

import hashlib
import importlib
import io
import logging
import math
import threading
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.errors import DomainError

SCHEMAS = {
    "drawing.hog.v1": ("drawing", 8100),
    "drawing.handpd.v1": ("drawing", 9),
    "gait.vgrf.v1": ("gait", 5),
    "voice.uci16.v1": ("voice", 16),
}
VOICE_FEATURES = [
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
]
logger = logging.getLogger(__name__)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1, max_length=100)
    feature_schema: Literal["drawing.hog.v1", "drawing.handpd.v1", "gait.vgrf.v1", "voice.uci16.v1"]
    model_file: str
    scaler_file: str
    positive_class: int = 1
    # Server configuration only: a trusted function accepting WAV bytes and
    # returning {feature-name: finite-number} using the training definitions.
    wav_extractor: str | None = None


@dataclass(frozen=True)
class Prediction:
    value: float
    model_version: str
    feature_schema: str


def image_hog(data: bytes):
    import cv2
    from PIL import Image, UnidentifiedImageError
    from skimage.feature import hog

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG", "BMP"} or image.width * image.height > 16_000_000:
                    raise ValueError("Use a JPEG, PNG or BMP image with at most 16 million pixels")
                pixels = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))
        return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    except (
        ValueError,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise DomainError(422, "invalid_image", "Không đọc được ảnh JPEG/PNG/BMP hợp lệ.") from exc


def gait_features(data: bytes):
    try:
        matrix = np.loadtxt(io.StringIO(data.decode("utf-8-sig")), ndmin=2)
        if matrix.shape[1] != 19 or matrix.shape[0] < 10:
            raise ValueError("Expected at least 10 rows and exactly 19 columns")
        if not np.isfinite(matrix).all() or np.any(np.diff(matrix[:, 0]) <= 0) or np.any(matrix[:, 1:] < 0):
            raise ValueError("Non-finite, negative force or non-increasing timestamps")
        # format.txt columns 18/19, zero-based indices 17/18.
        left, right = matrix[:, 17], matrix[:, 18]
        return [
            left.mean(),
            right.mean(),
            left.std(ddof=1),
            right.std(ddof=1),
            abs(left.mean() - right.mean()),
        ]
    except (UnicodeError, ValueError) as exc:
        raise DomainError(
            422, "invalid_gait_data", "Dữ liệu Gait cần 19 cột số, ít nhất 10 dòng và thời gian tăng dần."
        ) from exc


def validate_wav(data: bytes):
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels, width, rate, frames, compression, _ = audio.getparams()
            if compression != "NONE" or channels not in (1, 2) or width not in (1, 2, 3, 4):
                raise ValueError("Unsupported PCM format")
            if not 8000 <= rate <= 96000 or not 1 <= frames / rate <= 120:
                raise ValueError("Expected 1–120 seconds and 8–96 kHz")
            if len(audio.readframes(frames)) != frames * width * channels:
                raise ValueError("Truncated audio")
    except (wave.Error, EOFError, ValueError) as exc:
        raise DomainError(
            422, "invalid_wav", "Cần file WAV PCM hợp lệ, dài 1–120 giây, tần số lấy mẫu 8–96 kHz."
        ) from exc


class ModelRegistry:
    def __init__(self, directory: Path):
        self.directory = directory
        self._cache = {}
        self._lock = threading.Lock()

    def _load(self, modality):
        directory = self.directory / modality
        manifest_path = directory / "backend-model.json"
        if not manifest_path.is_file():
            raise DomainError(503, "model_unavailable", f"Chưa cấu hình backend-model.json cho {modality}.")
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = Manifest.model_validate_json(manifest_bytes)
            if SCHEMAS[manifest.feature_schema][0] != modality:
                raise ValueError("Schema belongs to a different modality")
            paths = []
            for name in (manifest.model_file, manifest.scaler_file):
                path = directory / name
                if Path(name).name != name or not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("Model paths must be filenames within the model directory")
                paths.append(path)
            signature = (manifest_bytes, *((p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
            with self._lock:
                cached = self._cache.get(modality)
                if cached and cached[0] == signature:
                    return cached[1]
                import joblib
                from sklearn.base import is_classifier, is_regressor
                from sklearn.exceptions import InconsistentVersionWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    model, scaler = (joblib.load(p) for p in paths)
                dimension = SCHEMAS[manifest.feature_schema][1]
                if not (is_regressor(model) if modality == "voice" else is_classifier(model)):
                    raise ValueError("Expected a regressor for Voice and a classifier for Drawing/Gait")
                if model.n_features_in_ != dimension or scaler.n_features_in_ != dimension:
                    raise ValueError("Feature dimensions do not match the declared schema")
                if modality != "voice" and manifest.positive_class not in model.classes_:
                    raise ValueError("Positive class is absent from classifier")
                extractor = None
                if manifest.wav_extractor:
                    module, name = manifest.wav_extractor.split(":", 1)
                    extractor = getattr(importlib.import_module(module), name)
                    if not callable(extractor):
                        raise ValueError("WAV extractor must be callable")
                digest = hashlib.sha256(manifest_bytes)
                for path in paths:
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                version = manifest.version + ":" + digest.hexdigest()
                loaded = (manifest, model, scaler, version, extractor)
                self._cache[modality] = (signature, loaded)
                return loaded
        except DomainError:
            raise
        except Exception as exc:
            logger.warning("Model configuration failed for %s: %s", modality, type(exc).__name__)
            raise DomainError(
                503,
                "model_incompatible",
                f"Model/scaler/cấu hình {modality} không tương thích; kiểm tra phiên bản và schema đặc trưng.",
            ) from exc

    def status(self):
        result = {}
        for modality in ("drawing", "gait", "voice"):
            try:
                manifest, _, _, version, extractor = self._load(modality)
                result[modality] = {
                    "ready": True,
                    "model_version": version,
                    "feature_schema": manifest.feature_schema,
                    "feature_input": True,
                    "file_input": manifest.feature_schema in {"drawing.hog.v1", "gait.vgrf.v1"}
                    or (modality == "voice" and extractor is not None),
                }
            except DomainError as exc:
                result[modality] = {"ready": False, "code": exc.code, "message": exc.message}
        return result

    def predict_features(self, modality, schema, values):
        return self._predict_loaded(modality, schema, values, self._load(modality))

    def _predict_loaded(self, modality, schema, values, loaded):
        manifest, model, scaler, version, _ = loaded
        if schema != manifest.feature_schema:
            raise DomainError(422, "feature_schema_mismatch", "Schema đầu vào không khớp schema của model.")
        try:
            vector = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DomainError(422, "invalid_features", "Đặc trưng phải là các giá trị số hữu hạn.") from exc
        if vector.shape != (SCHEMAS[schema][1],) or not np.isfinite(vector).all():
            raise DomainError(
                422, "invalid_features", "Sai số lượng đặc trưng hoặc chứa giá trị không hữu hạn."
            )
        try:
            scaled = scaler.transform(vector.reshape(1, -1))
            if modality == "voice":
                value = float(model.predict(scaled)[0])
            else:
                index = list(model.classes_).index(manifest.positive_class)
                value = float(model.predict_proba(scaled)[0][index]) * 100
            if not math.isfinite(value) or value < 0 or (modality != "voice" and value > 100):
                raise ValueError("Invalid model output")
            return Prediction(value=value, model_version=version, feature_schema=schema)
        except Exception as exc:
            raise DomainError(503, "inference_failed", "Model không trả về kết quả hợp lệ.") from exc

    def predict_file(self, modality, data):
        loaded = self._load(modality)
        manifest, _, _, _, extractor = loaded
        if modality == "drawing":
            if manifest.feature_schema != "drawing.hog.v1":
                raise DomainError(
                    422,
                    "unsupported_model_input",
                    "Model Drawing này cần 9 đặc trưng HandPD; không thể thay bằng thống kê pixel của ảnh.",
                )
            values = image_hog(data)
        elif modality == "gait":
            values = gait_features(data)
        else:
            validate_wav(data)
            if extractor is None:
                raise DomainError(
                    503,
                    "extractor_unavailable",
                    "Chưa có bộ trích 16 đặc trưng WAV tương thích tập huấn luyện. Có thể dùng API đặc trưng đã trích xuất.",
                )
            try:
                features = extractor(data)
                if set(features) != set(VOICE_FEATURES):
                    raise ValueError("Expected the exact 16 feature names")
                values = [features[name] for name in VOICE_FEATURES]
            except Exception as exc:
                raise DomainError(
                    422, "voice_extraction_failed", "Không trích được đủ 16 đặc trưng giọng nói hợp lệ."
                ) from exc
        return self._predict_loaded(modality, manifest.feature_schema, values, loaded)
