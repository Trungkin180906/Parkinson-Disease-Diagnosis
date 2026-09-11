import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
 id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, full_name TEXT NOT NULL,
 age INTEGER NOT NULL, sex TEXT NOT NULL, height_cm REAL NOT NULL, weight_kg REAL NOT NULL,
 phone TEXT, diagnosis_status TEXT NOT NULL DEFAULT 'unknown',
 version INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnoses (
 id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
 status TEXT NOT NULL, clinician_name TEXT NOT NULL, assessed_on TEXT NOT NULL,
 notes TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screenings (
 id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id), performed_on TEXT NOT NULL,
 notes TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
 drawing_weight REAL NOT NULL, referral_threshold REAL NOT NULL,
 overall_risk REAL, risk_level TEXT, recommendation TEXT, completed_at TEXT
);
CREATE TABLE IF NOT EXISTS screening_results (
 id TEXT PRIMARY KEY, screening_id TEXT NOT NULL REFERENCES screenings(id),
 modality TEXT NOT NULL CHECK(modality IN ('drawing','gait')), risk_score REAL NOT NULL CHECK(risk_score BETWEEN 0 AND 100),
 model_version TEXT NOT NULL, feature_schema TEXT NOT NULL, input_kind TEXT NOT NULL,
 input_sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(screening_id, modality)
);
CREATE TABLE IF NOT EXISTS voice_records (
 id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id), recorded_at TEXT NOT NULL,
 notes TEXT NOT NULL, total_updrs REAL NOT NULL CHECK(total_updrs >= 0), voice_score REAL NOT NULL,
 reference_max REAL NOT NULL, change_threshold REAL NOT NULL,
 model_version TEXT NOT NULL, feature_schema TEXT NOT NULL, input_kind TEXT NOT NULL,
 input_sha256 TEXT NOT NULL, eligibility_basis TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(patient_id, recorded_at)
);
CREATE INDEX IF NOT EXISTS screenings_patient ON screenings(patient_id, performed_on, created_at);
CREATE INDEX IF NOT EXISTS voice_patient ON voice_records(patient_id, recorded_at);
CREATE INDEX IF NOT EXISTS diagnosis_patient ON diagnoses(patient_id, created_at);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("Unsupported database schema version")
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA user_version=1")

    @contextmanager
    def transaction(self, write=False):
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def as_dict(row):
    return dict(row) if row is not None else None


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
