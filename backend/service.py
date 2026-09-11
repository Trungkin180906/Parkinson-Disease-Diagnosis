import hashlib
import math
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from backend.db import as_dict, canonical
from backend.errors import DomainError
from backend.policy import recommendation, risk_level, voice_history, voice_score

NOTICE = "Kết quả AI hỗ trợ sàng lọc và theo dõi; không thay thế chẩn đoán của bác sĩ. Voice Score là chỉ số hiển thị của dự án."


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid4())


def required(conn, table, identifier):
    # table is a server constant, never an HTTP parameter.
    row = conn.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise DomainError(404, "not_found", "Không tìm thấy hồ sơ hoặc phiên thực hiện.")
    return dict(row)


def insert(conn, table, data):
    conn.execute(
        f"INSERT INTO {table} ({','.join(data)}) VALUES ({','.join('?' for _ in data)})", tuple(data.values())
    )


class Service:
    def __init__(self, db, settings, predictor):
        self.db, self.settings, self.predictor = db, settings, predictor

    def create_patient(self, payload):
        identifier, timestamp = uid(), now()
        row = dict(
            id=identifier,
            code="PD-" + identifier.replace("-", "").upper(),
            **payload.model_dump(),
            diagnosis_status="unknown",
            version=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self.db.transaction(write=True) as conn:
            insert(conn, "patients", row)
        return row

    def list_patients(self, search, limit, offset):
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where = " WHERE full_name LIKE ? ESCAPE '\\' OR code LIKE ? ESCAPE '\\'"
        args = ("%" + escaped + "%",) * 2
        with self.db.transaction() as conn:
            total = conn.execute("SELECT count(*) FROM patients" + where, args).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM patients" + where + " ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                (*args, limit, offset),
            )
            return {"items": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}

    def patient(self, patient_id):
        with self.db.transaction() as conn:
            row = required(conn, "patients", patient_id)
            row["monitoring_eligibility"] = self.eligibility(conn, row)
            return row

    def update_patient(self, patient_id, payload):
        with self.db.transaction(write=True) as conn:
            row = required(conn, "patients", patient_id)
            if row["version"] != payload.version:
                raise DomainError(409, "version_conflict", "Hồ sơ đã thay đổi; tải lại trước khi cập nhật.")
            values = payload.model_dump(exclude={"version"})
            values.update(version=row["version"] + 1, updated_at=now())
            conn.execute(
                "UPDATE patients SET " + ",".join(k + "=?" for k in values) + " WHERE id=?",
                (*values.values(), patient_id),
            )
            return required(conn, "patients", patient_id)

    def record_diagnosis(self, patient_id, payload):
        with self.db.transaction(write=True) as conn:
            patient = required(conn, "patients", patient_id)
            if patient["version"] != payload.version:
                raise DomainError(409, "version_conflict", "Hồ sơ đã thay đổi; tải lại trước khi cập nhật.")
            insert(
                conn,
                "diagnoses",
                dict(
                    id=uid(),
                    patient_id=patient_id,
                    **payload.model_dump(mode="json", exclude={"version"}),
                    created_at=now(),
                ),
            )
            conn.execute(
                "UPDATE patients SET diagnosis_status=?, version=version+1, updated_at=? WHERE id=?",
                (payload.status, now(), patient_id),
            )
            return required(conn, "patients", patient_id)

    def eligibility(self, conn, patient):
        basis = None
        if patient["diagnosis_status"] == "confirmed":
            basis = "clinician_confirmed"
        elif (
            patient["diagnosis_status"] != "ruled_out"
            and self.settings.monitoring_policy == "at_risk_or_confirmed"
        ):
            latest = conn.execute(
                "SELECT * FROM screenings WHERE patient_id=? AND status='completed' ORDER BY performed_on DESC, created_at DESC, id DESC LIMIT 1",
                (patient["id"],),
            ).fetchone()
            if latest is not None and latest["overall_risk"] >= latest["referral_threshold"]:
                basis = "screening:" + latest["id"]
        return {"eligible": basis is not None, "basis": basis, "policy": self.settings.monitoring_policy}

    def ensure_monitoring(self, conn, patient_id):
        patient = required(conn, "patients", patient_id)
        eligibility = self.eligibility(conn, patient)
        if not eligibility["eligible"]:
            raise DomainError(
                409,
                "monitoring_not_eligible",
                "Hồ sơ chưa đủ điều kiện theo dõi Voice theo cấu hình hệ thống.",
            )
        return eligibility["basis"]

    def create_screening(self, patient_id, payload):
        row = dict(
            id=uid(),
            patient_id=patient_id,
            **payload.model_dump(mode="json"),
            status="pending",
            created_at=now(),
            drawing_weight=self.settings.drawing_weight,
            referral_threshold=self.settings.referral_threshold,
            overall_risk=None,
            risk_level=None,
            recommendation=None,
            completed_at=None,
        )
        with self.db.transaction(write=True) as conn:
            required(conn, "patients", patient_id)
            insert(conn, "screenings", row)
        return self.screening(row["id"])

    def _screening(self, conn, screening_id):
        row = required(conn, "screenings", screening_id)
        row["results"] = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM screening_results WHERE screening_id=? ORDER BY modality", (screening_id,)
            )
        ]
        row["missing_modalities"] = sorted({"drawing", "gait"} - {r["modality"] for r in row["results"]})
        row["notice"] = NOTICE
        return row

    def screening(self, screening_id):
        with self.db.transaction() as conn:
            return self._screening(conn, screening_id)

    def list_screenings(self, patient_id):
        with self.db.transaction() as conn:
            required(conn, "patients", patient_id)
            ids = conn.execute(
                "SELECT id FROM screenings WHERE patient_id=? ORDER BY performed_on DESC, created_at DESC, id DESC",
                (patient_id,),
            ).fetchall()
            return [self._screening(conn, r["id"]) for r in ids]

    def analyze(self, modality, features=None, data=None):
        if features is not None:
            digest = hashlib.sha256(
                canonical(features.model_dump(include={"feature_schema", "values"})).encode()
            ).hexdigest()
            prediction = self.predictor.predict_features(modality, features.feature_schema, features.values)
            kind = "features"
        else:
            digest = hashlib.sha256(data).hexdigest()
            prediction = self.predictor.predict_file(modality, data)
            kind = "file"
        if (
            not math.isfinite(prediction.value)
            or prediction.value < 0
            or (modality != "voice" and prediction.value > 100)
        ):
            raise DomainError(503, "invalid_model_output", "Model trả về giá trị ngoài phạm vi hợp lệ.")
        return prediction, kind, digest

    def add_screening_result(self, screening_id, modality, features=None, data=None):
        with self.db.transaction() as conn:
            screening = self._screening(conn, screening_id)
            if screening["status"] == "completed" or modality not in screening["missing_modalities"]:
                raise DomainError(
                    409, "result_exists", "Kết quả đã được lưu. Tạo phiên mới để phân tích lại."
                )
        prediction, kind, digest = self.analyze(modality, features, data)
        with self.db.transaction(write=True) as conn:
            screening = self._screening(conn, screening_id)
            if screening["status"] == "completed" or modality not in screening["missing_modalities"]:
                raise DomainError(409, "result_exists", "Một yêu cầu khác đã lưu kết quả này.")
            insert(
                conn,
                "screening_results",
                dict(
                    id=uid(),
                    screening_id=screening_id,
                    modality=modality,
                    risk_score=prediction.value,
                    model_version=prediction.model_version,
                    feature_schema=prediction.feature_schema,
                    input_kind=kind,
                    input_sha256=digest,
                    created_at=now(),
                ),
            )
            scores = {r["modality"]: r["risk_score"] for r in self._screening(conn, screening_id)["results"]}
            if len(scores) == 2:
                score = round(
                    scores["drawing"] * screening["drawing_weight"]
                    + scores["gait"] * (1 - screening["drawing_weight"]),
                    2,
                )
                conn.execute(
                    "UPDATE screenings SET status='completed', overall_risk=?, risk_level=?, recommendation=?, completed_at=? WHERE id=?",
                    (
                        score,
                        risk_level(score),
                        recommendation(score, screening["referral_threshold"]),
                        now(),
                        screening_id,
                    ),
                )
            else:
                conn.execute("UPDATE screenings SET status='partial' WHERE id=?", (screening_id,))
            return self._screening(conn, screening_id)

    def add_voice(self, patient_id, recorded_at, notes, features=None, data=None):
        timestamp = recorded_at.isoformat()
        with self.db.transaction() as conn:
            self.ensure_monitoring(conn, patient_id)
            if conn.execute(
                "SELECT 1 FROM voice_records WHERE patient_id=? AND recorded_at=?", (patient_id, timestamp)
            ).fetchone():
                raise DomainError(409, "record_exists", "Đã có kết quả Voice tại thời điểm này.")
        prediction, kind, digest = self.analyze("voice", features, data)
        row = dict(
            id=uid(),
            patient_id=patient_id,
            recorded_at=timestamp,
            notes=notes,
            total_updrs=prediction.value,
            voice_score=voice_score(prediction.value, self.settings.voice_reference_max),
            reference_max=self.settings.voice_reference_max,
            change_threshold=self.settings.voice_change_threshold,
            model_version=prediction.model_version,
            feature_schema=prediction.feature_schema,
            input_kind=kind,
            input_sha256=digest,
            created_at=now(),
        )
        try:
            with self.db.transaction(write=True) as conn:
                row["eligibility_basis"] = self.ensure_monitoring(conn, patient_id)
                insert(conn, "voice_records", row)
                return next(r for r in self._history(conn, patient_id) if r["id"] == row["id"])
        except sqlite3.IntegrityError as exc:
            raise DomainError(409, "record_exists", "Đã có kết quả Voice tại thời điểm này.") from exc

    def _history(self, conn, patient_id):
        rows = conn.execute(
            "SELECT * FROM voice_records WHERE patient_id=? ORDER BY recorded_at, id", (patient_id,)
        )
        return voice_history(rows)

    def history(self, patient_id):
        with self.db.transaction() as conn:
            required(conn, "patients", patient_id)
            return self._history(conn, patient_id)

    def report(self, patient_id):
        with self.db.transaction() as conn:
            patient = required(conn, "patients", patient_id)
            ids = conn.execute(
                "SELECT id FROM screenings WHERE patient_id=? ORDER BY performed_on, created_at, id",
                (patient_id,),
            ).fetchall()
            screenings = [self._screening(conn, r["id"]) for r in ids]
            diagnoses = [
                as_dict(r)
                for r in conn.execute(
                    "SELECT * FROM diagnoses WHERE patient_id=? ORDER BY created_at, id", (patient_id,)
                )
            ]
            return {
                "generated_at": now(),
                "patient": patient,
                "diagnoses": diagnoses,
                "screenings": screenings,
                "voice_history": self._history(conn, patient_id),
                "monitoring_eligibility": self.eligibility(conn, patient),
                "notice": NOTICE,
            }
