import numpy as np
import joblib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "drawing_model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "drawing_scaler.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict_from_features(features, model, scaler):
    """
    features: 9 đặc trưng của Drawing
    """

    features = np.array(features, dtype=float).reshape(1, -1)

    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]

    # Nhãn 1 được mã hóa thành 0, nhãn 2 được mã hóa thành 1
    risk_score = proba[1] * 100

    label = "Parkinson" if risk_score >= 50 else "Khỏe mạnh"

    confidence = max(proba) * 100

    return {
        "risk_score": round(risk_score, 2),
        "label": label,
        "confidence": round(confidence, 2)
    }


if __name__ == "__main__":
    model, scaler = load_model()

    if model is None:
        print("Chưa có model. Hãy chạy train.py trước!")
    else:
        print("Drawing model đã load thành công!")
        print("Số đặc trưng model yêu cầu:", model.n_features_in_)