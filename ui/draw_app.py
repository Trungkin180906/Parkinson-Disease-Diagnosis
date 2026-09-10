import streamlit as st
import numpy as np
import joblib
import os
import cv2
from PIL import Image

# =========================
# CẤU HÌNH
# =========================
st.set_page_config(
    page_title="Drawing Analysis",
    page_icon="🌀",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai", "drawing", "drawing_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ai", "drawing", "drawing_scaler.pkl")

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# GIAO DIỆN
# =========================
st.markdown(
    '<h1 style="color:#6C63FF;text-align:center;">🌀 Drawing Analysis</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="text-align:center;">Tải lên ảnh bài kiểm tra vẽ xoắn ốc để AI phân tích</p>',
    unsafe_allow_html=True
)

st.markdown("---")

st.header("1. Upload ảnh hình vẽ")

uploaded_file = st.file_uploader(
    "Chọn ảnh hình vẽ",
    type=["jpg", "jpeg", "png", "bmp"]
)

# =========================
# TRÍCH 9 ĐẶC TRƯNG ẢNH
# =========================
def extract_features(img_array):

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Chuẩn hóa ảnh
    gray = cv2.resize(gray, (128, 128))

    # 9 đặc trưng cơ bản
    features = [
        np.mean(gray),
        np.std(gray),
        np.mean(gray ** 2),
        np.min(gray),
        np.max(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
        np.sum(gray < 128)
    ]

    return np.array(features, dtype=float)


# =========================
# HIỂN THỊ ẢNH
# =========================
if uploaded_file:

    img_pil = Image.open(uploaded_file).convert("RGB")

    st.image(
        img_pil,
        caption="Ảnh hình vẽ đã tải lên",
        use_container_width=True
    )

# =========================
# PHÂN TÍCH
# =========================
st.markdown("---")

if st.button(
    "🔍 PHÂN TÍCH NGUY CƠ PARKINSON",
    type="primary",
    use_container_width=True
):

    if not uploaded_file:
        st.warning("⚠️ Vui lòng tải ảnh lên trước!")
        st.stop()

    with st.spinner("AI đang phân tích hình vẽ..."):

        img_array = np.array(img_pil)

        # Lấy 9 features
        features = extract_features(img_array)

        # Chuẩn hóa
        features_scaled = scaler.transform([features])

        # Dự đoán
        proba = model.predict_proba(features_scaled)[0]

        risk_score = round(proba[1] * 100, 2)
        healthy_score = round(proba[0] * 100, 2)

    # =========================
    # KẾT QUẢ
    # =========================
    st.markdown("---")
    st.header("2. Kết quả phân tích")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🌀 Drawing Risk Score",
            f"{risk_score}%"
        )

    with col2:
        st.metric(
            "✅ Healthy Score",
            f"{healthy_score}%"
        )

    st.markdown("**Mức độ nguy cơ:**")
    st.progress(int(risk_score))

    if risk_score < 30:
        st.success(
            f"🟢 **Nguy cơ thấp ({risk_score}%)**"
        )

    elif risk_score < 60:
        st.warning(
            f"🟡 **Nguy cơ trung bình ({risk_score}%)**"
        )

    else:
        st.error(
            f"🔴 **Nguy cơ cao ({risk_score}%)**"
        )

    st.markdown("---")

    st.caption(
        "⚠️ Kết quả chỉ mang tính tham khảo và hỗ trợ sàng lọc, "
        "không thay thế chẩn đoán của bác sĩ chuyên khoa."
    )