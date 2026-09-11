import streamlit as st
import numpy as np
import joblib
import os
import cv2
from skimage.feature import hog
from PIL import Image
# --- Cấu hình trang web ---
st.set_page_config(
    page_title="Drawing Analysis",
    page_icon="🌀",
    layout="centered"
)
# --- Tiêu đề ---
st.markdown('<h1 style="color:#6C63FF;text-align:center;">🌀 Drawing Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Tải lên ảnh bài kiểm tra vẽ xoắn ốc để AI phân tích nguy cơ Parkinson</p>', unsafe_allow_html=True)
# --- Cấu hình xử lý ảnh (giữ nguyên như preprocess.py) ---
IMG_SIZE = (128, 128)
# --- Đường dẫn file model ---
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(CURRENT_DIR, 'drawing_model.pkl')
SCALER_PATH  = os.path.join(CURRENT_DIR, 'drawing_scaler.pkl')
# -------------------------------------------------------
# BƯỚC 1: TẢI MÔ HÌNH VÀO RAM (cache lại, không load lại mỗi lần bấm)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
model, scaler = load_model()
# Kiểm tra model có sẵn không
if model is None:
    st.error("❌ Chưa có file mô hình! Hãy chạy train.py trước.")
    st.code("python drawing/train.py")
    st.stop()
# -------------------------------------------------------
# BƯỚC 2: HÀM TRÍCH XUẤT ĐẶC TRƯNG HOG TỪ ẢNH
# -------------------------------------------------------
def extract_hog(img_array: np.ndarray) -> np.ndarray:
    """
    Nhận numpy array ảnh, chuyển xám, resize, trích xuất vector HOG.
    Phải giống hệt cấu hình trong preprocess.py.
    """
    # Nếu ảnh màu (3 kênh) thì chuyển về grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    # Resize về kích thước chuẩn 128x128
    resized = cv2.resize(gray, IMG_SIZE)
    # Trích xuất HOG — đúng cấu hình như lúc train
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features
# -------------------------------------------------------
# BƯỚC 3: GIAO DIỆN UPLOAD ẢNH
# -------------------------------------------------------
st.markdown("---")
st.header("1. Upload Ảnh Hình Vẽ")
st.info("📋 Hướng dẫn: Yêu cầu người dùng vẽ lại hình xoắn ốc trên giấy trắng, chụp ảnh rõ nét rồi tải lên đây.")
# Nút upload — chấp nhận các định dạng ảnh phổ biến
uploaded_file = st.file_uploader(
    "Chọn ảnh hình vẽ (.jpg, .jpeg, .png, .bmp)",
    type=["jpg", "jpeg", "png", "bmp"]
)
if uploaded_file:
    # Hiển thị ảnh vừa upload để người dùng xác nhận
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Ảnh vừa tải lên", use_container_width=True)
# -------------------------------------------------------
# BƯỚC 4: NÚT BẤM → PHÂN TÍCH
# -------------------------------------------------------
st.markdown("---")
if st.button("🔍 PHÂN TÍCH NGUY CƠ PARKINSON", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ Vui lòng tải ảnh lên trước!")
        st.stop()
    with st.spinner("AI đang phân tích nét vẽ..."):
        # Chuyển ảnh PIL → numpy array để xử lý
        img_array = np.array(img_pil)
        # Trích xuất vector HOG từ ảnh
        features = extract_hog(img_array)
        # Chuẩn hóa vector về cùng hệ quy chiếu với lúc train
        features_sc = scaler.transform([features])
        # Lấy xác suất dự đoán
        # proba[0][0] = xác suất Khỏe mạnh
        # proba[0][1] = xác suất Parkinson
        proba = model.predict_proba(features_sc)[0]
        risk_score = round(proba[1] * 100, 2)   # % nguy cơ Parkinson
        healthy_score = round(proba[0] * 100, 2) # % khỏe mạnh
    st.markdown("---")
    st.header("2. Kết Quả Phân Tích")
    # Hiển thị 2 chỉ số song song
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🌀 Drawing Risk Score", value=f"{risk_score}%", delta="Nguy cơ Parkinson")
    with col2:
        st.metric(label="✅ Healthy Score", value=f"{healthy_score}%", delta="Không có dấu hiệu")
    # Thanh progress bar trực quan
    st.markdown("**Mức độ nguy cơ:**")
    st.progress(int(risk_score))    
    # Phân loại và cảnh báo theo ngưỡng
    st.markdown("---")
    if risk_score < 30:
        st.success(f"🟢 **Nguy cơ thấp ({risk_score}%)** — Hình vẽ không có nhiều dấu hiệu bất thường. Khuyến nghị tiếp tục theo dõi sức khỏe định kỳ.")
    elif risk_score < 60:
        st.warning(f"🟡 **Nguy cơ trung bình ({risk_score}%)** — Phát hiện một số dấu hiệu bất thường trong nét vẽ. Nên tham khảo ý kiến bác sĩ.")
    elif risk_score < 80:
        st.error(f"🔴 **Nguy cơ cao ({risk_score}%)** — Hình vẽ có nhiều đặc điểm tương đồng với bệnh nhân Parkinson. Khuyến nghị đến cơ sở y tế kiểm tra.")
    else:
        st.error(f"🔴 **Nguy cơ rất cao ({risk_score}%)** — Hình vẽ có đặc điểm rất rõ ràng của bệnh nhân Parkinson. Cần đến bác sĩ ngay để được đánh giá chuyên sâu.")
        # Ghi chú quan trọng
    st.markdown("---")
    st.caption("⚠️ **Lưu ý:** Kết quả này chỉ mang tính chất tham khảo hỗ trợ sàng lọc. AI không thể thay thế kết luận chẩn đoán cuối cùng của bác sĩ chuyên khoa.")