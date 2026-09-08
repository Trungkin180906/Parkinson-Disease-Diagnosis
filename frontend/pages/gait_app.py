import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Cấu hình trang web ---
st.set_page_config(
    page_title="Gait Analysis",
    page_icon="👣",
    layout="centered"
)

# --- Tiêu đề ---
st.markdown('<h1 style="color:#20B2AA;text-align:center;">👣 Gait Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Tải lên file dữ liệu cảm biến bước chân để AI hỗ trợ sàng lọc nguy cơ Parkinson</p>', unsafe_allow_html=True)

# --- Cấu hình đường dẫn model ---
# Chú ý chỉnh lại đường dẫn này nếu cấu trúc thư mục của ông khác đi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Lùi về thư mục gốc rồi trỏ tới ai/gait
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")) 
MODEL_PATH   = os.path.join(PROJECT_ROOT, "ai", "gait", "gait_model.pkl")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "ai", "gait", "gait_scaler.pkl")

# -------------------------------------------------------
# BƯỚC 1: TẢI MÔ HÌNH VÀO RAM (Dùng Cache để web chạy mượt)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_model()

if model is None:
    st.error("❌ Chưa tìm thấy file mô hình (gait_model.pkl). Hãy kiểm tra lại đường dẫn!")
    st.stop()

# -------------------------------------------------------
# BƯỚC 2: HÀM TRÍCH XUẤT ĐẶC TRƯNG TỪ FILE SỐ LIỆU
# -------------------------------------------------------
def extract_features(df: pd.DataFrame):
    """
    Hàm này phải giống hệt hàm trong preprocess.py lúc train.
    Trích xuất 5 đặc trưng: mean_L, mean_R, std_L, std_R, asymmetry.
    """
    try:
        total_left = df.iloc[:, 16]
        total_right = df.iloc[:, 17]
        
        mean_l = total_left.mean()
        mean_r = total_right.mean()
        std_l = total_left.std()
        std_r = total_right.std()
        asymmetry = abs(mean_l - mean_r)
        
        return np.array([mean_l, mean_r, std_l, std_r, asymmetry])
    except Exception as e:
        st.error(f"⚠️ Lỗi định dạng file: Không thể đọc cột số liệu. Chi tiết lỗi: {e}")
        return None

# -------------------------------------------------------
# BƯỚC 3: GIAO DIỆN UPLOAD FILE
# -------------------------------------------------------
st.markdown("---")
st.header("1. Upload Dữ Liệu Dáng Đi")
st.info("📋 Tải lên file .txt chứa dữ liệu đo lường từ cảm biến lực bàn chân (Vertical Ground Reaction Force).")

uploaded_file = st.file_uploader(
    "Chọn file dữ liệu (.txt)",
    type=["txt", "csv"]
)

if uploaded_file:
    # Đọc file để người dùng xem trước 5 dòng
    df_preview = pd.read_csv(uploaded_file, sep='\t', header=None, on_bad_lines='skip')
    st.write("👀 **Xem trước dữ liệu cảm biến:**")
    st.dataframe(df_preview.head(5), use_container_width=True)

# -------------------------------------------------------
# BƯỚC 4: NÚT BẤM → PHÂN TÍCH
# -------------------------------------------------------
st.markdown("---")
if st.button("🔍 PHÂN TÍCH NGUY CƠ PARKINSON", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("⚠️ Vui lòng tải file số liệu lên trước!")
        st.stop()
        
    with st.spinner("AI đang phân tích nhịp bước chân..."):
        # Đưa thanh cuộn về đầu file trước khi đọc lại
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep='\t', header=None, on_bad_lines='skip')
        
        # Trích xuất đặc trưng
        features = extract_features(df)
        
        if features is not None:
            # Chuẩn hóa vector
            features_sc = scaler.transform([features])
            
            # Lấy xác suất dự đoán (proba[0][1] là nhãn 1 - Parkinson)
            proba = model.predict_proba(features_sc)[0]
            risk_score = round(proba[1] * 100, 2)   # % nguy cơ Parkinson
            healthy_score = round(proba[0] * 100, 2) # % khỏe mạnh
            
            st.markdown("---")
            st.header("2. Kết Quả Phân Tích")
            
            # Hiển thị 2 chỉ số song song
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="👣 Gait Risk Score", value=f"{risk_score}%", delta="Nguy cơ Parkinson", delta_color="inverse")
            with col2:
                st.metric(label="✅ Healthy Score", value=f"{healthy_score}%", delta="Nhịp điệu ổn định")
            
            # Thanh progress bar trực quan
            st.markdown("**Mức độ nguy cơ:**")
            st.progress(int(risk_score))    
            
            # Phân loại và cảnh báo theo ngưỡng (Đồng bộ với yêu cầu nhóm)
            st.markdown("---")
            if risk_score <= 30:
                st.success(f"🟢 **Nguy cơ thấp ({risk_score}%)** — Không phát hiện nhiều dấu hiệu bất thường. Dáng đi ổn định. Khuyến nghị tiếp tục theo dõi sức khỏe định kỳ.")
            elif risk_score <= 60:
                st.warning(f"🟡 **Nguy cơ trung bình ({risk_score}%)** — Phát hiện một số độ trễ hoặc bất đối xứng nhẹ trong bước đi. Có thể cần kiểm tra thêm.")
            elif risk_score <= 80:
                st.error(f"🔴 **Nguy cơ cao ({risk_score}%)** — Dáng đi có nhiều đặc điểm bất đối xứng và ngập ngừng tương đồng với bệnh nhân Parkinson. Khuyến nghị đến cơ sở y tế để được bác sĩ đánh giá.")
            else:
                st.error(f"🔴 **Nguy cơ rất cao ({risk_score}%)** — Dữ liệu cảm biến cho thấy sự bất ổn định nghiêm trọng đặc trưng của Parkinson. Người dùng có nguy cơ cao mắc Parkinson. Khuyến nghị đến cơ sở y tế ngay.")
            
            # Ghi chú quan trọng
            st.markdown("---")
            st.caption("⚠️ **Lưu ý:** Kết quả này chỉ mang tính chất tham khảo hỗ trợ sàng lọc thông qua dữ liệu dáng đi. AI không thể thay thế kết luận chẩn đoán cuối cùng của bác sĩ chuyên khoa.")