import numpy as np
import joblib
import os
import cv2
from skimage.feature import hog

IMG_SIZE = (128, 128)
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(CURRENT_DIR, 'drawing_model.pkl')
SCALER_PATH  = os.path.join(CURRENT_DIR, 'drawing_scaler.pkl')
def load_model():
    """
    Tải mô hình và scaler từ file pkl.
    Trả về (model, scaler) hoặc (None, None) nếu chưa có file.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
def extract_hog_from_image(img_array: np.ndarray) -> np.ndarray:
    """
    Nhận một mảng numpy ảnh (đã đọc bằng cv2 hoặc PIL),
    resize và trích xuất vector HOG.
    Trả về: 1D numpy array (vector đặc trưng)
    """
    # Chuyển sang grayscale nếu ảnh màu
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array
    # Resize về kích thước chuẩn
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    # Trích xuất HOG — giống hệt cấu hình ở preprocess.py
    features = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features
def predict_from_array(img_array: np.ndarray, model, scaler) -> dict:
    """
    Hàm chính: Nhận ảnh dạng numpy array, trả về kết quả dự đoán.
    Output là một dict chứa:
      - risk_score  : % nguy cơ Parkinson (0.0 ~ 100.0)
      - label       : 'Parkinson' hoặc 'Khỏe mạnh'
      - confidence  : xác suất dự đoán của nhãn được chọn
    """
    # Trích xuất vector HOG từ ảnh
    features = extract_hog_from_image(img_array)
    # Chuẩn hóa về cùng hệ quy chiếu với lúc train
    features_sc = scaler.transform([features])
    # Lấy xác suất dự đoán từ Random Forest
    # proba[0][0] = xác suất là "Khỏe mạnh" (nhãn 0)
    # proba[0][1] = xác suất là "Parkinson"  (nhãn 1)
    proba = model.predict_proba(features_sc)[0]
    risk_score  = proba[1] * 100           # Đổi sang % nguy cơ Parkinson
    label       = 'Parkinson' if risk_score >= 50 else 'Khỏe mạnh'
    confidence  = max(proba) * 100
    return {
        'risk_score': round(risk_score, 2),    # VD: 82.15 (%)
        'label'     : label,                   # VD: 'Parkinson'
        'confidence': round(confidence, 2)     # VD: 82.15 (%)
    }
def predict_from_file(image_path: str, model, scaler) -> dict:
    """
    Tiện ích: Nhận đường dẫn file ảnh thay vì numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được file ảnh: {image_path}")
    return predict_from_array(img, model, scaler)

if __name__ == "__main__":
    import sys
    model, scaler = load_model()
    if model is None:
        print(" Chưa có model. Hãy chạy train.py trước!")
    elif len(sys.argv) > 1:
        result = predict_from_file(sys.argv[1], model, scaler)
        print(f"  Kết quả dự đoán:")
        print(f"   Drawing Risk Score : {result['risk_score']}%")
        print(f"   Nhãn dự đoán       : {result['label']}")
        print(f"   Độ tự tin          : {result['confidence']}%")
    else:
        print("  Dùng: python predict.py <đường_dẫn_ảnh.png>")
