# ============================================================
# FILE: preprocess.py
# MỤC ĐÍCH: Đọc ảnh từ 2 thư mục, trích xuất đặc trưng HOG,
#            lưu ra file numpy để train.py dùng
# ============================================================

import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import joblib

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CONTROL_DIR  = r"D:\drawing\SpiralControl"   # Thư mục ảnh người khỏe mạnh (nhãn 0)
PATIENTS_DIR = r"D:\drawing\SpiralPatients"  # Thư mục ảnh bệnh nhân Parkinson (nhãn 1)
SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))  # Lưu cùng thư mục với file này

# --- CẤU HÌNH XỬ LÝ ẢNH ---
IMG_SIZE = (128, 128)   # Kích thước resize đồng nhất tất cả ảnh
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')  # Định dạng ảnh được nhận


def load_and_extract(folder, label):
    """
    Đọc toàn bộ ảnh trong một thư mục, chuyển sang grayscale,
    resize về IMG_SIZE, rồi trích xuất vector đặc trưng HOG.
    Trả về: danh sách vector đặc trưng (X) và danh sách nhãn (y)
    """
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)]
    print(f"  → Đang xử lý {len(files)} ảnh từ: {folder}")

    for fname in files:
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dạng thang xám
        if img is None:
            print(f"  ⚠️  Bỏ qua (không đọc được): {fname}")
            continue

        # Resize ảnh về kích thước chuẩn để đồng nhất kích thước vector đặc trưng
        img = cv2.resize(img, IMG_SIZE)

        # Trích xuất đặc trưng HOG (Histogram of Oriented Gradients)
        # HOG bắt được các cạnh, hướng nét bút, và kết cấu hình vẽ
        features = hog(
            img,
            orientations=9,        # Chia hướng cạnh thành 9 góc
            pixels_per_cell=(8, 8),# Mỗi ô nhỏ gồm 8x8 pixel
            cells_per_block=(2, 2),# Mỗi block gồm 2x2 ô để chuẩn hóa
            block_norm='L2-Hys',   # Phương pháp chuẩn hóa L2
            visualize=False,
            feature_vector=True    # Trả về 1 vector 1 chiều
        )
        X.append(features)
        y.append(label)

    return X, y


def main():
    print("=" * 50)
    print("🔄 BƯỚC 1: ĐỌC VÀ TRÍCH XUẤT ĐẶC TRƯNG ẢNH")
    print("=" * 50)

    # Xử lý ảnh người khỏe (nhãn 0) và bệnh nhân (nhãn 1)
    X0, y0 = load_and_extract(CONTROL_DIR, label=0)
    X1, y1 = load_and_extract(PATIENTS_DIR, label=1)

    # Ghép 2 nhóm lại thành 1 bộ dữ liệu
    X = np.array(X0 + X1, dtype=np.float32)
    y = np.array(y0 + y1, dtype=np.int32)

    print(f"\n✅ Tổng số mẫu: {len(X)} ảnh")
    print(f"   - Khỏe mạnh (Control) : {len(X0)}")
    print(f"   - Bệnh nhân (Patients) : {len(X1)}")
    print(f"   - Kích thước 1 vector  : {X.shape[1]} chiều")

    # Lưu ra file .npy để train.py đọc lại nhanh (không cần xử lý ảnh lại)
    np.save(os.path.join(SAVE_DIR, 'X_drawing.npy'), X)
    np.save(os.path.join(SAVE_DIR, 'y_drawing.npy'), y)
    print(f"\n💾 Đã lưu: X_drawing.npy  và  y_drawing.npy")


if __name__ == "__main__":
    main()
