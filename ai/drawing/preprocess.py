# ============================================================
# FILE: preprocess.py (THƯ MỤC DRAWING)
# MỤC ĐÍCH: Tiền xử lý ảnh (chuyển xám, đổi kích thước) và 
#           trích xuất đặc trưng HOG (Histogram of Oriented Gradients).
# ============================================================

import os
import numpy as np
import cv2
from skimage.feature import hog

# ---------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & KÍCH THƯỚC ẢNH
# ---------------------------------------------------------
# Khai báo đường dẫn đến 2 thư mục chứa ảnh gốc
CONTROL_DIR  = r"D:\drawing\SpiralControl"   # Ảnh người khỏe mạnh (Nhãn 0)
PATIENTS_DIR = r"D:\drawing\SpiralPatients"  # Ảnh bệnh nhân Parkinson (Nhãn 1)

# Lấy đường dẫn thư mục hiện tại để lưu file kết quả
SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))  

# Ép tất cả ảnh về chung một kích thước (128x128 pixel) 
# Việc này bắt buộc để vector đặc trưng HOG của mọi ảnh đều có cùng chiều dài
IMG_SIZE = (128, 128)   

# Chỉ đọc các file có đuôi mở rộng là ảnh
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')  


def load_and_extract(folder, label):
    """
    Hàm đọc toàn bộ ảnh trong 1 thư mục, chuyển thành ảnh xám, 
    resize và dùng thuật toán HOG bóc tách đặc trưng.
    """
    X = [] # List chứa các vector đặc trưng của ảnh
    y = [] # List chứa nhãn (0 hoặc 1)
    
    # Quét qua tất cả file trong thư mục, chỉ lấy các file có đuôi ảnh
    files = [f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)]
    print(f"  → Đang xử lý {len(files)} ảnh từ: {folder}")

    for fname in files:
        img_path = os.path.join(folder, fname)
        
        # Dùng OpenCV đọc ảnh. cv2.IMREAD_GRAYSCALE tự động chuyển ảnh màu thành trắng đen
        # Lý do: Hình vẽ nét bút chỉ quan tâm đến hình dáng (edges/gradients), không quan tâm màu sắc.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        if img is None:
            print(f"  ⚠️  Bỏ qua (không đọc được): {fname}")
            continue

        # Đổi kích thước ảnh về 128x128
        img = cv2.resize(img, IMG_SIZE)

        # ---------------------------------------------------------
        # THUẬT TOÁN HOG (Histogram of Oriented Gradients)
        # Bắt các cạnh, góc, đường gãy khúc và độ run của nét bút
        # ---------------------------------------------------------
        features = hog(
            img,
            orientations=9,        # Chia không gian hướng của nét bút thành 9 góc (0-180 độ)
            pixels_per_cell=(8, 8),# Chia nhỏ ảnh thành các ô vuông 8x8 pixel để phân tích cục bộ
            cells_per_block=(2, 2),# Gom 4 ô (2x2) thành 1 khối để chuẩn hóa ánh sáng
            block_norm='L2-Hys',   # Kỹ thuật chuẩn hóa tránh nhiễu do bóng mờ, độ sáng không đều
            visualize=False,       # Không vẽ ảnh HOG ra màn hình (tiết kiệm bộ nhớ)
            feature_vector=True    # Dàn phẳng ma trận HOG thành 1 vector 1 chiều dài (VD: 8100 số)
        )
        
        # Lưu vector và nhãn vào List
        X.append(features)
        y.append(label)

    return X, y


def main():
    print("=" * 50)
    print("🔄 BƯỚC 1: ĐỌC VÀ TRÍCH XUẤT ĐẶC TRƯNG ẢNH BẰNG HOG")
    print("=" * 50)

    # 1. Gọi hàm xử lý cho tập người khỏe mạnh (gắn nhãn = 0)
    X0, y0 = load_and_extract(CONTROL_DIR, label=0)
    
    # 2. Gọi hàm xử lý cho tập bệnh nhân Parkinson (gắn nhãn = 1)
    X1, y1 = load_and_extract(PATIENTS_DIR, label=1)

    # 3. Gộp 2 tập lại thành 1 bộ Data tổng (Mảng Numpy)
    # np.float32 và np.int32 giúp tối ưu bộ nhớ RAM
    X = np.array(X0 + X1, dtype=np.float32)
    y = np.array(y0 + y1, dtype=np.int32)

    print(f"\n✅ Tổng số mẫu: {len(X)} ảnh")
    print(f"   - Khỏe mạnh (Control) : {len(X0)}")
    print(f"   - Bệnh nhân (Patients) : {len(X1)}")
    print(f"   - Kích thước 1 vector  : {X.shape[1]} chiều (Mỗi bức ảnh được tóm tắt thành {X.shape[1]} con số)")

    # 4. Lưu ra file nhị phân .npy
    # Điểm lợi: File .npy đọc/ghi cực nhanh. Lần sau chạy train.py sẽ load trong chớp mắt 
    # mà không cần phải gọi thuật toán HOG xử lý lại hàng trăm tấm ảnh.
    np.save(os.path.join(SAVE_DIR, 'X_drawing.npy'), X)
    np.save(os.path.join(SAVE_DIR, 'y_drawing.npy'), y)
    print(f"\n💾 Đã lưu thành công: X_drawing.npy và y_drawing.npy")

if __name__ == "__main__":
    main()
