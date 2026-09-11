# ============================================================
# FILE: train_voice.py
# MỤC ĐÍCH: Học máy (Machine Learning). Chia dữ liệu 60/20/20,
#            sử dụng GridSearch tối ưu tham số và huấn luyện 
#            mô hình RandomForestRegressor (Dự đoán điểm số).
# ============================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Trỏ đường dẫn đến 2 file đã được lọc ở bước preprocess
X_PATH = os.path.join(CURRENT_DIR, 'X_voice.npy')
Y_PATH = os.path.join(CURRENT_DIR, 'y_voice.npy')

def main():
    # -------------------------------------------------------
    # BƯỚC 1: KIỂM TRA VÀ ĐỌC DỮ LIỆU TỪ FILE .NPY
    # -------------------------------------------------------
    if not os.path.exists(X_PATH):
        print("❌ Chưa có file X_voice.npy. Hãy chạy preprocess_voice.py trước!")
        return

    # Load 2 ma trận dữ liệu vào RAM
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"✅ Đã đọc dữ liệu Voice: {X.shape[0]} mẫu.")

    # -------------------------------------------------------
    # BƯỚC 2: CHIA DỮ LIỆU THEO CHIẾN LƯỢC 60% / 20% / 20%
    # -------------------------------------------------------
    # Vì bài toán là Regression (Dự đoán 1 con số cụ thể, VD: 28.5 điểm) 
    # nên ta KHÔNG dùng tham số 'stratify' (tham số này chỉ dùng cho Phân loại Có/Không).
    
    # Cắt 20% làm tập Test (Để dành cuối cùng thi thật)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Từ 80% (temp) còn lại, cắt 25% làm tập Validation (= 20% của tổng số)
    # Phần còn lại 75% của temp (= 60% của tổng số) làm tập Train
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print(f"\n✂️ Cấu trúc chia dữ liệu:")
    print(f"   Train Set (60%)      : {len(X_train)} mẫu (Dùng để AI học)")
    print(f"   Validation Set (20%) : {len(X_val)} mẫu (Dùng để AI tìm tham số tốt nhất)")
    print(f"   Test Set (20%)       : {len(X_test)} mẫu (Dùng để đánh giá thi thật)")

    # -------------------------------------------------------
    # BƯỚC 3: CHUẨN HÓA DỮ LIỆU (STANDARDIZATION)
    # -------------------------------------------------------
    # Tính chất: Thu hẹp độ trải dài của các con số (VD: từ 0-1000 và 0-0.1 về chung một dải)
    scaler = StandardScaler()
    
    # Lệnh fit_transform: Bắt Scaler ĐO ĐẠC trên tập Train rồi tiến hành co giãn tập Train
    X_train_sc = scaler.fit_transform(X_train)
    # Lệnh transform: Đối với Val và Test, chỉ được dùng thước đo ĐÃ ĐO từ Train để co giãn.
    # Tuyệt đối không được 'fit' lại trên Test để tránh rò rỉ dữ liệu (Data Leakage)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # -------------------------------------------------------
    # BƯỚC 4: TÌM KIẾM THAM SỐ TỐT NHẤT (HYPERPARAMETER TUNING)
    # -------------------------------------------------------
    print("\n🔍 Đang dùng GridSearch để rà soát siêu tham số...")
    # Cung cấp các kịch bản để máy tự chạy thử và tìm cấu hình xịn nhất
    param_grid = {
        'n_estimators': [100, 200],         # Số lượng "cây quyết định" trong khu rừng
        'max_depth': [10, 15, 20, None],    # Độ sâu của cây (khả năng suy luận chi tiết)
        'min_samples_split': [2, 5]         # Số lượng mẫu tối thiểu để chẻ nhánh
    }
    
    # Cấu hình "bộ máy" tự động thi sát hạch
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), # Thuật toán: Hồi quy Rừng ngẫu nhiên
        param_grid, 
        cv=3, # Chia Validation ra làm 3 nếp gấp (Cross-validation)
        scoring='neg_mean_absolute_error', # Tiêu chí chọn người chiến thắng: Sai số (Error) thấp nhất
        n_jobs=-1, # Dùng full lõi CPU
        verbose=1
    )
    
    # Chạy thi sát hạch trên tập Train (GridSearchCV tự chia Validation bên trong bằng cv=3)
    grid.fit(X_train_sc, y_train)
    print(f"\n✅ Cấu hình tham số chiến thắng: {grid.best_params_}")

    # -------------------------------------------------------
    # BƯỚC 5: HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG TRÊN TẬP TRAIN
    # -------------------------------------------------------
    print("\n🤖 Đang dùng cấu hình chiến thắng để học cẩn thận trên tập Train...")
    # Khởi tạo mô hình với dấu ** (giải nén dictionary tham số chiến thắng)
    model = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    # -------------------------------------------------------
    # BƯỚC 6: LƯU TOÀN BỘ KẾT QUẢ VÀO FILE .PKL ĐỂ XÀI TRÊN WEB
    # -------------------------------------------------------
    joblib.dump(model,  os.path.join(CURRENT_DIR, 'voice_model.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'voice_scaler.pkl'))

    # Lưu thêm thứ tự 16 tên đặc trưng để lúc người dùng đưa file .wav vào web, 
    # dữ liệu trích xuất sẽ được sắp xếp đúng hệt như lúc AI đang học
    features = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    joblib.dump(features, os.path.join(CURRENT_DIR, 'voice_features.pkl'))

    print(f"\n💾 Đã lưu thành công 3 file 'bộ não':")
    print(f"   ✅ voice_model.pkl")
    print(f"   ✅ voice_scaler.pkl")
    print(f"   ✅ voice_features.pkl")
    print("\n👉 Bước tiếp theo: Chạy evaluate_voice.py để xem biểu đồ kết quả!")

if __name__ == "__main__":
    main()
