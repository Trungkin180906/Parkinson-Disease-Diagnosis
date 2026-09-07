# ============================================================
# FILE: train_voice.py
# MỤC ĐÍCH: Đọc dữ liệu .npy, chia 60/20/20, dùng GridSearch
#            train RandomForestRegressor, lưu model.
# ============================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(CURRENT_DIR, 'X_voice.npy')
Y_PATH = os.path.join(CURRENT_DIR, 'y_voice.npy')

def main():
    # BƯỚC 1: KIỂM TRA VÀ ĐỌC DỮ LIỆU
    if not os.path.exists(X_PATH):
        print("❌ Chưa có file X_voice.npy. Hãy chạy preprocess_voice.py trước!")
        return

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"✅ Đã đọc dữ liệu Voice: {X.shape[0]} mẫu.")

    # BƯỚC 2: CHIA DỮ LIỆU 60 / 20 / 20
    # Vì là Regression (y là số thực) nên KHÔNG dùng stratify như Classification
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print(f"\n✂️ Chia dữ liệu 60 / 20 / 20:")
    print(f"   Train Set      : {len(X_train)} mẫu")
    print(f"   Validation Set : {len(X_val)} mẫu")
    print(f"   Test Set       : {len(X_test)} mẫu")

    # BƯỚC 3: CHUẨN HÓA DỮ LIỆU
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # BƯỚC 4: GRIDSEARCH TRÊN VALIDATION SET
    print("\n🔍 Đang GridSearch tìm tham số tốt nhất...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
    )
    grid.fit(X_val_sc, y_val)
    print(f"\n✅ Tham số tốt nhất: {grid.best_params_}")

    # BƯỚC 5: TRAIN MÔ HÌNH CUỐI CÙNG TRÊN TRAIN SET
    print("\n🤖 Đang train mô hình cuối cùng trên Train Set...")
    model = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    # BƯỚC 6: LƯU FILE MODEL & SCALER
    # Lưu bằng tên chuẩn hóa mới
    joblib.dump(model,  os.path.join(CURRENT_DIR, 'voice_model.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'voice_scaler.pkl'))

    # Lưu lại list danh sách đặc trưng để App gọi cho đúng thứ tự
    features = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    joblib.dump(features, os.path.join(CURRENT_DIR, 'voice_features.pkl'))

    print(f"\n💾 Đã lưu thành công:")
    print(f"   ✅ voice_model.pkl")
    print(f"   ✅ voice_scaler.pkl")
    print(f"   ✅ voice_features.pkl")
    print("\n👉 Chạy evaluate_voice.py để xem biểu đồ kết quả!")

if __name__ == "__main__":
    main()
