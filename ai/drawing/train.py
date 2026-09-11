# ============================================================
# FILE: train.py
# MỤC ĐÍCH: Đọc dữ liệu đã xử lý, chia 60/20/20, dùng GridSearch
#            tìm tham số tốt nhất, train RandomForest, lưu model
# ============================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(CURRENT_DIR, 'X_drawing.npy')
Y_PATH = os.path.join(CURRENT_DIR, 'y_drawing.npy')


def main():
    # BƯỚC 1: KIỂM TRA VÀ ĐỌC DỮ LIỆU
    if not os.path.exists(X_PATH):
        print("Chua co file X_drawing.npy. Hay chay preprocess.py truoc!")
        return

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Da doc du lieu: {X.shape[0]} mau, {X.shape[1]} dac trung/mau")

    # BƯỚC 2: CHIA DỮ LIỆU 60 / 20 / 20
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"Chia du lieu 60 / 20 / 20:")
    print(f"   Train Set      : {len(X_train)} mau")
    print(f"   Validation Set : {len(X_val)} mau")
    print(f"   Test Set       : {len(X_test)} mau")

    # BƯỚC 3: CHUAN HOA
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # BƯỚC 4: GRIDSEARCH
    print("\nDang GridSearch tim tham so tot nhat...")
    param_grid = {
        'n_estimators'    : [100, 200],
        'max_depth'       : [10, 20, None],
        'min_samples_split': [2, 5],
        'max_features'    : ['sqrt', 'log2']
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_val_sc, y_val)
    print(f"\nTham so tot nhat: {grid.best_params_}")

    # BƯỚC 5: TRAIN MÔ HÌNH CUỐI CÙNG
    print("\nDang train mo hinh cuoi cung tren Train Set...")
    model = RandomForestClassifier(
        **grid.best_params_, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_sc, y_train)

    # BƯỚC 6: ĐÁNH GIÁ TRÊN TEST SET
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 50)
    print("KET QUA DANH GIA TREN TEST SET")
    print("=" * 50)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nChi tiet:")
    print(classification_report(y_test, y_pred, target_names=['Khoe manh', 'Parkinson']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # BƯỚC 7: LƯU FILE
    joblib.dump(model,  os.path.join(CURRENT_DIR, 'drawing_model.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'drawing_scaler.pkl'))
    print(f"\nDa luu:")
    print(f"   drawing_model.pkl")
    print(f"   drawing_scaler.pkl")


if __name__ == "__main__":
    main()
