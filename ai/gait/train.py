import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Lấy đường dẫn tuyệt đối của thư mục ai/gait/ hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 50)
    print("🤖 BƯỚC 2: HUẤN LUYỆN MÔ HÌNH DÁNG ĐI (GAIT)")
    print("=" * 50)
    
    # 1. Load dữ liệu đã xử lý
    try:
        X = np.load(os.path.join(CURRENT_DIR, 'X_gait.npy'))
        y = np.load(os.path.join(CURRENT_DIR, 'y_gait.npy'))
        print(f"✅ Đã load dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng.")
    except FileNotFoundError:
        print("❌ Thiếu file dữ liệu. Hãy chạy preprocess.py trước!")
        return

    # 2. Chia tập Train/Test (80% để học, 20% để thi thử)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Chuẩn hóa số liệu (đưa các con số về cùng hệ quy chiếu)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # 4. Huấn luyện mô hình
    print("⏳ Đang huấn luyện mô hình Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_sc, y_train)
    
    # 5. Thi thử và chấm điểm
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Độ chính xác trên tập Test: {acc * 100:.2f}%")
    
    print("\n📊 Báo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Khỏe mạnh (0)', 'Parkinson (1)']))
    
    # 6. Lưu lại "não" của AI
    joblib.dump(model, os.path.join(CURRENT_DIR, 'gait_model.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'gait_scaler.pkl'))
    print("\n💾 Đã xuất xưởng thành công: gait_model.pkl và gait_scaler.pkl")

if __name__ == "__main__":
    main()