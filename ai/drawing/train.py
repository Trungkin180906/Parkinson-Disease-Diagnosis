import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X = np.load(os.path.join(BASE_DIR, "X_drawing.npy"))
y = np.load(os.path.join(BASE_DIR, "y_drawing.npy"))

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# Đánh giá
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Drawing Model")
print("Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, y_pred))

# Lưu model + scaler
joblib.dump(
    model,
    os.path.join(BASE_DIR, "drawing_model.pkl")
)

joblib.dump(
    scaler,
    os.path.join(BASE_DIR, "drawing_scaler.pkl")
)

print("Đã lưu drawing_model.pkl")
print("Đã lưu drawing_scaler.pkl")