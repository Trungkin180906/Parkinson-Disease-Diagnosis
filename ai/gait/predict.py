import pandas as pd
import numpy as np
import joblib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(CURRENT_DIR, 'gait_model.pkl')
SCALER_PATH = os.path.join(CURRENT_DIR, 'gait_scaler.pkl')

def predict_gait_risk(file_path):
    # Load model
    if not os.path.exists(MODEL_PATH):
        return {"error": "Chưa có model. Hãy chạy train.py trước!"}
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    try:
        # Đọc file test mới
        df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
        total_left = df.iloc[:, 16]
        total_right = df.iloc[:, 17]
        
        # Bắt buộc phải rút trích đặc trưng y hệt như lúc train
        features = [
            total_left.mean(), total_right.mean(),
            total_left.std(), total_right.std(),
            abs(total_left.mean() - total_right.mean())
        ]
        
        # Scale & Dự đoán
        features_sc = scaler.transform([features])
        proba = model.predict_proba(features_sc)[0]
        
        risk_score = round(proba[1] * 100, 2)
        return {
            "model": "gait",
            "risk_score": risk_score,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

# Test thử nếu chạy file trực tiếp
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = predict_gait_risk(sys.argv[1])
        print(result)