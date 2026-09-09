import os
import numpy as np
import pandas as pd

# --- CẤU HÌNH ĐƯỜNG DẪN TUYỆT ĐỐI KHÔNG SỢ LỖI TERMINAL ---
# 1. Lấy vị trí thư mục hiện tại của file preprocess.py (đang ở trong ai/gait)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Lùi lại 2 bước (ra khỏi 'gait', ra khỏi 'ai') để về lại thư mục gốc 'Parkinson-Disease-Diagnosis'
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# 3. Nối đường dẫn chuẩn xác vào thư mục dataset
CONTROL_DIR  = os.path.join(PROJECT_ROOT, "dataset", "gait", "control")
PATIENTS_DIR = os.path.join(PROJECT_ROOT, "dataset", "gait", "parkinson")

# 4. Nơi lưu file X_gait.npy và y_gait.npy (lưu luôn ở thư mục ai/gait)
SAVE_DIR = CURRENT_DIR

def extract_features_from_file(file_path):
    """
    Đọc file .txt của PhysioNet và rút trích đặc trưng.
    Thử thách: Xử lý file nhiễu hoặc thiếu dòng.
    """
    try:
        # Đọc dữ liệu, bỏ qua các dòng lỗi
        df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
        
        # Cột 16 là Tổng lực chân trái (Total VGRF Left)
        # Cột 17 là Tổng lực chân phải (Total VGRF Right)
        total_left = df.iloc[:, 16]
        total_right = df.iloc[:, 17]
        
        # 1. Tính lực trung bình
        mean_l = total_left.mean()
        mean_r = total_right.mean()
        
        # 2. Tính độ lệch chuẩn (phản ánh độ chập chững / run)
        std_l = total_left.std()
        std_r = total_right.std()
        
        # 3. Tính độ bất đối xứng (Asymmetry) - Cực quan trọng
        asymmetry = abs(mean_l - mean_r)
        
        # Trả về 1 vector gồm 5 con số đại diện cho cả quá trình đi bộ
        return [mean_l, mean_r, std_l, std_r, asymmetry]
        
    except Exception as e:
        print(f"⚠️ Lỗi đọc file {file_path}: {e}")
        return None

def process_folder(folder_path, label):
    X, y = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"Đang xử lý {len(files)} file từ {folder_path}...")
    
    for fname in files:
        file_path = os.path.join(folder_path, fname)
        features = extract_features_from_file(file_path)
        if features is not None:
            X.append(features)
            y.append(label)
            
    return X, y

def main():
    print("BƯỚC 1: RÚT TRÍCH ĐẶC TRƯNG TỪ CẢM BIẾN DÁNG ĐI")
    X0, y0 = process_folder(CONTROL_DIR, label=0)
    X1, y1 = process_folder(PATIENTS_DIR, label=1)
    
    X = np.array(X0 + X1, dtype=np.float32)
    y = np.array(y0 + y1, dtype=np.int32)
    
    np.save(os.path.join(SAVE_DIR, 'X_gait.npy'), X)
    np.save(os.path.join(SAVE_DIR, 'y_gait.npy'), y)
    print(f"✅ Đã lưu xong dữ liệu: {X.shape[0]} mẫu, mỗi mẫu có {X.shape[1]} đặc trưng.")

if __name__ == "__main__":
    main()