import numpy as np
import joblib 
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
def main():
    #tải tài nguyên 
    model_path=os.path.join(CURRENT_DIR, "voice_model.pkl")
    scaler_path=os.path.join(CURRENT_DIR, "voice_scaler.pkl")
    x_path=os.path.join(CURRENT_DIR, "x_voice.npy")
    y_path=os.path.join(CURRENT_DIR, "y_voice.npy")

    #báo lỗi quên chạy 2 bước trước
    for p in [model_path, scaler_path, x_path, y_path]:
        if not os.path.exists(p):
            print(f"Thiếu file: {p}. Hãy chạy train_voice.py trước!")
            return

    #load file vào ram
    model=joblib.load(model_path)
    scaler=joblib.load(scaler_path)
    X=np.load(x_path)
    y=np.load(y_path)

    #gọi lại tập tets
    x_temp, x_test, y_temp, y_test = train_test_split(X,
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=42)
    #chuẩn hóa tập test
    x_test_sc=scaler.transform(x_test)
    # Yêu cầu AI dự đoán ra điểm số y_pred từ bộ câu hỏi X_test_sc
    y_pred = model.predict(x_test_sc)

    # Đo khoảng cách giữa Đáp án thật (y_test) và Câu trả lời của AI (y_pred)
    mae = mean_absolute_error(y_test, y_pred) # Sai số trung bình (Điểm lệch)
    r2  = r2_score(y_test, y_pred)# R2 (Độ phù hợp của đường dự đoán)

    print("\n=" * 55)
    print("KẾT QUẢ ĐÁNH GIÁ CHI TIẾT - VOICE ANALYSIS MODEL")
    print("=" * 55)
    print(f" Sai số trung bình (MAE) :±{mae:.2f} điểm UPDRS")
    print(f" Độ tin cậy (R2 Score)   :{r2 * 100:.2f}%")

    #vẽ biểu đồ phân tán 
    plt.figure(figsize=(8, 6))
    # Chấm các điểm dự đoán lên trục tọa độ (Trục X: Thực tế, Trục Y: Dự đoán)
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Dự đoán')

     # Vẽ đường thẳng đường chéo màu đỏ (Đường y = x)
    # Nếu AI dự đoán đúng 100% (Sai số = 0), tất cả điểm chấm xanh sẽ nằm đè lên đường đỏ này.
    min_val=min(min(y_test), min(y_pred))
    max_val=max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Đường hoàn hảo')

     #Trang trí tiêu đề biểu đồ
    plt.title('Dự đoán điểm UPDRS vs Thực tế (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Điểm UPDRS thực tế', fontsize=12)
    plt.ylabel('Điểm UPDRS mô hình dự đoán', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
     #Xuất biểu đồ ra file hình ảnh (.png) 
    save_path = os.path.join(CURRENT_DIR, 'evaluation_voice_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    print(f"\nĐã lưu biểu đồ phân tán: evaluation_voice_result.png")
    plt.show()
if __name__ == "__main__":
    main()    


