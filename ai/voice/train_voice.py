import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#trỏ đường dẫn đến 2 file được lọc ở preprocess
X_PATH=os.path.join(CURRENT_DIR, 'x_voice.npy')
Y_PATH=os.path.join(CURRENT_DIR, 'y_voice.npy')

def main():
    #check và đọc data từ file wav
    if not os.path.exists(X_PATH):
        print("Chưa có file X_voice.npy hãy chạy preprocess trước")
        return

    #load 2 ma trận dữa liệu vào ram
    x=np.load(X_PATH)
    y=np.load(Y_PATH)
    print(f"Đã đọc data voice: {x.shape[0]} mẫu")

    #chia data theo 60.20.20
    #chia 20% cho tập test
    x_temp, x_test,y_temp, y_test=train_test_split(x,
                                                   y,
                                                   test_size=0.2,
                                                   random_state=42)
    ##chia 20% cho tập validation
    x_train, x_val, y_train, y_val=train_test_split(x_temp,
                                                    y_temp,
                                                    test_size=0.25, 
                                                    random_state=42)

    print("Cấu trúc chia dữ liệu")
    print(f"   Train Set (60%)      : {len(x_train)} mẫu (Dùng để AI học)")
    print(f"   Validation Set (20%) : {len(x_val)} mẫu (Dùng để AI tìm tham số tốt nhất)")
    print(f"   Test Set (20%)       : {len(x_test)} mẫu (Dùng để đánh giá thi thật)")

    #chuẩn hóa dữ liệu
    scaler=StandardScaler()
    #fit_tranform đo đạc trên tập train
    x_train_sc=scaler.fit_transform(x_train)
    x_val_sc=scaler.transform(x_val)
    x_test_sc=scaler.transform(x_test)

    #tìm tham số tốt nhất
    print("\nĐang dùng GridSearch để tìm tham số tốt nhất")
    # Cung cấp các kịch bản để máy tự chạy thử và tìm cấu hình xịn nhất
    param_grid={'n_estimators': [100, 200],         # Số lượng "cây quyết định" trong khu rừng
                   'max_depth': [10, 15, 20, None],    # Độ sâu của cây (khả năng suy luận chi tiết)
                   'min_samples_split': [2, 5]}        # Số lượng mẫu tối thiểu để chẻ nhánh

    #Cấu hình "bộ máy" tự động thi sát hạch
    grid=GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), # Thuật toán: Hồi quy Rừng ngẫu nhiên
        param_grid, 
        cv=3, # Chia Validation ra làm 3 nếp gấp (Cross-validation)
        scoring='neg_mean_absolute_error', # Tiêu chí chọn người chiến thắng: Sai số (Error) thấp nhất
        n_jobs=-1, # Dùng full lõi CPU
        verbose=1)
    grid.fit(x_train_sc,y_train)
    print(f"\n Tham số tốt nhất: {grid.best_params_}")

    #huấn luyện mô hình trên tạp train
    print(f"\nĐang dùng tham số tốt nhất để train")
    model = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    model.fit(x_train_sc, y_train)

    #lưu toàn bộ vào file pkl
    joblib.dump(model,  os.path.join(CURRENT_DIR, 'voice_model.pkl'))
    joblib.dump(scaler, os.path.join(CURRENT_DIR, 'voice_scaler.pkl'))
    # Lưu thêm thứ tự 16 tên đặc trưng để lúc người dùng đưa file .wav vào web, 
    # dữ liệu trích xuất sẽ được sắp xếp đúng hệt như lúc AI đang học
    features=['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    joblib.dump(features, os.path.join(CURRENT_DIR, 'voice_features.pkl'))

    print(f"\nĐã lưu thành công 3 file 'bộ não':")
    print(f"    voice_model.pkl")
    print(f"    voice_scaler.pkl")
    print(f"    voice_features.pkl")
    print("\n   Chạy evaluate.py để xem biểu đồ kết quả!")
if __name__ == "__main__":
    main()


    
