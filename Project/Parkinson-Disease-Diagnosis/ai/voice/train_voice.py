import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#đọc file data
DATA_FILE="D:\\parkinsons_updrs.data"
#kiểm tra xem file có tồn tại ko
if not os.path.exists(DATA_FILE):
    print(f"Không tìm thấy file {DATA_FILE}")
    exit()
#đọc file 
df=pd.read_csv(DATA_FILE)
print(f"Đã đọc {len(df)} mẫu dữ liệu")

#khai báo danh sách thông số âm thanh 
features=['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
x=df[features]#gán biến x là bảng chứa cột đặt trưng
y=df['total_UPDRS']#gán biến y là cột điểm đầu ra

#chia data theo tỉ lệ 6/2/2
#chia test = 20%
x_temp, x_test, y_temp, y_test=train_test_split(x,
                                                y,
                                                test_size=0.2,
                                                random_state=42)

#chia 20% cho validation
x_train, x_val, y_train, y_val=train_test_split(x_temp,
                                                y_temp, 
                                                test_size=0.25,
                                                random_state=42)

print(f"Chia data 60/20/20:")
print(f"   Train set:      {len(x_train)} mẫu")
print(f"   Validation set: {len(x_val)} mẫu")
print(f"   Test set:       {len(x_test)} mẫu\n")

#chuẩn hóa data
scaler=StandardScaler()#đưa các con số về cùng hệ quy chiếu 
x_train_scaled=scaler.fit_transform(x_train)#học, sau đó chuẩn hóa tập train
x_val_scaled=scaler.transform(x_val)
x_test_scaled=scaler.transform(x_test)

#tìm tham số tốt nhất bằng grid trên validation
print(f"Đang dùng validation set để tìm cách học tốt nhất ")
#liệt kê kịch bản tham số muốn máy thử nghiệm 
param_grid={'n_estimators':[100,200],#thử số lượng cây quyết định
            'max_depth':[10,15,20, None],#thử độ sau của cây
            'min_samples_split':[2,5]}#thử số lượng mẫu tối thiereru để chai nhanh 

#setting gridsearchSV
grid_search=GridSearchCV(RandomForestRegressor(random_state=42,
                                               n_jobs=-1),#thuật toán hồi quy rừng ngẫu nhiên
                        param_grid, 
                        cv=3,#cắt data validation ra 3 phần để test cheo và tính trung bình 
                        scoring="neg_mean_absolute_error",#tiêu chí chấm điểm: sai số càng thấp càng tốt
                        n_jobs=-1,#dùng toàn bộ lõi cpu để tính toán
                        verbose=1)
grid_search.fit(x_val_scaled, y_val)
print(f"\n Tham số tốt nhất tìm được: {grid_search.best_params_}\n")

print("Đang train mô hình cuối cùng trên train set")
model=RandomForestRegressor(**grid_search.best_params_,
                            random_state=42,
                            n_jobs=-1)
model.fit(x_train_scaled, y_train)

#đánh giá kết quả trên test
y_pred_val=model.predict(x_val_scaled)
y_pred_test=model.predict(x_test_scaled)

#tính sai số trung bình giữa kết quả dự đoán và kết quả thật 
mae_val=mean_absolute_error(y_val, y_pred_val)
mae_test=mean_absolute_error(y_test, y_pred_test)
r2_test=r2_score(y_test, y_pred_test)#tính độ tin cậy r2 (tỷ lệ %)

print("="*40)
print("Kết quả đánh giá")
print("="*40)
print(f"  Sai số trên Validation : ±{mae_val:.2f} điểm")
print(f"  Sai số trên Test set   : ±{mae_test:.2f} điểm")
print(f"  Độ tin cậy (R2 Score)  : {r2_test*100:.1f} %")

#lưu mô hình thành file 
save_dir=r"c:\Programming_Language\file bài tập .py\PROJECT\parkinson_al\voice"
os.makedirs(save_dir, exist_ok=True)#tự tạo thư mục nếu chưa có

#nối đường dẫn thư mục
model_path=os.path.join(save_dir, 'tele_model.pkl')
scaler_path=os.path.join(save_dir, 'tele_scaler.pkl')
feat_path=os.path.join(save_dir, 'tele_features.pkl')

#dùng thư viện joblib đóng gói, lưu bộ nhớ của al ra file
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(features, feat_path)

print("Đã lưu 3 file vào: {save_dir}")
print("   tele_model.pkl")
print("   tele_scaler.pkl")
print("   tele_features.pkl")
