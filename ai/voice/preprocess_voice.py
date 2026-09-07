import pandas as pd
import numpy as np
import os
#Đường dẫn cứng trỏ tới file Dataset tải từ UCI
DATA_FILE = r"D:\parkinsons_updrs.data"

#lấy đường dẫn của thư mục 
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))

def main():
    print("Đọc dữ liệu voice")
    #kiểm tra xem file data có toofnt tại ko
    if not os.path.exists(DATA_FILE):
        print(f"File không tồn tại {DATA_FILE}")
        return

    #đọc file
    df=pd.read_csv(DATA_FILE)
    print(f"Đã đọc thành công {len(df)} mẫu dữ liệu từ file")
    # Khai báo  16 cột thông số giọng nói cần lấy
    # Đây là những cột sinh học phản ánh độ rung thanh quản (Jitter, Shimmer, HNR...)
    features=['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

    #trích xuất x và y 
    x=df[features].values.astype(np.float32)
    y=df['total_UPDRS'].values.astype(np.float32)
    print(f"   Kích thước tập x: {x.shape[0]} mẫu, {x.shape[1]} đặc trưng")
    print(f"   Kích thước tập y: {y.shape[0]} nhãn (điểm số)")

    #lưu thành file nhị phân
    np.save(os.path.join(CURRENT_DIR, "x_voice.npy"), x)
    np.save(os.path.join(CURRENT_DIR, "y_voice.npy"), y)
    print(f"Đã lưu thành công mảng dữ liệu: x_voice và y_voice")

if __name__ == "__main__":
    main()

