import streamlit as st
import numpy as np
import joblib
import os
import time
import tempfile
import parselmouth
import librosa
from parselmouth.praat import call

#cài đặt cấu hình trang web
st.set_page_config(page_title='Parkinsons Severity Tracker',#tiêu đề
                   page_icon="📈", #icon tiêu đề
                   layout="centered")
#hiển thị tiêu đề 
st.markdown('<h1 style="color:#FF6A3D;text-align:center;">📈 Parkinson\'s Severity Tracker</h1>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Phân tích giọng nói để đo lường mức độ nghiêm trọng (điểm UPDRS)</p>', 
            unsafe_allow_html=True)

#đọc và tải mô hình từ các file pkl
#lấy đường dẫn thư mục tại file train_voice
current_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(current_dir, 'voice_model.pkl')
scaler_path=os.path.join(current_dir, 'voice_scaler.pkl')
feat_path=os.path.join(current_dir, 'voice_features.pkl')

# Hàm cache_resource giúp Streamlit lưu mô hình vào RAM (bộ nhớ tạm), 
# tránh việc mỗi lần người dùng bấm nút lại phải load file nặng từ đầu
@st.cache_resource
def load_tele_model():
    if not os.path.exists(model_path): return None, None, None
    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(feat_path)

#gán 3 file vào biến để tính toán 
model,scaler,features_name=load_tele_model()
if model is None:
    st.error("Chưa có file mô hình. hãy chạy train trước và đảm bảo 3 file cùng nằm cùng thư mục voice")
    st.stop()#dừng hết nếu ko có

#hàm phân tích file âm thanh ddeeer trích xuất 
def extract_16features(wav_path):
    #đọc file âm thanh bằng library parselmouth
    sound=parselmouth.Sound(wav_path)
    pitch=call(sound, "To pitch",0.0,75,600)#tính toán đường cong cao độ giới hạn 75-600hz
    point_process=call(sound,"To PointProcess (periodic, cc)", 75,600)#tính các điểmt tuần hoàn trong giọng nói

    #trích xuất 5 đặt trúng nhóm jitter (độ dao động tần số)
    j_pct=call(point_process, "Get jitter (local)",0,0,0.0001,0.02,1.3)*100
    j_abs=call(point_process, "Get jitter (local, absolute)",0,0,0.0001,0.02,1.3)
    j_rap=call(point_process, "Get jitter (rap)",0,0,0.0001,0.02,1.3)
    j_ppq=call(point_process, "Get jitter (ppq5)",0,0,0.0001,0.02,1.3)
    j_ddp=j_rap*3#ddp được xấp xỉ = rap *3

    #trích xuất 6 đặt trưng nhóm shimmer(dao động biên độ/âm lượng)
    s_loc=call([sound, point_process], "Get shimmer(local)",0,0,0.0001,0.02,1.3,1.6)
    s_db=call([sound, point_process], "Get shimmer (local_db)",0,0,0.0001, 0.02,1.3,1.6)
    s_apq3=call([sound, point_process], "Get shimmer (apq3)",0,0,0.0001,0.02,1.3,1.6)
    s_apq5=call([sound, point_process], "Get shimmer (apq5)",0,0,0.0001,0.02, 1.3,1.6)
    s_apq11=call([sound, point_process], "Get shimmer (apq11)",0,0,0.0001, 0.02,1.3,1.6)
    s_dda=s_apq3*3

    #trích xuất 2 đặc trưng nhóm nhiễu 
    #harmonicity đo tỉ lệ giữa tín hiệu thanh âm và tính hiệu nhiễu 
    harmonicity=call(sound, "To Harmonicity (cc)",0.01,75,0.1,1.0)
    hnr=call(sound, "Get mean",0,0)
    nhr=1.0/(10**(hnr/10)) if hnr>0 else 0#suy ra nhr từ hnr

    #trích xuất 3 đặc trưng phi tuyến tính 
    y_aud,sr=librosa.load(wav_path, sr=None)
    cent=librosa.feature.spectral_centroid(y=y_aud, sr=sr)[0]
    band=librosa.feature.spectral_bandwidth(y=y_aud, sr=sr)[0]

    #dfa xấp xỉ qua tỉ lệ phổ tần số
    dfa=float(np.mean(band)/(np.maen(cent)+1e-6))

    #prde xấp xỉ mức độ hỗn loạn qua độ lệch chuẩn phổ mfcc
    mfccs=librosa.feature.mfcc(y=y_aud, sr=sr, n_mfcc=20)
    rpde=float(np.std(mfccs))

    #ppe tính entropy phân phối xác suất của pitch
    pitch_vals=pitch.selected_array['frequency']
    pitch_vals=pitch_vals[pitch_vals>0]
    if len(pitch_vals)>0:
        hist,_=np.histogram(pitch_vals, bins=20, density=True)
        hist=hist[hist>0]
        ppe=float(-np.sum(hist*np.log(hist+1e-6)))#công thức shannon antropy
    else: ppe=0.5

    #đóng gói 16 kết quả thnahf 1 bộ dict 
    return {'Jitter(%)': j_pct, 'Jitter(Abs)': j_abs, 'Jitter:RAP': j_rap, 
            'Jitter:PPQ5': j_ppq, 'Jitter:DDP': j_ddp,'Shimmer': s_loc, 
            'Shimmer(dB)': s_db, 'Shimmer:APQ3': s_apq3, 'Shimmer:APQ5': s_apq5, 
            'Shimmer:APQ11': s_apq11, 'Shimmer:DDA': s_dda, 'NHR': nhr, 
            'HNR': hnr, 'RPDE': rpde, 'DFA': dfa, 'PPE': ppe}
             
    #xây dựng khung giao diện
st.header("1.Upload Voice Recording")
#nút upload file, chỉ cho phép đuôi wav
upload_file=st.file_uploader("Chọn file giọng nói (.wav)", type=["wav"])

if upload_file:
    #hiển thị thanh trình phst nhạc để nghe lại
    st.audio(upload_file)

#nút bấm trung tâm để bắt đầu 
if st.button("Phân tích điểm số UPDRS",type="primary", use_container_width=True):
    if not upload_file:
        st.warning("Vui lòng tải file lên trước")
        st.stop()

    #tạo một hiệu ứng loading xoay vòng
    with st.spinner("Al đang đo lường sự thay đổi vi mô trong giọng nói"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(upload_file.read())#đổ dữ liệu từ trình duyệt sang file rỗng
            tmp_path=tmp.name#lấy đường đãn cảu file tạm

        try:
            #gọi hàm ở phần này để lấy 16 con số từ file 
            feat=extract_16features(tmp_path)
            #sắp xếp đúng 15 con số theo thứ tự
            x_input=np.array([[feat[col] for col in features_name]])
            #chuẩn hóa scale cho số đó về cùng hệ quy chiếu 
            x_scaled=scaler.transform(x_input)
            predicted_updrs=model.predict(x_scaled)[0]

            st.markdown("---")
            st.header("2. kết quả đo lường (UPDRS)")
            st.info("  UPDRS (Unified Parkinson Disease Rating Scale) là thang điểm chuẩn y tế. Điểm càng cao, mức độ bệnh càng nặng")
            #in điểm số
            st.metric(label="Dự đoán điểm total UPDRS hiện tại: ",value=f"{predicted_updrs:.1f} điểm", delta="Mức độ: Cần bác sĩ đánh giá thêm", delta_color="off")

            #phân loại tình  trạng bệnh
            if predicted_updrs<20:
                st.success("Tình trạng: Giai đoạn rất nhẹ")
            elif predicted_updrs<40:
                st.success("Tình trạng: Giai đoạn trung bình")
            else:
                st.success("Tình trạng: Giai đoạn tiến triển") 

            #in bảng 16 con số ra xem
            st.markdown("---")
            st.subheader("3. 16 đặc trung đã đo đạc")
            st.write(feat)

        finally: os.unlink(tmp_path)


        