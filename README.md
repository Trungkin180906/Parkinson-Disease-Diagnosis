# Parkinson-Disease-Diagnosis

## Backend tích hợp

Backend FastAPI nằm trong `backend/`: hồ sơ bệnh nhân, phiên sàng lọc Drawing/Gait,
fusion, theo dõi Voice, lịch sử và báo cáo. Xem [hướng dẫn chạy và hợp đồng API](backend/README.md).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m backend
```

Swagger: http://127.0.0.1:8000/docs. API key local được sinh tại `.data/api-key.txt`.
API hoạt động độc lập với model; endpoint suy luận trả 503 nếu chưa có model/manifest
tương thích. Các model chưa được huấn luyện lại trong phần backend này.

Phần dưới là mô tả định hướng AI ban đầu của nhóm; không phải danh sách tính năng
đã hoàn tất. Backend hiện nhận ảnh HOG, dữ liệu cảm biến Gait `.txt` và đặc trưng
Voice; WAV cần bộ trích đặc trưng khớp dữ liệu huấn luyện. Video Gait chưa được hỗ trợ.

## Định hướng AI ban đầu

 Hệ Thống AI Đa Phương Thức Chẩn Đoán & Theo Dõi Bệnh Parkinson
(Multi-modal AI System for Parkinson's Disease Diagnosis & Monitoring)

Một dự án ứng dụng Trí tuệ Nhân tạo (Machine Learning & Deep Learning) toàn diện nhằm hỗ trợ các bác sĩ chẩn đoán sớm và theo dõi sự tiến triển của bệnh nhân Parkinson thông qua 4 phương pháp không xâm lấn: Giọng nói, Bản vẽ xoắn ốc, Chữ viết tay, và Dáng đi.

Dự án hướng tới việc cung cấp một bộ công cụ y tế từ xa (Telehealth) hoàn chỉnh, trực quan và chính xác.

Tổng Quan Dự Án (The 3 Modules)
Dự án được chia thành 4 phân hệ (module) phân tích độc lập:

1.  Phân Tích Giọng Nói (Voice Analysis - Telemonitoring)
Mục tiêu: Dự đoán mức độ nghiêm trọng của bệnh (Chấm điểm UPDRS).
Cách hoạt động: Bệnh nhân thu âm âm 'A' kéo dài. Hệ thống trích xuất 16 đặc trưng y sinh (Jitter, Shimmer, HNR, PPE...) bằng Praat/Parselmouth.
Mô hình: Random Forest Regressor (có sử dụng GridSearchCV tối ưu hóa).
2. Phân Tích Vẽ Xoắn Ốc (Spiral Drawing Analysis)
Mục tiêu: Phát hiện triệu chứng run tay vi mô (Micrographia/Tremor).
Cách hoạt động: Bệnh nhân đồ lại một hình xoắn ốc trên giấy hoặc tablet. Hệ thống sử dụng Computer Vision để phân tích độ chệch nét vẽ, lực ấn và vận tốc nét bút.
Mô hình: (Mô hình phân loại hình ảnh / Convolutional Neural Networks - CNN).
3. Phân Tích Dáng Đi (Gait Analysis)
Mục tiêu: Nhận diện sự mất thăng bằng và triệu chứng "đóng băng dáng đi" (Freezing of Gait).
Cách hoạt động: Phân tích video dáng đi của bệnh nhân sử dụng công nghệ ước lượng điểm neo cơ thể (Pose Estimation).
Mô hình: (MediaPipe / YOLO Pose kết hợp LSTM RNN).

Dự án là sự kết hợp của nhiều công nghệ AI hiện đại nhất:
Ngôn ngữ: Python 3.12
Giao diện Web (UI): Streamlit
Xử lý âm thanh: Praat (Parselmouth), Librosa
Xử lý ảnh & Video (Computer Vision): OpenCV, MediaPipe
Machine Learning & Deep Learning: Scikit-Learn, Pandas, Numpy (có khả năng mở rộng với TensorFlow/PyTorch)
