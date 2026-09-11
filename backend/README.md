# Backend tích hợp hệ thống Parkinson

Backend cung cấp API cho 9 màn hình đã mô tả: hồ sơ bệnh nhân, lựa chọn luồng,
sàng lọc Drawing/Gait, tổng hợp nguy cơ, theo dõi Voice, lịch sử và báo cáo.
Ứng dụng FastAPI lưu dữ liệu bằng SQLite; không cần cài một database server riêng.
Các trang Streamlit hiện tại vẫn là demo độc lập. Dùng client ở cuối tài liệu để
nối frontend mới vào API; backend không tự chuyển giao diện của các thành viên khác.

## Chạy trên Windows

Tại thư mục gốc repository, dùng Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m backend
```

`requirements-lock.txt` cố định toàn bộ môi trường đã kiểm thử, gồm công cụ phát triển.
`requirements-backend.txt` chỉ khai báo dependency runtime; `requirements-dev.txt`
bổ sung test và lint. Nếu model của nhóm được xuất bằng sklearn khác phiên bản,
dựng môi trường tương ứng và kiểm thử lại, hoặc huấn luyện/xuất lại model bằng môi
trường đã thống nhất. Không bỏ qua cảnh báo không tương thích phiên bản pickle.

- API chạy tại `http://127.0.0.1:8000`.
- Swagger tại `http://127.0.0.1:8000/docs`; schema tại `/openapi.json`.
- `GET /health` kiểm tra server/database, không yêu cầu key.
- Lần khởi động đầu tạo database `.data/parkinson.sqlite3` và key `.data/api-key.txt`.
  Đọc key trên máy, nhập vào nút **Authorize** của Swagger. Header của API là
  `Authorization: Bearer <key>`. Không commit key hoặc database.
- Mọi endpoint `/api/v1` dùng key, kể cả đọc hồ sơ và tải báo cáo. Đây là key dùng
  cho nhân viên/ứng dụng nội bộ, chưa phải hệ thống tài khoản bệnh nhân với phân quyền.
  Với Streamlit, giữ key phía server trong `st.secrets`; không nhúng key dùng chung
  vào mã JavaScript của một website công khai.

Chạy kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check backend tests
.\.venv\Scripts\ruff.exe format --check backend tests
```

## Cấu hình

Cấu hình được đọc từ biến môi trường khi ứng dụng khởi tạo. Không tự đọc file `.env`.
Ví dụ PowerShell:

```powershell
$env:PD_MONITORING_POLICY = 'confirmed_only'
$env:PD_DRAWING_WEIGHT = '0.6'
.\.venv\Scripts\python.exe -m backend
```

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PD_DATA_DIR` | `.data` tại gốc repo | Database và API key sinh tự động |
| `PD_MODEL_DIR` | `ai` tại gốc repo | Thư mục chứa drawing/gait/voice |
| `PD_API_KEY` | Tự sinh và lưu local | Key tối thiểu 16 ký tự nếu tự đặt |
| `PD_CORS_ORIGINS` | localhost:8501 và 127.0.0.1:8501, giao thức HTTP | Danh sách origin, phân cách bằng dấu phẩy |
| `PD_DRAWING_WEIGHT` | `0.5` | Trọng số Drawing; Gait nhận `1 - trọng số Drawing` |
| `PD_REFERRAL_THRESHOLD` | `70` | Ngưỡng đề nghị bác sĩ đánh giá, bao gồm dấu bằng |
| `PD_MONITORING_POLICY` | `at_risk_or_confirmed` | Theo bản mô tả giao diện mới nhất; có thể đổi thành `confirmed_only` theo outline ban đầu |
| `PD_VOICE_REFERENCE_MAX` | `100` | Tham số quy đổi Voice Score của dự án |
| `PD_VOICE_CHANGE_THRESHOLD` | `5` | Ngưỡng thay đổi UPDRS dự đoán giữa hai lần, phục vụ demo xu hướng |
| `PD_MAX_UPLOAD_BYTES` | `26214400` | Giới hạn 25 MiB mỗi file |

`GET /api/v1/config` trả các quy tắc hiển thị cho frontend, không trả key hay đường dẫn riêng tư.

## Các quyết định nghiệp vụ đã triển khai

**Hồ sơ và lần khám.** Mỗi bệnh nhân có UUID và mã `PD-...` tự sinh. Tuổi, giới tính,
chiều cao, cân nặng thuộc hồ sơ; ngày thực hiện thuộc từng phiên sàng lọc hoặc ghi âm.
API tạo hồ sơ không nhận trạng thái chẩn đoán. Ghi nhận chẩn đoán dùng endpoint riêng,
với tên người đánh giá, ngày đánh giá và lịch sử thay đổi. Trường này là thông tin do
người vận hành nhập; backend không xác minh danh tính bác sĩ. Cập nhật hồ sơ/chẩn đoán
phải gửi `version` hiện tại để phát hiện hai người sửa đè nhau.

**Sàng lọc.** Mỗi phiên có tối đa một kết quả Drawing và một kết quả Gait.
Trạng thái chuyển `pending → partial → completed`. Chỉ tính tổng hợp khi đủ hai kết quả:

```text
overall_risk = round(drawing_weight * drawing_risk + (1 - drawing_weight) * gait_risk, 2)
```

Khoảng phân loại liên tục: `[0,30]` thấp, `(30,60]` trung bình, `(60,80]` cao,
`(80,100]` rất cao. Mức cảnh báo và ngưỡng đề nghị đi khám là hai quy tắc khác nhau.
Ngưỡng và trọng số được chụp lại khi tạo phiên; thay đổi cấu hình không sửa lịch sử.
Kết quả đã lưu không ghi đè: yêu cầu lặp trả 409; muốn phân tích lại thì tạo phiên mới.
Gửi Drawing/Gait cùng lúc vẫn hoàn tất đúng một phiên nhờ transaction SQLite.

**Điều kiện Voice.** Theo mặc định, cho phép nếu bác sĩ đã xác nhận, hoặc phiên
sàng lọc hoàn tất gần nhất theo ngày thực hiện có risk đạt ngưỡng đã lưu của phiên đó.
`ruled_out` chặn theo dõi theo nguy cơ. `confirmed_only` chỉ cho phép sau ghi nhận
xác nhận. Điểm AI không tự cập nhật trạng thái chẩn đoán. Kiểm tra điều kiện ở thời
điểm gửi kết quả; lưu lại căn cứ cho phép vào bản ghi Voice.

**Voice Score.** Hai tài liệu chưa định nghĩa công thức, nên backend dùng quy ước
hiển thị có cấu hình `clamp(100 * (1 - total_updrs / reference_max), 0, 100)`.
Mặc định UPDRS 27.6 cho Voice Score 72.4. Đây là lựa chọn triển khai để frontend có
chỉ số nhất quán, không phải công thức lâm sàng hay độ tin cậy mô hình; cũng không
được diễn giải là xác suất mắc bệnh. Đổi cấu hình nếu nhóm thống nhất công thức khác.

**Xu hướng.** So sánh UPDRS dự đoán, không suy luận giai đoạn bệnh hoặc tự quyết định
tái khám. `baseline` là lần đầu; `increased/decreased/stable` dựa trên chênh lệch
và ngưỡng của bản ghi hiện tại. Ghi âm nhập ngược thứ tự vẫn được sắp xếp lại theo
thời gian thực tế. Hai bản ghi khác fingerprint model, schema hoặc tham số quy đổi
được đánh dấu `not_comparable`, không hiển thị một thay đổi giả do đổi model.
Mỗi bệnh nhân chỉ có một bản ghi tại cùng thời điểm UTC.

## Hợp đồng API

Các đường dẫn dưới đây bắt đầu bằng `/api/v1`. Swagger có schema request/response.
UUID lấy từ phản hồi tạo hồ sơ/phiên. Tất cả ngày dạng `YYYY-MM-DD`, thời điểm ghi âm
phải có múi giờ như `2026-09-11T09:00:00+07:00`; thời điểm tương lai bị từ chối.

| Method | Đường dẫn | Dùng cho |
|---|---|---|
| GET | `/config`, `/models` | Trang chủ, kiểm tra khả năng phân tích |
| POST / GET | `/patients` | Tạo / tìm hồ sơ (`q`, `limit`, `offset`) |
| GET / PUT | `/patients/{patient_id}` | Xem / cập nhật toàn bộ thông tin cơ bản |
| POST | `/patients/{patient_id}/diagnoses` | Ghi nhận đánh giá của bác sĩ |
| POST / GET | `/patients/{patient_id}/screenings` | Tạo / liệt kê phiên sàng lọc |
| GET | `/screenings/{screening_id}` | Kết quả và trạng thái tổng hợp |
| POST | `/screenings/{screening_id}/drawing/file` | Ảnh JPEG/PNG/BMP cho model HOG |
| POST | `/screenings/{screening_id}/gait/file` | Dữ liệu cảm biến 19 cột `.txt` |
| POST | `/screenings/{screening_id}/{drawing hoặc gait}/features` | Đặc trưng đúng schema đã huấn luyện |
| POST | `/patients/{patient_id}/voice/features` | 16 đặc trưng âm thanh → UPDRS/Voice Score |
| POST | `/patients/{patient_id}/voice/file` | WAV, khi đã cấu hình extractor tương thích |
| GET | `/patients/{patient_id}/voice` | Lịch sử theo thời gian, dữ liệu để vẽ biểu đồ |
| GET | `/patients/{patient_id}/voice.csv` | Xuất lịch sử CSV |
| GET | `/patients/{patient_id}/report` | Báo cáo tổng hợp JSON |
| GET | `/patients/{patient_id}/report.html` | Báo cáo HTML tải xuống, mở và in từ trình duyệt |

Tạo bệnh nhân:

```json
{
  "full_name": "Người dùng demo",
  "age": 65,
  "sex": "female",
  "height_cm": 160,
  "weight_kg": 55,
  "phone": null
}
```

Tạo phiên: gửi `{}` để lấy ngày hiện tại, hoặc `{"performed_on":"2026-09-11","notes":""}`.
Gửi file dạng `multipart/form-data`, tên trường `file`. Voice thêm `recorded_at` và
`notes` tùy chọn. Backend đọc nội dung thật, không tin phần mở rộng/tên file;
file không được dùng làm đường dẫn trên máy chủ.

API đặc trưng nhận `feature_schema` và `values` theo đúng thứ tự ở phần dưới.
Voice thêm `recorded_at` và `notes`. API không nhận risk/UPDRS do trình duyệt tự tính:
server thực hiện suy luận và lưu model version + SHA256 của đầu vào. File gốc và
vector đặc trưng không được lưu dài hạn; database lưu kết quả và nguồn gốc để truy vết.

Lỗi có dạng:

```json
{"error":{"code":"model_unavailable","message":"Chưa cấu hình model."}}
```

Mã HTTP: 401 key không hợp lệ; 404 không tìm thấy; 409 xung đột/không đủ điều kiện;
413 quá dung lượng; 422 đầu vào sai; 503 model/extractor/database chưa sẵn sàng.
Validation có thêm `error.details` chứa tên trường và lỗi, không phản chiếu dữ liệu bệnh nhân.
Khi nhận lỗi model, frontend giữ phiên đang `pending/partial` và cho phép thử lại sau
khi nhóm sửa model. Không có kết quả giả được lưu thay cho model lỗi.

## Kết nối model của nhóm

Backend **không tự tải model từ Internet, không huấn luyện lại, không nhập UI Streamlit**.
Nhánh `main` hiện thiếu artifact; Gait còn thiếu mã train; model Drawing ở nhánh
feature dùng schema 9 cột khác HOG. Các điều kiện này được biểu diễn qua `/models`.
Hồ sơ, lịch sử, báo cáo và Swagger vẫn chạy được khi cả ba model chưa sẵn sàng.

Mỗi thư mục `ai/drawing`, `ai/gait`, `ai/voice` cần:

1. Model và scaler được huấn luyện, xuất bằng phiên bản sklearn tương thích môi trường.
2. File `backend-model.json`, chọn mẫu trong `backend/model-examples/` phù hợp với
   dữ liệu thực sự đã dùng khi train. Các mẫu không được kích hoạt tự động.
3. Model phải có đúng số đặc trưng, đúng loại classifier/regressor và nhãn dương.
   Chỉ nạp pickle/joblib do nhóm kiểm soát; không có endpoint upload model từ người dùng.

```powershell
# Chỉ làm sau khi đã có model HOG 8100 đặc trưng, không áp dụng cho model 9 cột.
Copy-Item backend/model-examples/drawing-hog.json ai/drawing/backend-model.json
.\.venv\Scripts\python.exe -m backend.check_models
```

Lệnh kiểm tra trả exit code 1 nếu bất kỳ model chưa sẵn sàng. Fingerprint lưu trong
kết quả kết hợp phiên bản khai báo với SHA256 của manifest/model/scaler; thay file
model làm mất cache và đổi fingerprint, tránh so sánh nhầm phiên bản.

| Schema | Số đặc trưng | Đầu vào hỗ trợ |
|---|---:|---|
| `drawing.hog.v1` | 8100 | Ảnh hoặc vector; grayscale → resize 128×128 → HOG 9 hướng, cell 8×8, block 2×2, L2-Hys |
| `drawing.handpd.v1` | 9 | Chỉ vector 9 cột 3–11 zero-based của CSV đã train; không thay bằng 9 thống kê pixel |
| `gait.vgrf.v1` | 5 | TXT hoặc vector; mean trái, mean phải, std trái, std phải, chênh mean tuyệt đối |
| `voice.uci16.v1` | 16 | Vector; WAV nếu cấu hình extractor đã kiểm chứng |

Gait dùng tổng lực cột **18/19**, zero-based **17/18**, `std` với `ddof=1`. File có
19 cột số hữu hạn, ít nhất 10 dòng, thời gian tăng dần và lực không âm. Model cũ
train theo cột 16/17 của nhánh feature phải được train lại với cách trích đúng này;
không chỉ thêm manifest vào model cũ. Video/Pose Estimation nằm ngoài adapter hiện có.

Thứ tự Voice:

```text
Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP,
Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA,
NHR, HNR, RPDE, DFA, PPE
```

Không sử dụng các phép xấp xỉ RPDE/DFA/PPE và cách tính HNR đang lỗi trong
`ui/voice_app.py`. Để bật WAV, thành viên phụ trách Voice cung cấp hàm Python
trích **cùng định nghĩa và đơn vị** với dataset, trả dictionary đủ 16 tên trên:

```python
# Ví dụ giao diện hàm cho module do nhóm Voice triển khai.
def extract_uci16(wav_bytes: bytes) -> dict[str, float]:
    # Phải dùng thuật toán đã kiểm chứng với tập huấn luyện.
    ...
```

Sau đó đặt `"wav_extractor": "ai.voice.extractor:extract_uci16"` trong manifest.
WAV được kiểm tra PCM, 1–2 kênh, 8–96 kHz và 1–120 giây trước khi gọi hàm.
Không cấu hình hàm này thì API WAV trả `503 extractor_unavailable`; API vector vẫn
dự đoán được khi model/scaler sẵn sàng. Backend không tuyên bố phần AI còn thiếu đã hoàn tất.

## Kết nối Streamlit

Client Python chỉ dùng thư viện chuẩn, ở `backend/client.py`. Đặt API key trong
`.streamlit/secrets.toml` của frontend, ví dụ trường `backend_api_key`. Không đọc trực
tiếp database và không đưa model vào trang UI:

```python
import streamlit as st
from backend.client import BackendClient, BackendError

api = BackendClient(st.secrets["backend_api_key"])

# Sau khi người dùng gửi form bệnh nhân:
patient = api.request("POST", "/api/v1/patients", {
    "full_name": full_name, "age": age, "sex": sex,
    "height_cm": height_cm, "weight_kg": weight_kg, "phone": phone or None,
})
st.session_state["patient_id"] = patient["id"]

# Khi bắt đầu sàng lọc; chỉ tạo một lần, giữ id trong session_state.
screening = api.request("POST", f"/api/v1/patients/{patient['id']}/screenings", {})
st.session_state["screening_id"] = screening["id"]

# Khi bấm nút phân tích Drawing:
try:
    result = api.upload(
        f"/api/v1/screenings/{st.session_state['screening_id']}/drawing/file",
        uploaded_file.getvalue(),
    )
except BackendError as exc:
    st.error(str(exc))
```

Gait dùng cùng `screening_id` và đường dẫn `/gait/file`. Phản hồi lần thứ hai có
`overall_risk` và `recommendation`; frontend không tự tính lại. Monitoring đọc
`monitoring_eligibility` từ chi tiết hồ sơ để bật/tắt lựa chọn và vẫn để server kiểm tra.
Để gọi Voice WAV sau khi extractor sẵn sàng:

```python
record = api.upload(
    f"/api/v1/patients/{patient_id}/voice/file",
    uploaded_wav.getvalue(), recorded_at="2026-09-11T09:00:00+07:00",
)
history = api.request("GET", f"/api/v1/patients/{patient_id}/voice")
# history['items'] đã sắp theo thời gian, có recorded_at, voice_score, trend.
```

## Cấu trúc và phạm vi kiểm thử

- `app.py`: HTTP, xác thực, giới hạn upload, CORS, hợp đồng và xuất báo cáo.
- `schemas.py`, `outputs.py`: validation request và schema response.
- `service.py`: nghiệp vụ, transaction, phiên sàng lọc và theo dõi.
- `db.py`: schema SQLite phiên bản 1, foreign key, transaction và chỉ mục.
- `policy.py`: fusion labels, khuyến nghị và tính xu hướng.
- `inference.py`: registry model, kiểm tra hợp đồng và bộ trích đặc trưng.
- `client.py`: client gọi API cho frontend.

Các test nghiệp vụ dùng predictor cố định chỉ trong `tests/`; test adapter dùng
Random Forest thực được train nhanh trên dữ liệu tổng hợp, trong thư mục tạm.
Có test ảnh PNG đi qua HOG, file Gait của repository đi qua bộ trích 19 cột và suy
luận qua HTTP. Những test này xác nhận backend tích hợp đúng; không đánh giá độ
chính xác y khoa của model nhóm và không tạo model demo dùng trong ứng dụng thật.

SQLite phù hợp chạy demo nội bộ trên một máy. Dữ liệu và API key được giữ local;
các API không cache hồ sơ. Trước khi triển khai cho nhiều người dùng ngoài nhóm,
cần bổ sung đăng nhập/phân quyền, TLS và quy trình sao lưu phù hợp. Backend hiện
không có tác vụ xóa hồ sơ hay endpoint sửa trực tiếp điểm AI đã lưu.

Tài liệu framework tham khảo: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
và [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).
