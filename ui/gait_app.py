import streamlit as st
import tempfile
import os
import pandas as pd

from ai.gait.predict import predict_gait_risk


# ============================================================
# CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Parkinson's Gait Analysis",
    page_icon="🚶",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">
        🚶 Parkinson's Gait Analysis
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Phân tích dữ liệu dáng đi để đánh giá nguy cơ Parkinson
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 1. UPLOAD FILE
# ============================================================

st.header("1. Upload dữ liệu dáng đi")

upload_file = st.file_uploader(
    "Chọn file dữ liệu Gait (.txt)",
    type=["txt"]
)


# ============================================================
# 2. HIỂN THỊ THÔNG TIN FILE
# ============================================================

if upload_file is not None:

    st.success(f"Đã chọn file: {upload_file.name}")

    st.write(
        f"Kích thước file: {upload_file.size / 1024:.2f} KB"
    )


# ============================================================
# 3. NÚT PHÂN TÍCH
# ============================================================

if st.button(
    "🔍 Phân tích dáng đi",
    type="primary",
    use_container_width=True
):

    if upload_file is None:
        st.warning("⚠️ Vui lòng tải file dữ liệu Gait lên trước.")
        st.stop()

    # Tạo file tạm để predict.py đọc
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt"
    ) as tmp:

        tmp.write(upload_file.getvalue())
        tmp_path = tmp.name

    try:

        with st.spinner("🤖 AI đang phân tích dữ liệu dáng đi..."):

            # Gọi model Gait
            result = predict_gait_risk(tmp_path)

        # ====================================================
        # KIỂM TRA KẾT QUẢ
        # ====================================================

        if result.get("status") == "failed":
            st.error(
                f"❌ Không thể phân tích: {result.get('error')}"
            )
            st.stop()

        if "error" in result:
            st.error(f"❌ {result['error']}")
            st.stop()

        risk_score = result["risk_score"]

        # ====================================================
        # HIỂN THỊ KẾT QUẢ
        # ====================================================

        st.markdown("---")

        st.header("2. Kết quả phân tích")

        st.info(
            "Risk Score là xác suất do mô hình AI dự đoán dựa "
            "trên dữ liệu dáng đi. Đây không phải là chẩn đoán y tế."
        )

        st.metric(
            label="Gait Risk Score",
            value=f"{risk_score:.2f}%"
        )

        # ====================================================
        # PHÂN LOẠI
        # ====================================================

        if risk_score >= 50:

            st.warning(
                "⚠️ Mô hình dự đoán: Có nguy cơ Parkinson"
            )

        else:

            st.success(
                "✅ Mô hình dự đoán: Nguy cơ thấp"
            )

        # ====================================================
        # HIỂN THỊ 5 ĐẶC TRƯNG
        # ====================================================

        st.markdown("---")

        st.header("3. Đặc trưng dáng đi")

        # Đọc lại file để hiển thị các đặc trưng
        df = pd.read_csv(
            tmp_path,
            sep="\t",
            header=None,
            on_bad_lines="skip"
        )

        total_left = df.iloc[:, 16]
        total_right = df.iloc[:, 17]

        mean_left = total_left.mean()
        mean_right = total_right.mean()
        std_left = total_left.std()
        std_right = total_right.std()
        asymmetry = abs(mean_left - mean_right)

        feature_data = {
            "Đặc trưng": [
                "Mean lực chân trái",
                "Mean lực chân phải",
                "Std lực chân trái",
                "Std lực chân phải",
                "Độ bất đối xứng"
            ],
            "Giá trị": [
                mean_left,
                mean_right,
                std_left,
                std_right,
                asymmetry
            ]
        }

        feature_df = pd.DataFrame(feature_data)

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"❌ Có lỗi trong quá trình phân tích: {e}"
        )

    finally:

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)