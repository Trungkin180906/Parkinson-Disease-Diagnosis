import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
from system.utils import load_all_models, extract_voice_features

models = load_all_models()


def _go(step):
    st.session_state["step"] = step
    st.rerun()


def render_monitoring():
    step = st.session_state.get("step", 7)
    if step == 7:
        _step_voice()
    elif step == 8:
        _step_history()
    else:
        _step_voice()


# ----- Step 7: Voice monitoring -----
def _step_voice():
    st.markdown("## 🎙️ Buoc 7: Theo doi qua Giong noi (Voice)")
    st.info("Yeu cau benh nhan doc to, ro rang mot doan van ban trong 10-15 giay. Ghi am va tai len.")
    uploaded = st.file_uploader("Tai len file ghi am (.wav)", type=["wav"])
    if uploaded:
        st.audio(uploaded)
        if st.button("🧠 Phan tich AI", type="primary", use_container_width=True):
            vm = models.get("voice_model")
            vs = models.get("voice_scaler")
            vf = models.get("voice_feats")
            if vm and vs and vf:
                with st.spinner("AI dang do luong dac trung am hoc..."):
                    uploaded.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    try:
                        feat_dict = extract_voice_features(tmp_path)
                        x_input = np.array([[feat_dict[col] for col in vf]])
                        x_scaled = vs.transform(x_input)
                        updrs = round(float(vm.predict(x_scaled)[0]), 2)
                        voice_score = round(max(0, min(100, 100 - (updrs / 176 * 100))), 1)
                        st.session_state["results"]["updrs"] = updrs
                        st.session_state["results"]["voice_score"] = voice_score
                        st.session_state["results"]["voice_feats"] = feat_dict
                    finally:
                        os.unlink(tmp_path)
                st.success("Phan tich hoan tat!")
            else:
                st.error("Thieu model Voice. Kiem tra cac file .pkl")

    # Hien ket qua neu da phan tich
    if "updrs" in st.session_state.get("results", {}):
        res = st.session_state["results"]
        updrs       = res["updrs"]
        voice_score = res["voice_score"]
        feat_dict   = res.get("voice_feats", {})

        st.markdown("---")
        st.markdown("### 📊 Ket qua Phan tich")

        # KPI chinh
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Diem UPDRS", updrs, help="Thang diem 0-176. Cang cao cang nang.")
        c2.metric("🏆 Voice Score", f"{voice_score} / 100")
        # Phan loai
        if updrs < 20:
            c3.markdown('<div style="background:#27ae60;color:#fff;border-radius:10px;padding:14px;text-align:center;"><b>Giai doan nhe</b><br>UPDRS < 20</div>', unsafe_allow_html=True)
        elif updrs < 40:
            c3.markdown('<div style="background:#f39c12;color:#fff;border-radius:10px;padding:14px;text-align:center;"><b>Giai doan trung binh</b><br>UPDRS 20-40</div>', unsafe_allow_html=True)
        else:
            c3.markdown('<div style="background:#e74c3c;color:#fff;border-radius:10px;padding:14px;text-align:center;"><b>Giai doan tien trien</b><br>UPDRS > 40</div>', unsafe_allow_html=True)

        st.markdown("**Muc do Voice Score:**")
        st.progress(int(voice_score))

        # Bang 16 dac trung am hoc
        if feat_dict:
            st.markdown("---")
            st.markdown("### 🔊 Chi tiet 16 Dac trung Am hoc")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Nhom Jitter** *(dao dong tan so)*")
                jitter_data = {
                    "Dac trung": ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP"],
                    "Gia tri":   [round(feat_dict.get("Jitter(%)",0),5),
                                  round(feat_dict.get("Jitter(Abs)",0),7),
                                  round(feat_dict.get("Jitter:RAP",0),5),
                                  round(feat_dict.get("Jitter:PPQ5",0),5),
                                  round(feat_dict.get("Jitter:DDP",0),5)],
                    "Y nghia": ["% bien thien chu ky", "Do bien thien tuyet doi",
                                "Bien thien 3-diem", "Bien thien 5-diem", "Bien thien vi sai"]
                }
                st.dataframe(jitter_data, use_container_width=True, hide_index=True)

                st.markdown("**Nhom Phi tuyen tinh**")
                nonlin_data = {
                    "Dac trung": ["RPDE", "DFA", "PPE"],
                    "Gia tri":   [round(feat_dict.get("RPDE",0),4),
                                  round(feat_dict.get("DFA",0),4),
                                  round(feat_dict.get("PPE",0),4)],
                    "Y nghia": ["Do phuc tap dong luc hoc", "Chi so tu tuong quan",
                                "Entropy phan phoi pitch"]
                }
                st.dataframe(nonlin_data, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Nhom Shimmer** *(dao dong bien do)*")
                shimmer_data = {
                    "Dac trung": ["Shimmer", "Shimmer(dB)", "Shimmer:APQ3",
                                  "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA"],
                    "Gia tri":   [round(feat_dict.get("Shimmer",0),5),
                                  round(feat_dict.get("Shimmer(dB)",0),3),
                                  round(feat_dict.get("Shimmer:APQ3",0),5),
                                  round(feat_dict.get("Shimmer:APQ5",0),5),
                                  round(feat_dict.get("Shimmer:APQ11",0),5),
                                  round(feat_dict.get("Shimmer:DDA",0),5)],
                    "Y nghia": ["Bien thien am luong cuc bo", "Bien thien theo dB",
                                "TB truot 3-diem", "TB truot 5-diem",
                                "TB truot 11-diem", "Bien thien vi sai"]
                }
                st.dataframe(shimmer_data, use_container_width=True, hide_index=True)

                st.markdown("**Nhom Nhieu (NHR / HNR)**")
                noise_data = {
                    "Dac trung": ["NHR", "HNR"],
                    "Gia tri":   [round(feat_dict.get("NHR",0),5),
                                  round(feat_dict.get("HNR",0),3)],
                    "Y nghia": ["Ti le Nhieu/Thanh am", "Ti le Thanh am/Nhieu (dB)"]
                }
                st.dataframe(noise_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("Xem Lich su Benh nhan ➡", use_container_width=True):
            _go(8)


# ----- Step 8: Patient history list -----
def _step_history():
    st.markdown("## 📋 Buoc 8: Lich su Benh nhan da kham")
    history = st.session_state.get("patient_history", [])

    if not history:
        st.info("Chua co benh nhan nao duoc luu trong phien lam viec nay.\nVui long thuc hien Sang loc truoc.")
    else:
        st.success(f"Tong cong **{len(history)}** benh nhan da kham trong phien nay.")
        rows = []
        for p in reversed(history):
            rows.append({
                "Ma BN":      p.get("id", ""),
                "Ho ten":     p.get("name", ""),
                "Tuoi":       p.get("age", ""),
                "Gioi tinh":  p.get("gender", ""),
                "Ngay kham":  p.get("date", ""),
                "Hinh thuc":  p.get("mode", ""),
                "Drawing %":  p.get("drawing_risk", "-"),
                "Gait %":     p.get("gait_risk", "-"),
                "Final Risk":  p.get("final_risk", "-"),
                "UPDRS":      p.get("updrs", "-"),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Bieu do xu huong voice score
        voice_patients = [p for p in history if p.get("voice_score") not in ["-", None]]
        if len(voice_patients) >= 2:
            st.markdown("---")
            st.subheader("Bieu do Voice Score theo thoi gian")
            names  = [p.get("name","") for p in voice_patients]
            scores = [p.get("voice_score", 0) for p in voice_patients]
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(names, scores, marker="o", color="#E74C3C", lw=2)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Voice Score")
            ax.set_title("Xu huong Voice Score cac benh nhan")
            ax.grid(True, ls="--")
            st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Xuat Bao Cao", type="primary", use_container_width=True):
            st.session_state["page"] = "report"
            _go(9)
    with col2:
        if st.button("Ve Dashboard", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()
