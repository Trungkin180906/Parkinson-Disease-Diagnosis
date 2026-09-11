import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from system.utils import load_all_models, extract_hog, extract_gait_features

models = load_all_models()


def _go(step):
    st.session_state['step'] = step
    st.rerun()


def _progress_bar(step):
    steps = ['Thong tin BN', 'Chon mode', 'Net ve', 'Dang di', 'Ket qua']
    idx = step - 2
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        if i < idx:
            col.markdown(f'<div style="text-align:center;color:#27ae60;font-size:12px;">&#10003; {label}</div>', unsafe_allow_html=True)
        elif i == idx:
            col.markdown(f'<div style="text-align:center;color:#2980b9;font-weight:bold;font-size:12px;">&#9654; {label}</div>', unsafe_allow_html=True)
        else:
            col.markdown(f'<div style="text-align:center;color:#888;font-size:12px;">{label}</div>', unsafe_allow_html=True)
    st.markdown('---')


def render_screening():
    step = st.session_state.get('step', 2)
    if step == 2:
        _step_patient_info()
    elif step == 3:
        _step_choose_mode()
    elif step == 4:
        _step_drawing()
    elif step == 5:
        _step_gait()
    elif step == 6:
        _step_final_risk()
    else:
        _step_patient_info()


# ----- Step 2: Patient info -----
def _step_patient_info():
    _progress_bar(2)
    st.markdown('## 🧑 Buoc 2: Nhap thong tin Benh nhan')
    with st.form('patient_form'):
        c1, c2 = st.columns(2)
        with c1:
            name   = st.text_input('👤 Ho va ten *')
            age    = st.number_input('🎂 Tuoi *', min_value=1, max_value=120, value=60)
            gender = st.selectbox('⚧ Gioi tinh', ['Nam', 'Nu'])
        with c2:
            height = st.number_input('📏 Chieu cao (cm)', value=165)
            weight = st.number_input('⚖ Can nang (kg)', value=60)
            phone  = st.text_input('📞 So dien thoai')
        submitted = st.form_submit_button('Luu ho so & Tiep tuc ➡', type='primary')
        if submitted:
            if not name.strip():
                st.error('Vui long nhap Ho va ten!')
            else:
                st.session_state['patient_info'] = {
                    'id':     f"BN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'name':   name,   'age':    age,
                    'gender': gender, 'height': height,
                    'weight': weight, 'phone':  phone,
                    'date':   datetime.now().strftime('%d/%m/%Y'),
                }
                _go(3)


# ----- Step 3: Choose mode -----
def _step_choose_mode():
    _progress_bar(3)
    info = st.session_state.get('patient_info', {})
    st.markdown(f"## 🩺 Buoc 3: Lua chon hinh thuc kham  \n**Benh nhan:** {info.get('name','')} | Tuoi: {info.get('age','')} | {info.get('gender','')}")
    st.markdown('')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('### 🔍 Kham moi (Sang loc som)')
        st.info('Phan tich **Net ve** (Drawing) va **Dang di** (Gait).\nPhu hop khi lan dau kham hoac tai kham dinh ky.')
        if st.button('Chon KHAM MOI', use_container_width=True, type='primary'):
            st.session_state['mode'] = 'screening'
            _go(4)
    with c2:
        st.markdown('### 🎙 Theo doi dinh ky')
        st.warning('Phan tich **Giong noi** (Voice) de theo doi tien trien benh.\nDung cho benh nhan da co chan doan truoc.')
        if st.button('Chon THEO DOI', use_container_width=True):
            st.session_state['mode'] = 'monitoring'
            st.session_state['page'] = 'monitoring'
            _go(7)


# ----- Step 4: Drawing analysis -----
def _step_drawing():
    _progress_bar(4)
    st.markdown('## ✏ Buoc 4: Phan tich Net ve (Drawing)')
    st.info('Yeu cau benh nhan ve hinh xoan oc / song song tren giay, chup anh va tai len.')
    uploaded = st.file_uploader('Tai len anh bai kiem tra ve', type=['jpg', 'png', 'jpeg'])
    if uploaded:
        img_pil = Image.open(uploaded).convert('RGB')
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img_pil, caption='Anh net ve', use_container_width=True)
        with c2:
            st.markdown('**Hinh anh da tai len thanh cong.**')
            if st.button('🧠 Phan tich AI', type='primary', use_container_width=True):
                dm = models.get('draw_model')
                ds = models.get('draw_scaler')
                if dm and ds:
                    with st.spinner('AI dang phan tich net ve...'):
                        features    = extract_hog(np.array(img_pil))
                        features_sc = ds.transform([features])
                        proba       = dm.predict_proba(features_sc)[0]
                        risk        = round(proba[1] * 100, 2)
                        st.session_state['results']['drawing_risk'] = risk
                else:
                    st.error('Chua nap duoc model Drawing. Kiem tra file .pkl')

    # Nut Tiep tuc nam NGOAI if uploaded - luon hien khi da co ket qua
    if 'drawing_risk' in st.session_state.get('results', {}):
        risk = st.session_state['results']['drawing_risk']
        st.success(f'✅ Ket qua phan tich - Drawing Risk: **{risk}%**')
        st.progress(int(risk))
        if st.button('Tiep tuc: Phan tich Dang di ➡', type='primary', use_container_width=True):
            _go(5)


# ----- Step 5: Gait analysis -----
def _step_gait():
    _progress_bar(5)
    st.markdown('## 👣 Buoc 5: Phan tich Dang di (Gait)')
    d_risk = st.session_state['results'].get('drawing_risk', 'N/A')
    st.info(f'Drawing Risk da ghi nhan: **{d_risk}%**')
    st.markdown('Tai len file du lieu cam bien luc ban chan (.txt) thu duoc tu cam bien VGRF.')
    uploaded = st.file_uploader('Tai len du lieu cam bien (.txt / .csv)', type=['txt', 'csv'])
    if uploaded:
        df = pd.read_csv(uploaded, sep='\t', header=None, on_bad_lines='skip')
        st.write('**Xem truoc du lieu:**')
        st.dataframe(df.head(3), use_container_width=True)
        if st.button('🧠 Phan tich AI', type='primary', use_container_width=True):
            gm = models.get('gait_model')
            gs = models.get('gait_scaler')
            if gm and gs:
                with st.spinner('AI dang phan tich dang di...'):
                    features    = extract_gait_features(df)
                    features_sc = gs.transform([features])
                    proba       = gm.predict_proba(features_sc)[0]
                    risk        = round(proba[1] * 100, 2)
                    st.session_state['results']['gait_risk'] = risk
            else:
                st.error('Chua nap duoc model Gait. Kiem tra file .pkl')

    # Nut Tiep tuc nam NGOAI if uploaded
    if 'gait_risk' in st.session_state.get('results', {}):
        risk = st.session_state['results']['gait_risk']
        st.success(f'✅ Ket qua phan tich - Gait Risk: **{risk}%**')
        st.progress(int(risk))
        if st.button('Tiep tuc: Tong hop ket qua ➡', type='primary', use_container_width=True):
            _go(6)


# ----- Step 6: Final risk -----
def _step_final_risk():
    _progress_bar(6)
    st.markdown('## 📊 Buoc 6: Tong hop Ket qua Sang loc')
    d_risk = st.session_state['results'].get('drawing_risk', 0)
    g_risk = st.session_state['results'].get('gait_risk', 0)
    final  = round((d_risk + g_risk) / 2, 2)
    st.session_state['results']['final_risk'] = final

    c1, c2, c3 = st.columns(3)
    c1.metric('✏ Drawing Risk', f'{d_risk}%')
    c2.metric('👣 Gait Risk',    f'{g_risk}%')
    c3.metric('🚨 FINAL RISK',   f'{final}%')

    st.markdown('---')
    st.markdown('**Muc do nguy co tong hop:**')
    st.progress(int(final))

    st.markdown('---')
    if final < 30:
        st.success(f'🟢 **Nguy co thap ({final}%)** - Khong phat hien dau hieu bat thuong. Khuyen nghi tai kham sau 12 thang.')
    elif final < 60:
        st.warning(f'🟡 **Nguy co trung binh ({final}%)** - Co mot so dau hieu can theo doi. Khuyen nghi tai kham sau 6 thang.')
    else:
        st.error(f'🔴 **Nguy co cao ({final}%)** - Phat hien nhieu dau hieu bat thuong. Khuyen nghi lam them Voice Monitoring va chuyen tuyen.')

    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🎙 Chuyen sang Voice Monitoring', use_container_width=True):
            st.session_state['page'] = 'monitoring'
            _go(7)
    with col2:
        if st.button('📄 Xuat Bao Cao Cuoi Cung', type='primary', use_container_width=True):
            st.session_state['page'] = 'report'
            _go(9)
