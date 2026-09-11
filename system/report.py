import streamlit as st
from datetime import datetime


def _save_to_history():
    info    = st.session_state.get('patient_info', {})
    results = st.session_state.get('results', {})
    if not info.get('name'):
        return
    record = {
        **info,
        'mode':       st.session_state.get('mode', 'screening'),
        'final_risk': results.get('final_risk', '-'),
        'drawing_risk': results.get('drawing_risk', '-'),
        'gait_risk':  results.get('gait_risk', '-'),
        'updrs':      results.get('updrs', '-'),
        'voice_score': results.get('voice_score', '-'),
        'time':       datetime.now().strftime('%H:%M'),
    }
    history = st.session_state.get('patient_history', [])
    # Tranh luu trung
    if not any(p.get('id') == record.get('id') for p in history):
        history.append(record)
        st.session_state['patient_history'] = history


def render_report():
    st.markdown('## 📋 Buoc 9: Bao cao Tong hop Y te')
    info    = st.session_state.get('patient_info', {})
    results = st.session_state.get('results', {})

    if not info.get('name'):
        st.warning('Chua co thong tin benh nhan. Vui long thuc hien Sang loc truoc.')
        if st.button('Ve Dashboard'):
            st.session_state['page'] = 'dashboard'
            st.rerun()
        return

    # Tu dong luu vao lich su khi vao trang bao cao
    _save_to_history()

    final = results.get('final_risk', None)

    # Header BN
    st.markdown(f"""
    | Truong | Thong tin |
    |---|---|
    | **Ma BN** | {info.get('id','')} |
    | **Ho ten** | {info.get('name','')} |
    | **Tuoi / Gioi tinh** | {info.get('age','')} tuoi / {info.get('gender','')} |
    | **Ngay kham** | {info.get('date','')} |
    | **So dien thoai** | {info.get('phone','')} |
    """)

    st.markdown('---')
    st.subheader('🧠 Ket qua Phan tich AI')

    c1, c2, c3 = st.columns(3)
    if 'drawing_risk' in results:
        c1.metric('✏ Drawing Risk', f"{results['drawing_risk']}%")
    if 'gait_risk' in results:
        c2.metric('👣 Gait Risk', f"{results['gait_risk']}%")
    if final is not None:
        c3.metric('🚨 Final Risk', f'{final}%')

    if 'updrs' in results:
        st.markdown(f'**Diem UPDRS (Voice):** {results["updrs"]}  |  **Voice Score:** {results["voice_score"]}/100')

    st.markdown('---')
    if final is not None:
        if final < 30:
            st.success(f'🟢 **NGUY CO THAP ({final}%)** - Khong phat hien nhieu dau hieu bat thuong. Hen tai kham sau 12 thang.')
        elif final < 60:
            st.warning(f'🟡 **NGUY CO TRUNG BINH ({final}%)** - Co mot so dau hieu can theo doi. Hen tai kham sau 6 thang.')
        else:
            st.error(f'🔴 **NGUY CO CAO ({final}%)** - Khuyen nghi chuyen bac si chuyen khoa Than kinh.')

    st.markdown('---')
    st.info('Chu ky Bac si phu trach:\n\n\n\n.....................................................')

    col1, col2 = st.columns(2)
    with col1:
        if st.button('🖨 In Bao Cao', use_container_width=True):
            st.toast('Dang ket noi may in...')
    with col2:
        if st.button('🆕 Kham Benh Nhan Khac', type='primary', use_container_width=True):
            st.session_state['patient_info'] = {}
            st.session_state['results']      = {}
            st.session_state['mode']         = None
            st.session_state['step']         = 2
            st.session_state['page']         = 'screening'
            st.rerun()
