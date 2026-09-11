import streamlit as st
from datetime import datetime
import pandas as pd

def render_dashboard():
    st.markdown('''<style>
    .kpi-card { background:#fff; border-radius:16px; padding:24px 20px;
        box-shadow:0 2px 12px rgba(0,0,0,0.07); text-align:center; }
    .kpi-icon { font-size:32px; margin-bottom:6px; }
    .kpi-value { font-size:28px; font-weight:800; color:#1a5276; }
    .kpi-label { font-size:13px; color:#888; margin-top:4px; }
    .action-card { background:#fff; border-radius:14px; padding:22px;
        box-shadow:0 2px 10px rgba(0,0,0,0.06); text-align:center; min-height:150px; }
    .action-icon { font-size:38px; margin-bottom:10px; }
    .action-title { font-size:16px; font-weight:700; color:#2c3e50; }
    .action-desc { font-size:13px; color:#777; margin-top:6px; }
    .welcome-bar { background:linear-gradient(90deg,#1a5276,#2ecc71);
        border-radius:14px; padding:24px 30px; color:#fff; margin-bottom:28px; }
    </style>''', unsafe_allow_html=True)

    user = st.session_state.get('user', {})
    now  = datetime.now().strftime('%d/%m/%Y  %H:%M')
    history = st.session_state.get('patient_history', [])

    # Welcome banner
    st.markdown(f'''<div class="welcome-bar">
        <h2 style="margin:0;color:#fff;">Xin chao, {user.get('name','User')}!</h2>
        <p style="margin:4px 0 0;opacity:0.85;font-size:14px;">{user.get('role','')} &nbsp;|&nbsp; {now}</p>
    </div>''', unsafe_allow_html=True)

    # KPI cards - dynamic
    total = len(history)
    today = datetime.now().strftime('%d/%m/%Y')
    today_count = sum(1 for p in history if p.get('date','') == today)
    high_risk   = sum(1 for p in history if p.get('final_risk', 0) >= 60)
    avg_risk    = round(sum(p.get('final_risk',0) for p in history)/total, 1) if total else 0

    c1,c2,c3,c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, '👥', total,        'Tong benh nhan'),
        (c2, '🩺', today_count,  'Sang loc hom nay'),
        (c3, '⚠️',  high_risk,   'Nguy co cao'),
        (c4, '📊', f'{avg_risk}%','Risk trung binh'),
    ]:
        with col:
            st.markdown(f'''<div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Quick actions
    st.subheader('Thao tac nhanh')
    a1,a2,a3,a4 = st.columns(4)
    with a1:
        st.markdown('<div class="action-card"><div class="action-icon">🔍</div>'
            '<div class="action-title">Sang loc moi</div>'
            '<div class="action-desc">Phan tich Net ve + Dang di</div></div>', unsafe_allow_html=True)
        if st.button('Bat dau', key='act_screen', use_container_width=True, type='primary'):
            st.session_state['page'] = 'screening'
            st.session_state['step'] = 2
            st.session_state['results'] = {}
            st.session_state['patient_info'] = {}
            st.rerun()
    with a2:
        st.markdown('<div class="action-card"><div class="action-icon">🎙</div>'
            '<div class="action-title">Voice Monitoring</div>'
            '<div class="action-desc">Theo doi giong noi dinh ky</div></div>', unsafe_allow_html=True)
        if st.button('Bat dau', key='act_voice', use_container_width=True, type='primary'):
            st.session_state['page'] = 'monitoring'
            st.session_state['step'] = 7
            st.rerun()
    with a3:
        st.markdown('<div class="action-card"><div class="action-icon">📄</div>'
            '<div class="action-title">Xem bao cao</div>'
            '<div class="action-desc">Bao cao tong hop Y te</div></div>', unsafe_allow_html=True)
        if st.button('Xem', key='act_report', use_container_width=True):
            st.session_state['page'] = 'report'
            st.session_state['step'] = 9
            st.rerun()
    with a4:
        st.markdown('<div class="action-card"><div class="action-icon">📈</div>'
            '<div class="action-title">Lich su benh nhan</div>'
            '<div class="action-desc">Danh sach da kham</div></div>', unsafe_allow_html=True)
        if st.button('Xem', key='act_history', use_container_width=True):
            st.session_state['page'] = 'monitoring'
            st.session_state['step'] = 8
            st.rerun()

    st.markdown('<br>', unsafe_allow_html=True)

    # Recent activity - from patient_history
    st.subheader('🕐 Hoat dong gan day')
    if history:
        rows = []
        for p in reversed(history[-10:]):
            rows.append({
                'Thoi gian':  p.get('date',''),
                'Ma BN':      p.get('id',''),
                'Ho ten':     p.get('name',''),
                'Tuoi':       p.get('age',''),
                'Hinh thuc':  p.get('mode','Sang loc'),
                'Final Risk': f"{p.get('final_risk','-')}%",
                'UPDRS':      p.get('updrs', '-'),
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info('Chua co benh nhan nao duoc kham trong phien lam viec nay.')
