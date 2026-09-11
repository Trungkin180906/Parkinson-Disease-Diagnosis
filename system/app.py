import streamlit as st
import os, sys

# Allow imports from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from login import render_login
from dashboard import render_dashboard
from screening import render_screening
from monitoring import render_monitoring
from report import render_report

# ---------- Page config ----------
st.set_page_config(
    page_title="AI Parkinson Screening System",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Session defaults ----------
_defaults = {
    "authenticated": False,
    "user": None,
    "page": "login",
    "step": 1,
    "patient_info": {},
    "mode": None,
    "results": {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------- Routing ----------
if not st.session_state["authenticated"]:
    render_login()
else:
    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;margin-bottom:10px;">'
            '<span style="font-size:40px;">\U0001f9e0</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="text-align:center;font-weight:700;font-size:15px;margin:0;">'
            f'{st.session_state["user"]["name"]}</p>'
            f'<p style="text-align:center;color:#888;font-size:13px;">'
            f'{st.session_state["user"]["role"]}</p>',
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("\U0001f3e0  Dashboard", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()
        if st.button("\U0001fa7a  Sang loc moi", use_container_width=True):
            st.session_state["page"] = "screening"
            st.session_state["step"] = 2
            st.rerun()
        if st.button("\U0001f399  Theo doi giong noi", use_container_width=True):
            st.session_state["page"] = "monitoring"
            st.session_state["step"] = 7
            st.rerun()
        if st.button("\U0001f4c4  Bao cao", use_container_width=True):
            st.session_state["page"] = "report"
            st.session_state["step"] = 9
            st.rerun()
        st.divider()
        if st.button("\U0001f6aa  Dang xuat", use_container_width=True, type="primary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ---- Main area ----
    page = st.session_state["page"]
    if page == "dashboard":
        render_dashboard()
    elif page == "screening":
        render_screening()
    elif page == "monitoring":
        render_monitoring()
    elif page == "report":
        render_report()
    else:
        render_dashboard()
