import streamlit as st

# Demo user accounts
USERS = {
    "admin":        {"password": "admin123",  "name": "Admin",               "role": "Admin"},
    "bacsi":        {"password": "bacsi123",  "name": "BS. Nguyen Van A",    "role": "Bac si"},
    "kythuatvien":  {"password": "ktv123",    "name": "KTV. Tran Thi B",     "role": "Ky thuat vien"},
}

def render_login():
    # ---- Inject custom CSS ----
    st.markdown('''<style>
    /* Gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg,
            #a8e6cf 0%, #88d8b0 25%, #b8d4e3 50%,
            #d4b5d0 75%, #f0a5a5 100%) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        padding-top: 5vh !important;
        max-width: 1100px !important;
    }
    /* Card styles */
    .pd-info-card {
        background: rgba(255,255,255,0.88);
        border-radius: 20px;
        padding: 40px 36px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.10);
        backdrop-filter: blur(12px);
        min-height: 370px;
    }
    .pd-login-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 40px 36px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.10);
        min-height: 370px;
    }
    .pd-logo {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        color: #fff; font-weight: 800; font-size: 16px;
        margin-right: 10px; vertical-align: middle;
    }
    .pd-title {
        font-size: 21px; font-weight: 700; color: #1a5276;
        vertical-align: middle;
    }
    /* Green button */
    div[data-testid="stForm"] {
        border: none !important; padding: 0 !important;
    }
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #2ecc71 !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        margin-top: 8px;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #27ae60 !important;
    }
    </style>''', unsafe_allow_html=True)

    # ---- Two-column layout ----
    col_left, col_gap, col_right = st.columns([5, 1, 4])

    with col_left:
        st.markdown('''<div class="pd-info-card">
            <div style="margin-bottom:22px;">
                <span class="pd-logo">PD</span>
                <span class="pd-title">Parkinson Detection</span>
            </div>
            <p style="color:#555;font-size:14.5px;line-height:1.75;">
                Parkinson Detection la he thong thong minh su dung AI da mo thuc
                (Net ve, Dang di, Giong noi) de ho tro sang loc som va theo doi
                tien trien benh Parkinson.
            </p>
            <br>
            <p style="color:#888;font-size:13.5px;font-style:italic;">
                Day la cong cu huu ich cho:
            </p>
            <p style="color:#555;font-size:14px;margin:6px 0;">
                &#10003; &nbsp; Bac si chuyen khoa Than kinh</p>
            <p style="color:#555;font-size:14px;margin:6px 0;">
                &#10003; &nbsp; Ky thuat vien y te</p>
            <p style="color:#555;font-size:14px;margin:6px 0;">
                &#10003; &nbsp; Sinh vien & Nghien cuu sinh Y khoa</p>
        </div>''', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="pd-login-card">', unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            st.markdown("**Ten dang nhap**")
            username = st.text_input(
                "user_label", placeholder="user@example.com",
                label_visibility="collapsed",
            )
            st.markdown("**Mat khau**")
            password = st.text_input(
                "pass_label", type="password", placeholder="******",
                label_visibility="collapsed",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.checkbox("Ghi nho toi", key="remember_me")
            with c2:
                st.markdown(
                    '<p style="text-align:right;margin-top:6px;">'
                    '<a href="#" style="color:#3498db;font-size:13px;">'
                    'Quen mat khau?</a></p>',
                    unsafe_allow_html=True,
                )

            submitted = st.form_submit_button(
                "Dang nhap", type="primary", use_container_width=True,
            )
            if submitted:
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = USERS[username]
                    st.session_state["page"] = "dashboard"
                    st.rerun()
                else:
                    st.error("Sai ten dang nhap hoac mat khau!")

        st.markdown('</div>', unsafe_allow_html=True)
