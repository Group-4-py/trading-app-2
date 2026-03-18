"""
🏠 Home – Automated Daily Trading System
Main landing page (Streamlit entry point).
"""

import base64
import os

import streamlit as st

# ── Page Configuration (MUST be first Streamlit command) ─────────────
st.set_page_config(
    page_title="AutoTrader | AI-Powered Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.style import inject_custom_css
from utils.config import TICKERS, TEAM_MEMBERS, APP_LOGO

# ── Inject Theme ─────────────────────────────────────────────────────
inject_custom_css()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### AutoTrader")
    st.markdown("---")
    st.markdown(
        """
        **AI-Powered Daily Trading System**

        Navigate through the pages:
        - **Home** – Overview
        - **Go Live** – Real-time predictions
        - **Model Insights** – ML analysis
        - **Backtesting** – Strategy simulator
        """
    )
    st.markdown("---")

    # API Key input — store in a persistent key so other pages can read it
    api_key = st.text_input(
        "🔑 SimFin API Key",
        value=st.session_state.get("api_key_stored", ""),
        type="password",
        help="Enter your SimFin API key to fetch real market data. Leave empty for demo mode.",
    )
    st.session_state["api_key_stored"] = api_key

    if api_key:
        st.success("API key configured ✓")
    else:
        st.info("Running in **Demo Mode** with synthetic data.")

    st.markdown("---")
    st.caption("Built with Streamlit · Python · SimFin")


# ── Hero Section ─────────────────────────────────────────────────────
_logo_html = ""
_logo_abs = os.path.join(os.path.dirname(__file__), APP_LOGO)
if os.path.isfile(_logo_abs):
    with open(_logo_abs, "rb") as _f:
        _logo_data = base64.b64encode(_f.read()).decode()
    _logo_ext = _logo_abs.rsplit(".", 1)[-1].lower()
    _logo_mime = "image/jpeg" if _logo_ext in ("jpg", "jpeg") else f"image/{_logo_ext}"
    _logo_html = (
        f'<img src="data:{_logo_mime};base64,{_logo_data}" '
        f'style="width:300px;height:300px;object-fit:contain;margin-bottom:1rem;" />'
    )

st.markdown(
    f"""
    <div class="hero-bg" style="text-align: center;">
        {_logo_html}
        <div class="hero-title">AUTOTRADER</div>
        <div class="hero-subtitle">AI-Powered Daily Trading System</div>
        <br>
        <p style="color: #94a3b8; max-width: 700px; margin: 0 auto; font-size: 1.05rem; line-height: 1.7;">
            A machine-learning-driven platform that analyzes historical stock data,
            predicts next-day market movements, and generates actionable trading signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# ── Key Stats Banner ─────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stocks Tracked", f"{len(TICKERS)}")
with col2:
    st.metric("Model Used", "Classification")
with col3:
    st.metric("Features", "20+")
with col4:
    st.metric("Signal Frequency", "Daily")

st.markdown("")
st.markdown("---")

# ── System Architecture ──────────────────────────────────────────────
st.markdown("## System Architecture")
st.markdown("")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #00d4ff !important;">
                🔬 Part 1 — Data Analytics (Offline)
            </h3>
            <ul style="color: #94a3b8; line-height: 2;">
                <li><strong style="color: #00c853;">ETL Pipeline</strong> — Extract & transform SimFin bulk data</li>
                <li><strong style="color: #00c853;">Feature Engineering</strong> — 20+ technical indicators</li>
                <li><strong style="color: #00c853;">ML Classification</strong> — Predict UP/DOWN movements</li>
                <li><strong style="color: #00c853;">Model Export</strong> — Serialized .joblib for production</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #00d4ff !important;">
                🌐 Part 2 — Web System (Online)
            </h3>
            <ul style="color: #94a3b8; line-height: 2;">
                <li><strong style="color: #fbbf24;">PySimFin Wrapper</strong> — OOP API client for real-time data</li>
                <li><strong style="color: #fbbf24;">ETL Pipeline</strong> — Same pipeline re-runs on live data</li>
                <li><strong style="color: #fbbf24;">Go Live Dashboard</strong> — Interactive predictions & charts</li>
                <li><strong style="color: #fbbf24;">Strategy Engine</strong> — Backtesting & strategy simulation</li>
                <li><strong style="color: #fbbf24;">Cloud Deployment</strong> — Streamlit Cloud for public access</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Data Flow Diagram (Mermaid rendered via Streamlit) ───────────────
with st.expander("📐 View Detailed Data Flow Diagram", expanded=False):
    st.markdown(
        """
        ```
        ┌───────────────┐     ┌───────────────────────┐     ┌─────────────────────────────┐
        │   SimFin      │     │  Part 1: Analytics    │     │  Part 2: Web App            │
        │   Platform    │     │       (Offline)       │     │       (Online)              │
        │               │     │                       │     │                             │
        │  ┌─────────┐  │     │  ┌───────┐            │     │  ┌──────────┐               │
        │  │  Bulk   ├──┼────►│  │  ETL  ├──►Features │     │  │ PySimFin ├───┐           │
        │  │Download │  │     │  └───────┘    │       │     │  │ Wrapper  │   │           │
        │  └─────────┘  │     │               ▼       │     │  └──────────┘   ▼           │
        │               │     │  ┌───────┐  ┌───────┐ │     │            ┌───────┐        │
        │  ┌─────────┐  │     │  │  ML   ├─►│.joblib│─┼──┐  │            │  ETL  │        │
        │  │   API   ├──┼────►│  │ Train │  └───────┘ │  │  │            │(reuse)│        │
        │  │Endpoint │  │     │  └───────┘            │  │  │            └───┬───┘        │
        │  └─────────┘  │     └───────────────────────┘  │  │                │            │
        └───────────────┘                                │  │  ┌─────────────▼──────────┐ │
                                                         ├──┼─►│     Go Live Dashboard  │ │
                                                         │  │  └────────────────────────┘ │
                                                         │  │  ┌────────────────────────┐ │
                                                         ├──┼─►│     Model Insights     │ │
                                                         │  │  └────────────────────────┘ │
                                                         │  │  ┌────────────────────────┐ │
                                                         └──┼─►│  Backtester            │ │
                                                            │  │  ───► Strategy         │ │
                                                            │  └────────────────────────┘ │
                                                            └─────────────────────────────┘
        ```
        """,
    )

st.markdown("---")

# ── Companies We Track ───────────────────────────────────────────────
def _ticker_logo(image_path: str, fallback_icon: str, size: int = 56) -> str:
    """Return a base64 <img> tag for the company logo, or the emoji fallback."""
    abs_path = os.path.join(os.path.dirname(__file__), image_path)
    if os.path.isfile(abs_path):
        with open(abs_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = abs_path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return (
            f'<img src="data:{mime};base64,{data}" '
            f'style="width:{size}px;height:{size}px;object-fit:contain;margin-bottom:0.4rem;" />'
        )
    return f'<div class="icon">{fallback_icon}</div>'


st.markdown("## Companies We Track")
st.markdown("")

ticker_cols = st.columns(len(TICKERS))
for col, (ticker, info) in zip(ticker_cols, TICKERS.items()):
    with col:
        logo = _ticker_logo(info.get("image", ""), info["icon"], info.get("logo_size", 56))
        url = info.get("url", "#")
        st.markdown(
            f"""
            <a href="{url}" target="_blank" style="text-decoration: none;">
                <div class="feature-box">
                    {logo}
                    <h5>{ticker}</h5>
                    <p style="color: #94a3b8 !important; font-size: 0.8rem; margin: 0;">{info['name']}</p>
                    <p style="color: #a855f7 !important; font-size: 0.7rem; margin: 0;">{info['sector']}</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.markdown("---")

# ── How It Works ─────────────────────────────────────────────────────
st.markdown("## How It Works")
st.markdown("")

step_cols = st.columns(4)
steps = [
    ("1", "📥", "Data Ingestion", "Fetch real-time prices from SimFin via our PySimFin API wrapper"),
    ("2", "🔧", "ETL Process", "Clean, normalize, and compute 20+ technical features"),
    ("3", "🤖", "ML Prediction", "Classification model predicts next-day price direction"),
    ("4", "📊", "Signal & Action", "Generate BUY / SELL / HOLD signals with confidence scores"),
]

for col, (num, icon, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; min-height: 200px;">
                <div style="font-family: 'Orbitron', sans-serif; font-size: 2rem;
                            color: #00c853; ">
                    {icon}
                </div>
                <h4 style="font-family: 'Orbitron', sans-serif; color: #e2e8f0 !important;
                           font-size: 0.9rem; margin: 0.8rem 0 0.4rem 0;">
                    Step {num}: {title}
                </h4>
                <p style="color: #94a3b8 !important; font-size: 0.85rem; line-height: 1.5;">
                    {desc}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.markdown("---")

# ── Development Team ─────────────────────────────────────────────────
def _member_avatar(image_path: str) -> str:
    """Return an <img> tag with base64-encoded photo, or a fallback emoji div."""
    # image_path is relative to this file's directory (app/)
    abs_path = os.path.join(os.path.dirname(__file__), image_path)
    if os.path.isfile(abs_path):
        with open(abs_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = abs_path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return (
            f'<img src="data:{mime};base64,{data}" '
            f'style="width:90px;height:90px;border-radius:50%;object-fit:cover;'
            f'border:2px solid #00d4ff;margin-bottom:0.5rem;" />'
        )
    return '<div style="font-size:2.5rem;margin-bottom:0.5rem;">👤</div>'


st.markdown("## Development Team")
st.markdown("")

_LINKEDIN_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">
  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762
           0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-10h3v10zm-1.5
           -11.268c-.966 0-1.75-.784-1.75-1.75s.784-1.75 1.75-1.75
           1.75.784 1.75 1.75-.784 1.75-1.75 1.75zm13.5 11.268h-3v-5.604c0-1.337
           -.026-3.063-1.867-3.063-1.869 0-2.155 1.459-2.155 2.967v5.7h-3v-10h2.879
           v1.367h.041c.401-.761 1.381-1.563 2.843-1.563 3.041 0 3.604 2.002
           3.604 4.604v5.592z"/>
</svg>"""

team_cols = st.columns(len(TEAM_MEMBERS))
for col, member in zip(team_cols, TEAM_MEMBERS):
    with col:
        avatar = _member_avatar(member.get("image", ""))
        linkedin_url = member.get("linkedin", "")
        linkedin_btn = (
            f'<a href="{linkedin_url}" target="_blank" style="display:inline-flex;'
            f'align-items:center;gap:6px;margin-top:0.6rem;padding:5px 12px;'
            f'background:#0077b5;border-radius:6px;text-decoration:none;'
            f'color:white;font-size:0.75rem;font-weight:600;">'
            f'{_LINKEDIN_ICON}</a>'
        ) if linkedin_url else ""
        st.markdown(
            f"""
            <div class="team-card">
                {avatar}
                <h4>{member['name']}</h4>
                <div class="role">{member['role']}</div>
                <div class="focus">{member['focus']}</div>
                {linkedin_btn}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.markdown("---")

# ── Technology Stack ─────────────────────────────────────────────────
st.markdown("## Technology Stack")
st.markdown("")

tech_cols = st.columns(5)
techs = [
    ("🐍", "Python", "Core Language"),
    ("📊", "Streamlit", "Web Framework"),
    ("🧠", "Scikit-learn", "ML Library"),
    ("📈", "Plotly", "Visualization"),
    ("☁️", "Streamlit Cloud", "Deployment"),
]

for col, (icon, name, purpose) in zip(tech_cols, techs):
    with col:
        st.markdown(
            f"""
            <div class="feature-box" style="min-height: 120px;">
                <div class="icon">{icon}</div>
                <h5>{name}</h5>
                <p style="color: #94a3b8 !important; font-size: 0.75rem; margin: 0;">{purpose}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.markdown("")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0; border-top: 1px solid #1e293b; margin-top: 2rem;">
        <p style="color: #94a3b8 !important; font-size: 0.8rem;">
            AutoTrader v1.0 · Automated Daily Trading System · Group Assignment 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
