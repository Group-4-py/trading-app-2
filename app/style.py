"""
Custom CSS for the futuristic trading system theme.
Injects dark-mode, neon-accent styling into Streamlit.
"""

import streamlit as st


def inject_custom_css():
    """Inject the full custom CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* ── Import Fonts ─────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

        /* ── Root Variables ───────────────────────────────── */
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1a1f2e;
            --accent-cyan: #00d4ff;
            --accent-green: #00ff88;
            --accent-red: #ff3366;
            --accent-purple: #a855f7;
            --accent-yellow: #fbbf24;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --border: #1e293b;
            --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.3);
            --glow-green: 0 0 20px rgba(0, 255, 136, 0.3);
            --glow-red: 0 0 20px rgba(255, 51, 102, 0.3);
        }

        /* ── Global Overrides ─────────────────────────────── */
        .stApp {
            background: var(--bg-primary) !important;
            font-family: 'Rajdhani', sans-serif !important;
        }

        .stApp > header {
            background: transparent !important;
        }

        /* Main content area */
        .main .block-container {
            padding-top: 2rem !important;
            max-width: 1200px !important;
        }

        /* ── Sidebar ──────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
            border-right: 1px solid var(--border) !important;
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li {
            color: var(--text-secondary) !important;
            font-family: 'Rajdhani', sans-serif !important;
        }

        /* ── Typography ───────────────────────────────────── */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            color: var(--text-primary) !important;
            letter-spacing: 1px;
        }

        h1 {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800 !important;
        }

        p, li, span, label, div {
            color: var(--text-primary) !important;
        }

        /* ── Metric Cards ─────────────────────────────────── */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, var(--bg-card), #1e2538) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stMetric"]:hover {
            border-color: var(--accent-cyan) !important;
            box-shadow: var(--glow-cyan) !important;
            transform: translateY(-2px);
        }

        [data-testid="stMetric"] label {
            color: var(--text-secondary) !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif !important;
            color: var(--accent-cyan) !important;
            font-weight: 700 !important;
            font-size: 2rem !important;
        }

        [data-testid="stMetricDelta"] svg {
            display: none;
        }

        /* ── Tabs ─────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-secondary) !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 4px !important;
            border: 1px solid var(--border) !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 8px !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.5px !important;
            padding: 8px 16px !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(168, 85, 247, 0.15)) !important;
            color: var(--accent-cyan) !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* ── Selectbox / Inputs ───────────────────────────── */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stDateInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        .stSelectbox > div > div:focus-within,
        .stMultiSelect > div > div:focus-within {
            border-color: var(--accent-cyan) !important;
            box-shadow: var(--glow-cyan) !important;
        }

        /* ── Buttons ──────────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(168, 85, 247, 0.15)) !important;
            border: 1px solid var(--accent-cyan) !important;
            color: var(--accent-cyan) !important;
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            letter-spacing: 1.5px !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(168, 85, 247, 0.3)) !important;
            box-shadow: var(--glow-cyan) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Expander ─────────────────────────────────────── */
        .streamlit-expanderHeader {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
        }

        /* ── DataFrame ────────────────────────────────────── */
        .stDataFrame {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }

        /* ── Plotly Charts ────────────────────────────────── */
        .stPlotlyChart {
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* ── Divider ──────────────────────────────────────── */
        hr {
            border-color: var(--border) !important;
            opacity: 0.5 !important;
        }

        /* ── Custom Card Helper ───────────────────────────── */
        .glass-card {
            background: linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(30, 41, 59, 0.6));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 212, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(0, 212, 255, 0.3);
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
        }

        .signal-up {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 255, 136, 0.05));
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .signal-down {
            background: linear-gradient(135deg, rgba(255, 51, 102, 0.1), rgba(255, 51, 102, 0.05));
            border: 1px solid rgba(255, 51, 102, 0.3);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .signal-hold {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(251, 191, 36, 0.05));
            border: 1px solid rgba(251, 191, 36, 0.3);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .neon-text-cyan {
            color: var(--accent-cyan) !important;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        .neon-text-green {
            color: var(--accent-green) !important;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }

        .neon-text-red {
            color: var(--accent-red) !important;
            text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
        }

        .hero-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00d4ff, #a855f7, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.2rem;
            color: var(--text-secondary);
            text-align: center;
            letter-spacing: 3px;
            text-transform: uppercase;
        }

        .team-card {
            background: linear-gradient(135deg, var(--bg-card), #1e2538);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            min-height: 180px;
        }

        .team-card:hover {
            border-color: var(--accent-purple);
            box-shadow: 0 0 25px rgba(168, 85, 247, 0.2);
            transform: translateY(-3px);
        }

        .team-card h4 {
            font-family: 'Orbitron', sans-serif !important;
            color: var(--accent-cyan) !important;
            font-size: 1rem;
            margin-bottom: 0.3rem;
        }

        .team-card .role {
            color: var(--accent-purple) !important;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .team-card .focus {
            color: var(--text-secondary) !important;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }

        .feature-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .feature-box:hover {
            border-color: var(--accent-cyan);
            box-shadow: var(--glow-cyan);
        }

        .feature-box .icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .feature-box h5 {
            font-family: 'Orbitron', sans-serif !important;
            color: var(--accent-cyan) !important;
            font-size: 0.8rem;
            margin-bottom: 0.3rem;
        }

        .prediction-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: 2px;
        }

        .badge-up {
            background: rgba(0, 255, 136, 0.15);
            color: #00ff88;
            border: 2px solid rgba(0, 255, 136, 0.4);
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }

        .badge-down {
            background: rgba(255, 51, 102, 0.15);
            color: #ff3366;
            border: 2px solid rgba(255, 51, 102, 0.4);
            text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
        }

        /* ── Scrollbar ────────────────────────────────────── */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-cyan);
        }

        /* ── Slider ───────────────────────────────────────── */
        .stSlider > div > div > div > div {
            background: var(--accent-cyan) !important;
        }

        /* ── Hide Streamlit branding ──────────────────────── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[dict]):
    """
    Render a row of styled metric cards.
    metrics: list of dicts with keys 'label', 'value', 'delta' (optional), 'color' (optional)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        delta = m.get("delta")
        col.metric(label=m["label"], value=m["value"], delta=delta)
