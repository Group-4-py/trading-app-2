"""
Configuration for the Automated Daily Trading System.
Central place for all constants, tickers, and settings.
"""

# ── Supported Companies ──────────────────────────────────────────────
# Drop company logos into app/assets/logos/ with the matching filename (e.g. aapl.png).
# Supported formats: .png, .jpg, .jpeg, .svg, .webp
TICKERS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "icon": "", "image": "assets/logos/aapl.png", "logo_size": 56, "header_image": "assets/logos/header/aapl.png"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "icon": "", "image": "assets/logos/msft.png", "logo_size": 56, "header_image": "assets/logos/header/msft.png"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "icon": "", "image": "assets/logos/googl.png", "logo_size": 56, "header_image": "assets/logos/header/googl.png"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "icon": "", "image": "assets/logos/amzn.png", "logo_size": 56, "header_image": "assets/logos/header/amzn.png"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "icon": "", "image": "assets/logos/nvda.png", "logo_size": 56, "header_image": "assets/logos/header/nvda.png"},
}

TICKER_LIST = list(TICKERS.keys())

# ── SimFin API Configuration ─────────────────────────────────────────
SIMFIN_BASE_URL = "https://backend.simfin.com/api/v4"
SIMFIN_RATE_LIMIT = 0.5  # seconds between requests (max 2 req/s)

# ── Model Configuration ──────────────────────────────────────────────
MODEL_PATH = "models/"            # Directory where exported .pkl models live
MODEL_FEATURES = [                # Features the model expects (from ETL)
    "return_1d", "return_5d", "return_10d",
    "volatility_10d", "volatility_20d",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_width",
    "volume_sma_10", "volume_ratio",
    "atr_14",
]

# Model target: 1 = price goes UP, 0 = price goes DOWN
TARGET_COL = "target"

# ── Trading Strategy Defaults ────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST = 0.001  # 0.1% per trade

# ── App Theme Colors ─────────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827",
    "bg_card": "#1a1f2e",
    "accent_cyan": "#00d4ff",
    "accent_green": "#00ff88",
    "accent_red": "#ff3366",
    "accent_purple": "#a855f7",
    "accent_yellow": "#fbbf24",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "border": "#1e293b",
}

# ── Team Members ─────────────────────────────────────────────────────
# Drop each person's photo into app/assets/team/ with the matching filename.
# Supported formats: .jpg, .jpeg, .png, .webp
TEAM_MEMBERS = [
    {"name": "Bojana Belincevic", "role": "ML Engineer", "focus": "ETL & Model Development", "image": "assets/team/bojana.jpg", "linkedin": "https://www.linkedin.com/in/REPLACE_BOJANA"},
    {"name": "David Carrillo", "role": "Backend Developer", "focus": "API Wrapper & Data Pipeline", "image": "assets/team/david.jpg", "linkedin": "https://www.linkedin.com/in/davidcarrilloaguilera/"},
    {"name": "Sebastião Clemente", "role": "Frontend Developer", "focus": "App & Deployment", "image": "assets/team/sebastiao.jpg", "linkedin": "https://www.linkedin.com/in/sebastião-clemente/"},
    {"name": "Bassem El Halawani", "role": "Data Analyst", "focus": "Feature Engineering & Strategy", "image": "assets/team/bassem.jpg", "linkedin": "https://www.linkedin.com/in/bassemelhalawani/"},
    {"name": "Theo Henry", "role": "Product Lead", "focus": "Product Direction Shaping & Prioritization", "image": "assets/team/theo.jpg", "linkedin": "https://www.linkedin.com/in/theohenry1/"},
    {"name": "Ocke Moulijn", "role": "Insights Analyst", "focus": "Model Validation & Testing", "image": "assets/team/ocke.jpg", "linkedin": "https://www.linkedin.com/in/ocke-moulijn-8b4256252/"},
]
