"""
Configuration for the Automated Daily Trading System.
Central place for all constants, tickers, and settings.
"""

# ── Supported Companies ──────────────────────────────────────────────
TICKERS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "icon": "🍎"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "icon": "🪟"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "icon": "🔍"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "icon": "📦"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "icon": "🎮"},
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
    "bg_primary": "#090e1a",
    "bg_secondary": "#0d1526",
    "bg_card": "#111d30",
    "accent_orange": "#f5820d",
    "accent_green": "#22d07a",
    "accent_red": "#e8384f",
    "accent_blue": "#4a9eff",
    "accent_yellow": "#f5a623",
    "text_primary": "#f0f4f8",
    "text_secondary": "#8b9ec0",
    "border": "#1a2a44",
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
