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

# ── Team Members (Replace with real names) ───────────────────────────
TEAM_MEMBERS = [
    {"name": "Bojana Belincevic", "role": "ML Engineer", "focus": "ETL & Model Development"},
    {"name": "David Carrillo", "role": "Backend Developer", "focus": "API Wrapper & Data Pipeline"},
    {"name": "Sebastião Clemente", "role": "Frontend Developer", "focus": "App & Deployment"},
    {"name": "Bassem El Halawani", "role": "Data Analyst", "focus": "Feature Engineering & Strategy"},
    {"name": "Theo Henry", "role": "Product Lead", "focus": "Use Case Definition & Prioritization"},
    {"name": "Ocke Moulijn", "role": "Insights Analyst", "focus": "Model Validation & Testing"},
]
