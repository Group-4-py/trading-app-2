"""
Data Helpers – Generate demo data and bridge between API / local files.

When the SimFin API key is available, this module uses PySimFin to fetch
real data. Otherwise, it generates realistic synthetic stock price data
for development and demo purposes.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import logging

from utils.config import TICKERS, TICKER_LIST
from utils.pysimfin import PySimFin, SimFinAPIError
from utils.etl import run_etl

logger = logging.getLogger(__name__)

# ── Realistic base prices and drift for each ticker ──────────────────
_BASE_PARAMS = {
    "AAPL": {"price": 185, "drift": 0.0004, "vol": 0.018},
    "MSFT": {"price": 410, "drift": 0.0003, "vol": 0.016},
    "GOOGL": {"price": 155, "drift": 0.0003, "vol": 0.019},
    "AMZN": {"price": 190, "drift": 0.0004, "vol": 0.020},
    "NVDA": {"price": 790, "drift": 0.0006, "vol": 0.028},
}


def generate_demo_prices(ticker: str, days: int = 504) -> pd.DataFrame:
    """
    Generate realistic synthetic stock price data using geometric Brownian motion.

    Args:
        ticker: Stock ticker symbol.
        days: Number of trading days to simulate (default ~2 years).

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj. Close, Volume.
    """
    params = _BASE_PARAMS.get(ticker, {"price": 150, "drift": 0.0003, "vol": 0.02})
    rng = np.random.RandomState(hash(ticker) % 2**31)

    # Generate prices via GBM
    base_price = params["price"]
    drift = params["drift"]
    vol = params["vol"]

    # Daily log returns
    log_returns = rng.normal(drift, vol, days)
    cumulative = np.exp(np.cumsum(log_returns))
    close_prices = base_price * cumulative

    # Generate OHLCV from close
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=days, freq="B")
    # Ensure dates matches exactly
    dates = dates[:days]
    close_prices = close_prices[:len(dates)]

    n = len(dates)

    # Intraday range
    daily_range = rng.uniform(0.005, 0.025, n) * close_prices
    high = close_prices + daily_range * rng.uniform(0.3, 0.7, n)
    low = close_prices - daily_range * rng.uniform(0.3, 0.7, n)
    open_prices = low + (high - low) * rng.uniform(0.2, 0.8, n)

    # Volume: base volume with random fluctuation
    base_vol = rng.uniform(30_000_000, 80_000_000)
    volume = (base_vol * rng.lognormal(0, 0.4, n)).astype(int)

    df = pd.DataFrame({
        "Date": dates[:n],
        "Open": np.round(open_prices, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close_prices, 2),
        "Adj. Close": np.round(close_prices, 2),
        "Volume": volume,
    })

    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(ticker: str, days: int = 504, api_key: str = None) -> pd.DataFrame:
    """
    Load raw price data for a ticker.

    Priority:
      1. If api_key is provided → fetch from SimFin API
      2. Else → generate demo data

    Args:
        ticker: Stock ticker symbol.
        days: Number of trading days to load.
        api_key: SimFin API key (optional).

    Returns:
        Raw DataFrame with OHLCV data.
    """
    # Try SimFin API first
    if api_key:
        try:
            client = PySimFin(api_key=api_key)
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
            df = client.get_share_prices(ticker, start=start_date, end=end_date)
            if not df.empty and len(df) >= 50:
                logger.info(f"Loaded {len(df)} rows from SimFin for {ticker}")
                return df
        except SimFinAPIError as e:
            logger.warning(f"SimFin API error for {ticker}: {e}")
        except Exception as e:
            logger.warning(f"Error fetching {ticker} from SimFin: {e}")

    # Fallback: demo data
    logger.info(f"Using demo data for {ticker}")
    return generate_demo_prices(ticker, days)


@st.cache_data(ttl=300, show_spinner=False)
def load_processed_data(ticker: str, days: int = 504, api_key: str = None) -> pd.DataFrame:
    """
    Load price data and run the full ETL pipeline.

    Returns:
        Processed DataFrame with all features and target.
    """
    raw = load_price_data(ticker, days, api_key)
    processed = run_etl(raw, include_target=True)
    return processed


def get_latest_features(ticker: str, api_key: str = None) -> pd.DataFrame:
    """
    Get the most recent feature row for live prediction.

    Returns:
        Single-row DataFrame with feature columns.
    """
    df = load_processed_data(ticker, days=100, api_key=api_key)
    if df.empty:
        return pd.DataFrame()

    from utils.config import MODEL_FEATURES
    available = [f for f in MODEL_FEATURES if f in df.columns]
    return df[available].iloc[[-1]]  # last row as DataFrame


def get_prediction_history(ticker: str, model, n_days: int = 60, api_key: str = None) -> pd.DataFrame:
    """
    Generate prediction history for the last n_days.

    Returns:
        DataFrame with date, price, prediction, probability columns.
    """
    from utils.config import MODEL_FEATURES

    df = load_processed_data(ticker, days=max(n_days + 100, 200), api_key=api_key)
    if df.empty:
        return pd.DataFrame()

    # Get last n_days rows
    df_recent = df.tail(n_days).copy()

    available = [f for f in MODEL_FEATURES if f in df_recent.columns]
    X = df_recent[available]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result = pd.DataFrame({
        "date": df_recent["date"].values,
        "price": df_recent["price"].values,
        "prediction": predictions,
        "prob_down": probabilities[:, 0],
        "prob_up": probabilities[:, 1],
        "confidence": np.abs(probabilities[:, 1] - 0.5) * 2,
        "actual": df_recent["target"].values if "target" in df_recent.columns else np.nan,
    })

    return result
