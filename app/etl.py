"""
ETL Module – Extract, Transform, Load Pipeline

Transforms raw share-price DataFrames (from SimFin or bulk CSV) into a
feature-rich dataset ready for the ML classification model.

Transformations applied:
  1. Clean & validate raw data
  2. Compute daily returns (1d, 5d, 10d)
  3. Compute rolling volatility (10d, 20d)
  4. Compute moving averages (SMA 5/10/20/50, EMA 12/26)
  5. Compute RSI-14
  6. Compute MACD + signal line
  7. Compute Bollinger Bands (20, 2σ)
  8. Compute volume features
  9. Compute ATR-14
  10. Generate the binary target column
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Clean and validate raw share-price data.

    Expects columns like: Date, Open, High, Low, Close, Adj. Close, Volume.
    Handles missing values, duplicates, and type conversions.
    """
    df = df.copy()

    # Standardize column names
    col_map = {
        "Adj. Close": "adj_close",
        "Adj Close": "adj_close",
    }
    df = df.rename(columns=col_map)

    # Ensure lowercase
    df.columns = [c.lower().replace(" ", "_").replace(".", "") for c in df.columns]

    # Ensure date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("DataFrame must contain a 'Date' column.")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Forward-fill missing price values (common for adj_close)
    price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill()

    # Use adj_close if available, else close
    if "adj_close" in df.columns and df["adj_close"].notna().any():
        df["price"] = df["adj_close"]
    elif "close" in df.columns:
        df["price"] = df["close"]
    else:
        raise ValueError("DataFrame must contain 'Close' or 'Adj. Close' column.")

    # Drop rows where price is still NaN
    df = df.dropna(subset=["price"]).reset_index(drop=True)

    logger.info(f"Cleaned data: {len(df)} rows, date range {df['date'].min()} to {df['date'].max()}")
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Compute daily, 5-day, and 10-day returns."""
    df = df.copy()
    df["return_1d"] = df["price"].pct_change(1)
    df["return_5d"] = df["price"].pct_change(5)
    df["return_10d"] = df["price"].pct_change(10)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Compute rolling volatility (std of returns)."""
    df = df.copy()
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Compute SMA and EMA indicators (normalized by price)."""
    df = df.copy()
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["price"].rolling(window=w).mean() / df["price"] - 1
    df["ema_12"] = df["price"].ewm(span=12, adjust=False).mean() / df["price"] - 1
    df["ema_26"] = df["price"].ewm(span=26, adjust=False).mean() / df["price"] - 1
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Step 5: Compute RSI (Relative Strength Index)."""
    df = df.copy()
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Compute MACD and signal line."""
    df = df.copy()
    ema_12 = df["price"].ewm(span=12, adjust=False).mean()
    ema_26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = (ema_12 - ema_26) / df["price"]  # Normalize
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Step 7: Compute Bollinger Bands (normalized)."""
    df = df.copy()
    sma = df["price"].rolling(window=window).mean()
    std = df["price"].rolling(window=window).std()
    df["bb_upper"] = (sma + num_std * std) / df["price"] - 1
    df["bb_lower"] = (sma - num_std * std) / df["price"] - 1
    df["bb_width"] = (2 * num_std * std) / df["price"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 8: Compute volume-based features."""
    df = df.copy()
    if "volume" in df.columns:
        df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"].replace(0, np.nan)
        # Normalize volume_sma_10
        df["volume_sma_10"] = df["volume_sma_10"] / df["volume_sma_10"].max()
    else:
        df["volume_sma_10"] = 0
        df["volume_ratio"] = 1
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Step 9: Compute Average True Range (normalized)."""
    df = df.copy()
    if all(c in df.columns for c in ["high", "low", "close"]):
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=period).mean() / df["price"]
    else:
        df["atr_14"] = df["volatility_10d"]  # fallback
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10: Generate the binary target column.
    target = 1 if next day's close > today's close (price goes UP)
    target = 0 if next day's close <= today's close (price goes DOWN)
    """
    df = df.copy()
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return df


def run_etl(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Execute the full ETL pipeline on a raw share-price DataFrame.

    Args:
        df: Raw DataFrame with at least Date, Close columns.
        include_target: Whether to add the target column (True for training,
                        False for live prediction).

    Returns:
        Feature-rich DataFrame ready for ML model.
    """
    logger.info(f"Starting ETL pipeline on {len(df)} rows...")

    df = clean_raw_prices(df)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_atr(df)

    if include_target:
        df = add_target(df)

    # Drop rows with NaN in features (from rolling windows)
    df = df.dropna().reset_index(drop=True)

    logger.info(f"ETL complete: {len(df)} rows, {len(df.columns)} columns.")
    return df
