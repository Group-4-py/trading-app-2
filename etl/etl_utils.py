# etl/etl_utils.py

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------- I/O ----------

def load_share_prices(path: Path) -> pd.DataFrame:
    """Load full SimFin daily share prices CSV."""
    df = pd.read_csv(
        path,
        sep=";",        # SimFin bulk files use ';'
        engine="python"
    )
    return df


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed features as parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


# ---------- basic cleaning / filtering ----------

def filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter prices for a single ticker and sort by date."""
    out = df[df["Ticker"] == ticker].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date")

    # Standardize column names expected by feature functions
    out.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
        errors="ignore",
    )

    # If there is no separate adj_close column, use close
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]

    # Convenience alias used in Features-used.csv
    out["price"] = out["close"]

    return out


# ---------- technical features ----------

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic returns and simple moving averages.
    Matches the columns: return_1d, return_5d, return_10d,
    sma_5, sma_10, sma_20, sma_50.
    """
    df = df.copy()

    # Calculate short-, medium-, and long-term percentage returns
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    # Compute moving averages to identify general price trends across different time horizons
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility (std of daily returns)."""
    df = df.copy()

    daily_ret = df["close"].pct_change()
    df["volatility_10d"] = daily_ret.rolling(10).std()
    df["volatility_20d"] = daily_ret.rolling(20).std()

    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_trend_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA12, EMA26, MACD, MACD signal, RSI14.
    """
    df = df.copy()

    # EMAs
    df["ema_12"] = _ema(df["close"], span=12)
    df["ema_26"] = _ema(df["close"], span=26)

    # MACD and signal
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = _ema(df["macd"], span=9)

    # RSI (14-period)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()

    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


def add_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Bollinger bands (20-day, 2 std) and width:
    bb_upper, bb_lower, bb_width.
    """
    df = df.copy()

    # Calculate Bollinger Bands bounds to measure price volatility and potential overbought/oversold levels
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()

    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume moving average and ratio."""
    df = df.copy()

    # Analyze trading volume spikes compared to the recent average
    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_10"]

    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) over given period.
    """
    df = df.copy()

    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift()).abs()
    low_close_prev = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(period).mean()

    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary target:
    target = 1 if next day's close > today's close, else 0.
    """
    df = df.copy()
    df["close_next_day"] = df["close"].shift(-1)
    df["target"] = (df["close_next_day"] > df["close"]).astype(int)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature-engineering pipeline.
    """
    # Sequentially build out the target and all technical indicators needed for ML modeling
    df = add_basic_features(df)
    df = add_volatility_features(df)
    df = add_trend_momentum_features(df)
    df = add_bollinger_features(df)
    df = add_volume_features(df)
    df = add_atr(df)
    df = add_target_column(df)
    
    return df


# ---------- final column selection ----------

def select_feature_columns(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Select and order columns to match Features-used.csv.
    Optionally attach a ticker column.
    """
    df = df.copy()

    if ticker is not None and "ticker" not in df.columns:
        df["ticker"] = ticker

    cols = [
        "Date",          # date
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "price",
        "return_1d",
        "return_5d",
        "return_10d",
        "volatility_10d",
        "volatility_20d",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "volume_sma_10",
        "volume_ratio",
        "atr_14",
        "target",
    ]

    # Keep only columns that exist, in the right order
    cols_existing = [c for c in cols if c in df.columns]
    out = df[cols_existing].dropna().reset_index(drop=True)

    # Rename Date -> date if you prefer lower case in parquet
    out.rename(columns={"Date": "date"}, inplace=True)

    return out


def run_etl_from_dataframe(
    ticker: str,
    prices_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Run sequence of ETL steps on an already loaded SimFin DataFrame.
    """
    # 1. Extract and standardize column structures just for the target stock
    prices_ticker = filter_ticker(prices_df, ticker)
    
    # 2. Apply the full sequence of mathematical transforms to create predictors
    prices_feat = add_all_features(prices_ticker)
    
    # 3. Filter down to the final schema required for downstream machine learning
    features = select_feature_columns(prices_feat, ticker=ticker)
    
    # 4. Save the prepared dataset to disk in compressed parquet format
    save_features(features, output_path)

def run_etl_for_ticker(
    ticker: str,
    share_prices_path: Path,
    output_path: Path,
) -> None:
    """
    Full ETL for a single ticker:
    load raw CSV -> filter -> feature engineering -> select cols -> save.
    """
    # 1. Load historical data block into memory
    prices = load_share_prices(share_prices_path)
    
    # Delegate to the DataFrame-based function
    run_etl_from_dataframe(ticker, prices, output_path)


