"""Helpers for running the shared ETL pipeline from CLI scripts."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.config import TICKER_LIST
from utils.etl import run_etl

PRICE_COLUMNS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"]


def load_share_prices(path: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    """Load only the requested share-price rows from the SimFin bulk file."""
    source_tickers = set(tickers) if tickers else None
    chunks = []
    for chunk in pd.read_csv(path, sep=";", usecols=PRICE_COLUMNS, chunksize=250_000):
        if source_tickers:
            chunk = chunk[chunk["Ticker"].isin(source_tickers)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return pd.concat(chunks, ignore_index=True)


def filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return one ticker sorted chronologically."""
    out = df[df["Ticker"] == ticker].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)


def run_etl_for_ticker(
    ticker: str,
    share_prices_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Run the exact same ETL pipeline used by the app and model."""
    prices = load_share_prices(share_prices_path, [ticker])
    return run_etl_for_ticker_from_df(ticker, prices, output_path)


def run_etl_for_ticker_from_df(
    ticker: str,
    prices: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Run the shared ETL for one ticker from an already loaded price frame."""
    prices_ticker = filter_ticker(prices, ticker)
    features = run_etl(prices_ticker, include_target=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


__all__ = [
    "TICKER_LIST",
    "load_share_prices",
    "filter_ticker",
    "run_etl_for_ticker",
    "run_etl_for_ticker_from_df",
]
