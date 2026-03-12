"""
ml/train_model.py – Train the all-tickers classification model.

Reads raw SimFin share-price bulk-download CSV, applies the exact same ETL
pipeline used by the web app, and trains a LogisticRegression classifier.
Saves the result to ml/model/all_tickers_model.joblib.

Usage (from project root):
    python ml/train_model.py
    python ml/train_model.py --data path/to/us-shareprices-daily.csv

Steps to run before this script:
  1. Download 'us-shareprices-daily.csv' from SimFin → Bulk Download
  2. Place it at data/raw/us-shareprices-daily.csv
  3. Run: python ml/train_model.py
  4. Commit ml/model/all_tickers_model.joblib to git
"""

import sys
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "us-shareprices-daily.csv"
MODEL_OUTPUT = PROJECT_ROOT / "ml" / "model" / "all_tickers_model.joblib"

# Add app/ to sys.path so we import the exact same package the web app uses
sys.path.insert(0, str(APP_DIR))

from utils.config import MODEL_FEATURES, TICKER_LIST
from utils.etl import run_etl

TICKERS = TICKER_LIST
PRICE_COLUMNS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"]


# ── Step 1: Load raw CSV ──────────────────────────────────────────────

def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    print(f"Loading raw data from: {csv_path}")
    source_tickers = set(TICKERS)
    chunks = []
    for chunk in pd.read_csv(csv_path, sep=";", usecols=PRICE_COLUMNS, chunksize=250_000):
        chunk = chunk[chunk["Ticker"].isin(source_tickers)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No rows found for source tickers: {sorted(source_tickers)}")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  {len(df):,} rows, {df['Ticker'].nunique()} unique tickers in file")
    return df


# ── Step 2: Apply ETL per ticker ─────────────────────────────────────

def process_ticker(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter one ticker and run the full app ETL pipeline on it."""
    df = raw_df[raw_df["Ticker"] == ticker].copy()
    if df.empty:
        print(f"  WARNING: no data for {ticker} in CSV — skipping")
        return pd.DataFrame()

    processed = run_etl(df, include_target=True)
    processed["ticker"] = ticker   # ensure ticker label is present after ETL
    print(f"  {ticker}: {len(processed):,} usable rows after ETL")
    return processed


def build_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning ETL for each ticker...")
    frames = [process_ticker(raw_df, t) for t in TICKERS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No data processed. Check the CSV path and ticker symbols.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset: {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


# ── Step 3: Build feature matrix ─────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """
    One-hot encode the ticker column and select MODEL_FEATURES + ticker dummies.

    drop_first=True means AAPL (alphabetically first) is the implicit baseline.
    The generated dummy column names must match TICKER_DUMMIES in app/utils/config.py.
    """
    df = pd.get_dummies(df, columns=["ticker"], drop_first=True)
    ticker_dummies = sorted(c for c in df.columns if c.startswith("ticker_"))

    feature_cols = [c for c in MODEL_FEATURES + ticker_dummies if c in df.columns]
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: these MODEL_FEATURES are missing from the ETL output: {missing}")

    X = df[feature_cols].dropna()
    y = df.loc[X.index, "target"]

    print(f"\nFeature matrix : {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"Ticker dummies : {ticker_dummies}")
    return X, y, ticker_dummies


# ── Step 4: Train ─────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Time-ordered split → StandardScaler → LogisticRegression pipeline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # preserve temporal ordering
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy  : {acc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"]))
    print(f"Train samples  : {len(X_train):,}")
    print(f"Test samples   : {len(X_test):,}")
    return pipeline


# ── Step 5: Save ──────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\nModel saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the all-tickers trading model.")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_CSV,
        help="Path to SimFin bulk share-price CSV (semicolon-separated).",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"\nERROR: data file not found at: {args.data}")
        print(
            "\nTo fix this:\n"
            "  1. Log in to SimFin (simfin.com)\n"
            "  2. Go to Bulk Download → Share Prices → US\n"
            "  3. Download 'us-shareprices-daily.csv'\n"
            f"  4. Place the file at: {args.data}"
        )
        sys.exit(1)

    raw_df   = load_raw_csv(args.data)
    combined = build_dataset(raw_df)
    X, y, ticker_dummies = prepare_features(combined)
    pipeline = train_model(X, y)
    save_model(pipeline, MODEL_OUTPUT)

    print("\n" + "=" * 60)
    print("IMPORTANT – verify app/utils/config.py TICKER_DUMMIES matches:")
    print(f"  TICKER_DUMMIES = {ticker_dummies}")
    print("Then commit ml/model/all_tickers_model.joblib to git.")
    print("=" * 60)


if __name__ == "__main__":
    main()
