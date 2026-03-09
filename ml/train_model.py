# ml/train_model.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def load_all_processed_data(processed_dir: Path) -> pd.DataFrame:
    """Load and concatenate processed datasets for all tickers."""
    dfs = []
    for ticker in TICKERS:
        path = processed_dir / f"{ticker}.parquet"
        df = pd.read_parquet(path)
        df["ticker"] = ticker
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full


def split_features_target(df: pd.DataFrame):
    """
    Separate features (X) and target (y).
    Include ticker as a categorical feature via one-hot encoding.
    Uses the engineered features that match Features-used.csv.
    """
    df = df.copy()

    # One‑hot encode ticker
    df = pd.get_dummies(df, columns=["ticker"], drop_first=True)

    feature_cols = [
        "close",
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
    ] + [c for c in df.columns if c.startswith("ticker_")]

    # Keep only columns that actually exist (defensive)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["target"]      # from etl_utils.add_target_column

    return X, y, feature_cols


def train_model(X, y):
    """Train a simple Logistic Regression classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,  # preserve time order
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return model


def save_model(model, path: Path) -> None:
    """Persist trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    model_path = project_root / "ml" / "models" / "all_tickers_model.joblib"

    print(f"Loading processed data from: {processed_dir}")
    df = load_all_processed_data(processed_dir)

    X, y, feature_cols = split_features_target(df)
    print(f"Using features: {feature_cols}")

    model = train_model(X, y)
    save_model(model, model_path)


if __name__ == "__main__":
    main()


