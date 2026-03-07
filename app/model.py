"""
Model Module – ML Model Interface & Dummy Classifier

Provides a unified interface for the trading prediction model.
Ships with a DummyClassifier for development/testing. When Part 1 is ready,
simply export your trained model as a .pkl file and update MODEL_PATH.

Usage:
    model = load_model("AAPL")          # Tries to load real model, falls back to dummy
    prediction = model.predict(features) # Returns 1 (UP) or 0 (DOWN)
    proba = model.predict_proba(features) # Returns [P(down), P(up)]
"""

import os
import logging
import numpy as np
import pandas as pd

from utils.config import MODEL_PATH, MODEL_FEATURES

logger = logging.getLogger(__name__)


class DummyClassifier:
    """
    Dummy classifier that simulates a trained binary classification model.

    This model uses a simple heuristic based on technical indicators to
    produce realistic-looking predictions. Replace with your real trained
    model from Part 1.

    The heuristic:
        - Positive short-term momentum (return_1d > 0) → slight bullish bias
        - RSI < 30 (oversold) → bullish signal
        - RSI > 70 (overbought) → bearish signal
        - MACD > signal → bullish
        - Price below SMA_20 → bearish
        + Random noise to simulate model uncertainty
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self.feature_names_ = MODEL_FEATURES
        self._rng = np.random.RandomState(seed)
        # Simulated feature importances (for Model Insights page)
        self.feature_importances_ = self._generate_feature_importances()

    def _generate_feature_importances(self) -> np.ndarray:
        """Generate realistic-looking feature importance scores."""
        rng = np.random.RandomState(self.seed)
        n = len(MODEL_FEATURES)
        importances = rng.dirichlet(np.ones(n) * 2)

        # Boost importance of key features
        boost_map = {
            "rsi_14": 3.0, "macd": 2.5, "return_1d": 2.0,
            "volatility_10d": 1.8, "sma_20": 1.5, "bb_width": 1.5,
            "volume_ratio": 1.3, "atr_14": 1.2,
        }
        for i, feat in enumerate(MODEL_FEATURES):
            if feat in boost_map:
                importances[i] *= boost_map[feat]

        # Normalize
        importances = importances / importances.sum()
        return importances

    def _compute_score(self, row: pd.Series) -> float:
        """Compute a heuristic score in [0, 1] for a single row."""
        score = 0.5  # neutral baseline

        # Momentum signal
        if "return_1d" in row.index and pd.notna(row["return_1d"]):
            score += np.clip(row["return_1d"] * 5, -0.15, 0.15)

        # RSI signal
        if "rsi_14" in row.index and pd.notna(row["rsi_14"]):
            rsi = row["rsi_14"]
            if rsi < 30:
                score += 0.15  # oversold → bullish
            elif rsi > 70:
                score -= 0.15  # overbought → bearish
            else:
                score += (50 - rsi) / 200  # mild signal

        # MACD signal
        if "macd" in row.index and "macd_signal" in row.index:
            if pd.notna(row["macd"]) and pd.notna(row["macd_signal"]):
                if row["macd"] > row["macd_signal"]:
                    score += 0.08
                else:
                    score -= 0.08

        # SMA trend signal
        if "sma_20" in row.index and pd.notna(row["sma_20"]):
            # sma_20 is normalized: (SMA/price - 1), so negative means price > SMA
            if row["sma_20"] < 0:
                score += 0.05  # price above SMA → bullish
            else:
                score -= 0.05

        # Add noise to simulate model uncertainty
        noise = self._rng.normal(0, 0.12)
        score += noise

        return np.clip(score, 0.05, 0.95)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels (0=DOWN, 1=UP).

        Args:
            X: DataFrame with feature columns matching MODEL_FEATURES.

        Returns:
            Array of predictions (0 or 1).
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: DataFrame with feature columns.

        Returns:
            Array of shape (n_samples, 2) with [P(down), P(up)].
        """
        scores = X.apply(self._compute_score, axis=1).values
        probas = np.column_stack([1 - scores, scores])
        return probas

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances."""
        return pd.DataFrame({
            "feature": MODEL_FEATURES,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_model_metrics(self) -> dict:
        """
        Return simulated model performance metrics.
        Replace with real metrics from your model evaluation in Part 1.
        """
        return {
            "accuracy": 0.573,
            "precision": 0.581,
            "recall": 0.563,
            "f1_score": 0.572,
            "auc_roc": 0.604,
            "train_samples": 3_750,
            "test_samples": 1_250,
            "model_type": "DummyClassifier (replace with your trained model)",
        }


def load_model(ticker: str = None):
    """
    Load the ML model for a given ticker.

    Attempts to load a real trained model from MODEL_PATH. If not found,
    falls back to the DummyClassifier.

    Args:
        ticker: Stock ticker (e.g., 'AAPL'). If models are per-ticker,
                this determines which model file to load.

    Returns:
        Model object with predict() and predict_proba() methods.
    """
    # ── Try loading a real model ─────────────────────────────────────
    if ticker:
        model_file = os.path.join(MODEL_PATH, f"model_{ticker.lower()}.pkl")
    else:
        model_file = os.path.join(MODEL_PATH, "model.pkl")

    if os.path.exists(model_file):
        try:
            import joblib
            model = joblib.load(model_file)
            logger.info(f"Loaded trained model from {model_file}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model from {model_file}: {e}")

    # ── Fallback to dummy ────────────────────────────────────────────
    logger.info(f"No trained model found for {ticker}. Using DummyClassifier.")
    seed = hash(ticker) % 10000 if ticker else 42
    return DummyClassifier(seed=seed)
