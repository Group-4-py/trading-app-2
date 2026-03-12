"""
Model Module – ML Model Interface

Loads the trained all-tickers LogisticRegression model and wraps it with
helper methods needed by the app (feature importance, model metrics).
Falls back to a DummyClassifier when no trained model is available.
"""

import os
import logging
import numpy as np
import pandas as pd

from utils.config import MODEL_PATH, MODEL_FEATURES, TICKER_DUMMIES

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Thin wrapper around the trained sklearn model that adds helper methods
    required by the Model Insights page (get_feature_importance, get_model_metrics).
    """

    def __init__(self, model, ticker: str = None):
        self._model = model
        self._ticker = ticker
        self.classes_ = getattr(model, "classes_", np.array([0, 1]))
        self.is_dummy = False
        self.model_path = os.path.join(MODEL_PATH, "all_tickers_model.joblib")

    # ── Prediction passthrough ───────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = _add_ticker_columns(X, self._ticker)
        return self._model.predict(X_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = _add_ticker_columns(X, self._ticker)
        return self._model.predict_proba(X_enc)

    # ── Helpers for Model Insights page ──────────────────────────────
    def get_feature_importance(self) -> pd.DataFrame:
        all_features = MODEL_FEATURES + TICKER_DUMMIES
        # Support both bare models and Pipeline objects
        clf = self._model
        if hasattr(clf, "named_steps"):
            clf = clf.named_steps.get("clf", clf)
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
            importances = importances / importances.sum()
        else:
            importances = np.ones(len(all_features)) / len(all_features)
        return pd.DataFrame({
            "feature": all_features,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_model_metrics(self) -> dict:
        clf = self._model
        if hasattr(clf, "named_steps"):
            step_names = " + ".join(type(step).__name__ for step in clf.named_steps.values())
            model_type = f"Pipeline({step_names})"
        else:
            model_type = type(clf).__name__
        return {
            "model_type": model_type,
            "model_path": self.model_path,
            "is_dummy": self.is_dummy,
        }


class DummyClassifier:
    """
    Heuristic-based dummy classifier used when no trained model is available.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self.is_dummy = True
        self.model_path = None
        self._rng = np.random.RandomState(seed)
        all_features = MODEL_FEATURES + TICKER_DUMMIES
        n = len(all_features)
        rng = np.random.RandomState(seed)
        importances = rng.dirichlet(np.ones(n) * 2)
        boost_map = {
            "rsi_14": 3.0, "macd": 2.5, "return_1d": 2.0,
            "volatility_10d": 1.8, "sma_20": 1.5, "bb_width": 1.5,
            "volume_ratio": 1.3, "atr_14": 1.2,
        }
        for i, feat in enumerate(all_features):
            if feat in boost_map:
                importances[i] *= boost_map[feat]
        self.feature_importances_ = importances / importances.sum()

    def _compute_score(self, row: pd.Series) -> float:
        score = 0.5
        if "return_1d" in row.index and pd.notna(row["return_1d"]):
            score += np.clip(row["return_1d"] * 5, -0.15, 0.15)
        if "rsi_14" in row.index and pd.notna(row["rsi_14"]):
            rsi = row["rsi_14"]
            if rsi < 30:
                score += 0.15
            elif rsi > 70:
                score -= 0.15
            else:
                score += (50 - rsi) / 200
        if "macd" in row.index and "macd_signal" in row.index:
            if pd.notna(row["macd"]) and pd.notna(row["macd_signal"]):
                score += 0.08 if row["macd"] > row["macd_signal"] else -0.08
        if "sma_20" in row.index and pd.notna(row["sma_20"]):
            score += 0.05 if row["sma_20"] < 0 else -0.05
        noise = self._rng.normal(0, 0.12)
        score += noise
        return np.clip(score, 0.05, 0.95)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = X.apply(self._compute_score, axis=1).values
        return np.column_stack([1 - scores, scores])

    def get_feature_importance(self) -> pd.DataFrame:
        all_features = MODEL_FEATURES + TICKER_DUMMIES
        return pd.DataFrame({
            "feature": all_features,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_model_metrics(self) -> dict:
        return {
            "model_type": "DummyClassifier (no trained model found)",
            "model_path": None,
            "is_dummy": self.is_dummy,
        }


def _add_ticker_columns(X: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Add one-hot ticker columns expected by the all-tickers model."""
    X = X.copy()
    for col in TICKER_DUMMIES:
        expected_ticker = col.replace("ticker_", "")
        X[col] = 1.0 if ticker and ticker.upper() == expected_ticker else 0.0
    return X


def load_model(ticker: str = None):
    """
    Load the trained all-tickers model and return a ModelWrapper.
    Falls back to DummyClassifier if no model file is found.
    """
    # Resolve path relative to the project root.
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_module_dir))
    model_file = os.path.join(_project_root, MODEL_PATH, "all_tickers_model.joblib")

    if os.path.exists(model_file):
        try:
            import joblib
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_model = joblib.load(model_file)
            # Patch for sklearn version mismatch on bare LogisticRegression
            if hasattr(raw_model, "named_steps"):
                clf = raw_model.named_steps.get("clf")
                if clf and not hasattr(clf, "multi_class"):
                    clf.multi_class = "auto"
            elif not hasattr(raw_model, "multi_class"):
                raw_model.multi_class = "auto"
            logger.info(f"Loaded trained model from {model_file}")
            return ModelWrapper(raw_model, ticker=ticker)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_file}: {e}")

    logger.info(f"No trained model found. Using DummyClassifier.")
    seed = hash(ticker) % 10000 if ticker else 42
    return DummyClassifier(seed=seed)


def calculate_model_metrics(model, X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> dict:
    """Evaluate a loaded model using a temporal train/test split."""
    metrics = model.get_model_metrics().copy()

    if X.empty or len(X) != len(y):
        metrics.update({
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "auc_roc": np.nan,
            "train_samples": None,
            "test_samples": int(len(y)),
        })
        return metrics

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    split_idx = int(len(X) * train_ratio)
    split_idx = min(max(split_idx, 1), len(X) - 1) if len(X) > 1 else len(X)

    X_test = X.iloc[split_idx:]
    y_true = pd.Series(y).iloc[split_idx:].astype(int)

    if X_test.empty or y_true.empty:
        metrics.update({
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "auc_roc": np.nan,
            "train_samples": int(split_idx),
            "test_samples": 0,
        })
        return metrics

    y_pred = pd.Series(model.predict(X_test)).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics.update({
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)) if y_true.nunique() > 1 else np.nan,
        "train_samples": int(split_idx),
        "test_samples": int(len(y_true)),
    })
    return metrics
