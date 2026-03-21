"""
🧠 Model Insights – ML Methodology & Performance Analysis
Explains the ML pipeline, shows metrics, feature importance, and confusion matrix.
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))  # or os.path.dirname(os.path.dirname(...)) for pages
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

st.set_page_config(page_title="Model Insights | AutoTrader", page_icon="🧠", layout="wide")

from utils.style import inject_custom_css
from utils.config import TICKERS, TICKER_LIST, MODEL_FEATURES
from utils.model import load_model, calculate_model_metrics
from utils.data_helpers import load_processed_data, get_prediction_history
from utils.charts import (
    feature_importance_chart, confusion_matrix_chart,
    accuracy_over_time_chart, prediction_distribution_chart,
)

inject_custom_css()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Insights")
    st.markdown("---")

    selected_ticker = st.selectbox(
        "Select Company",
        TICKER_LIST,
        format_func=lambda t: f"{t} — {TICKERS[t]['name']}",
    )

    _sb_info = TICKERS[selected_ticker]
    _sb_logo_path = os.path.join(os.path.dirname(__file__), "..", _sb_info.get("header_image", ""))
    if os.path.isfile(_sb_logo_path):
        with open(_sb_logo_path, "rb") as _f:
            _sb_data = base64.b64encode(_f.read()).decode()
        _sb_ext = _sb_logo_path.rsplit(".", 1)[-1].lower()
        _sb_mime = "image/jpeg" if _sb_ext in ("jpg", "jpeg") else f"image/{_sb_ext}"
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem 0;">'
            f'<img src="data:{_sb_mime};base64,{_sb_data}" '
            f'style="max-width:100%;max-height:36px;object-fit:contain;" /></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    api_key = st.session_state.get("api_key_stored", "")

# ── Load Model & Data ────────────────────────────────────────────────
model = load_model(selected_ticker)
processed_df = load_processed_data(
    selected_ticker,
    days=504,
    api_key=api_key,
    include_target=True,
)
pred_history = get_prediction_history(selected_ticker, model, n_days=120, api_key=api_key)

available_features = [f for f in MODEL_FEATURES if f in processed_df.columns]
metric_source_df = processed_df.dropna(subset=available_features + ["target"]).copy() if available_features else pd.DataFrame()
if not metric_source_df.empty:
    metrics = calculate_model_metrics(
        model,
        metric_source_df[available_features],
        metric_source_df["target"],
    )
else:
    metrics = model.get_model_metrics()

# ── Page Header ──────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem;">Model Insights & Methodology</h1>
        <p style="color: #94a3b8; font-size: 1rem;">
            Deep dive into the machine learning pipeline powering our predictions
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Model Overview ───────────────────────────────────────────────────
st.markdown("### Model Overview")
st.markdown("")

ov1, ov2, ov3 = st.columns(3)

with ov1:
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="font-family: 'Orbitron', sans-serif; color: #00d4ff !important; font-size: 0.9rem;">
                🎯 Problem Definition
            </h4>
            <p style="color: #94a3b8 !important; line-height: 1.7; font-size: 0.9rem;">
                <strong style="color: #e2e8f0;">Task:</strong> Binary Classification<br>
                <strong style="color: #e2e8f0;">Target:</strong> Next-day price direction<br>
                <strong style="color: #e2e8f0;">Classes:</strong> UP (1) vs DOWN (0)<br>
                <strong style="color: #e2e8f0;">Frequency:</strong> Daily predictions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with ov2:
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="font-family: 'Orbitron', sans-serif; color: #00d4ff !important; font-size: 0.9rem;">
                🔧 Feature Engineering
            </h4>
            <p style="color: #94a3b8 !important; line-height: 1.7; font-size: 0.9rem;">
                <strong style="color: #e2e8f0;">Returns:</strong> 1d, 5d, 10d<br>
                <strong style="color: #e2e8f0;">Momentum:</strong> RSI, MACD<br>
                <strong style="color: #e2e8f0;">Trend:</strong> SMA, EMA, Bollinger<br>
                <strong style="color: #e2e8f0;">Volatility:</strong> ATR, Rolling Std
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with ov3:
    train_samples = "N/A" if metrics["train_samples"] is None else f"{metrics['train_samples']:,}"
    test_samples = "N/A" if metrics["test_samples"] is None else f"{metrics['test_samples']:,}"
    st.markdown(
        f"""
        <div class="glass-card">
            <h4 style="font-family: 'Orbitron', sans-serif; color: #00d4ff !important; font-size: 0.9rem;">
                📊 Dataset Info
            </h4>
            <p style="color: #94a3b8 !important; line-height: 1.7; font-size: 0.9rem;">
                <strong style="color: #e2e8f0;">Training:</strong> {train_samples} samples<br>
                <strong style="color: #e2e8f0;">Testing:</strong> {test_samples} samples<br>
                <strong style="color: #e2e8f0;">Features:</strong> {len(MODEL_FEATURES)}<br>
                <strong style="color: #e2e8f0;">Model:</strong> StandardScalar + LogitRegression
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Performance Metrics ──────────────────────────────────────────────
st.markdown("### Model Performance Metrics")
st.markdown("")

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Accuracy", "N/A" if pd.isna(metrics["accuracy"]) else f"{metrics['accuracy']:.1%}")
mc2.metric("Precision", "N/A" if pd.isna(metrics["precision"]) else f"{metrics['precision']:.1%}")
mc3.metric("Recall", "N/A" if pd.isna(metrics["recall"]) else f"{metrics['recall']:.1%}")
mc4.metric("F1 Score", "N/A" if pd.isna(metrics["f1_score"]) else f"{metrics['f1_score']:.1%}")
mc5.metric("AUC-ROC", "N/A" if pd.isna(metrics["auc_roc"]) else f"{metrics['auc_roc']:.3f}")

st.markdown("")

# Explanation
with st.expander("ℹ️ What do these metrics mean?", expanded=False):
    st.markdown(
        """
        - **Accuracy** — Fraction of correct predictions overall (both UP and DOWN).
        - **Precision** — When the model predicts UP, how often is it correct?
          High precision means fewer false positives.
        - **Recall** — Of all actual UP days, how many did the model correctly identify?
          High recall means fewer missed opportunities.
        - **F1 Score** — Harmonic mean of Precision and Recall. Balances both concerns.
        - **AUC-ROC** — Area Under the ROC Curve. Measures the model's ability to
          distinguish between UP and DOWN at all thresholds. 0.5 = random, 1.0 = perfect.

        > **Note:** Stock market prediction is inherently noisy. An accuracy of 55–60%
        > can still be profitable when combined with a good trading strategy and proper
        > risk management.
        """
    )

st.markdown("---")

# ── Feature Importance ───────────────────────────────────────────────
st.markdown("### Feature Importance")
st.markdown("")

fi_df = model.get_feature_importance()

col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.plotly_chart(feature_importance_chart(fi_df), use_container_width=True)

with col_table:
    st.markdown(
        """
        <div class="glass-card" style="padding: 0rem;">
            <h4 style="font-family: 'Orbitron', sans-serif; color: #00d4ff !important;
                       font-size: 0.85rem; margin-bottom: 0rem; margin-left: 1rem">
                Top Features by Importance
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_fi = fi_df.head(10).copy()
    top_fi["importance"] = top_fi["importance"].apply(lambda x: f"{x:.4f}")
    top_fi.index = range(1, len(top_fi) + 1)
    top_fi.index.name = "Rank"
    st.dataframe(top_fi, use_container_width=True)

st.markdown("---")

# ── Confusion Matrix ────────────────────────────────────────────────
st.markdown("### Confusion Matrix")
st.markdown("")

if not pred_history.empty and pred_history["actual"].notna().any():
    valid = pred_history.dropna(subset=["actual"])
    y_true = valid["actual"].astype(int).values
    y_pred = valid["prediction"].astype(int).values

    cm_col1, cm_col2 = st.columns([1.5, 1])

    with cm_col1:
        st.plotly_chart(
            confusion_matrix_chart(y_true, y_pred),
            use_container_width=True,
        )

    with cm_col2:
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=["DOWN", "UP"], output_dict=True)

        st.markdown(
            f"""
        <div class="glass-card" style="padding: 0rem;">
            <h4 style="font-family: 'Orbitron', sans-serif; color: #00d4ff !important;
                       font-size: 0.85rem; margin-bottom: 0rem; margin-left: 1rem">
                Classification Report ({len(valid)}-day window)
            </h4>
        </div>
        """,
            unsafe_allow_html=True,
        )

        report_df = pd.DataFrame(report).T
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

else:
    st.info("Confusion matrix will appear once actual values are available for comparison.")

st.markdown("---")

# ── Prediction Analysis ─────────────────────────────────────────────
st.markdown("### Prediction Analysis")
st.markdown("")

if not pred_history.empty:
    tab_acc, tab_dist, tab_corr = st.tabs([
        "📈 Accuracy Over Time",
        "📊 Probability Distribution",
        "🔗 Feature Correlations",
    ])

    with tab_acc:
        window_size = st.slider("Rolling window size", 10, 60, 20, key="acc_window")
        st.plotly_chart(
            accuracy_over_time_chart(pred_history, window=window_size),
            use_container_width=True,
        )

    with tab_dist:
        st.plotly_chart(
            prediction_distribution_chart(pred_history),
            use_container_width=True,
        )

    with tab_corr:
        # Feature correlation heatmap
        import plotly.graph_objects as go

        avail_features = [f for f in MODEL_FEATURES if f in processed_df.columns]
        if avail_features:
            corr_features = avail_features[:12]  # Top 12 for readability
            corr_matrix = processed_df[corr_features].corr()

            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_features, y=corr_features,
                colorscale=[[0, "#ff3366"], [0.5, "#0a0e17"], [1, "#00d4ff"]],
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=10),
            ))
            fig.update_layout(
                height=500,
                font=dict(family="Rajdhani, sans-serif", color="#94a3b8", size=11),
                paper_bgcolor="rgba(10, 14, 23, 0)",
                plot_bgcolor="rgba(10, 14, 23, 0)",
                title="Feature Correlation Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── ML Pipeline Explanation ──────────────────────────────────────────
st.markdown("### ML Pipeline Details")
st.markdown("")

tab_etl, tab_model, tab_eval = st.tabs([
    "🔧 ETL Process", "🤖 Model Architecture", "✅ Evaluation Methodology"
])

with tab_etl:
    st.markdown(
        """
        #### ETL Pipeline Steps

        Our ETL (Extract, Transform, Load) pipeline converts raw OHLCV price data
        into a feature-rich dataset suitable for machine learning:

        **1. Data Extraction** — Download historical share prices from SimFin (bulk or API).

        **2. Data Cleaning** — Handle missing values via forward-fill, remove duplicate dates,
        standardize column names, and convert types.

        **3. Feature Engineering** — Compute 20+ technical indicators including returns
        (1d, 5d, 10d), rolling volatility, moving averages (SMA, EMA), RSI-14, MACD with
        signal line, Bollinger Bands, volume ratios, and ATR-14. All price-based features
        are normalized by the current price to ensure stationarity.

        **4. Target Generation** — Binary label: 1 if next day's close > today's close
        (UP), 0 otherwise (DOWN).

        **5. Data Validation** — Drop rows with NaN values (from rolling window warmup)
        and verify feature completeness.
        """
    )

with tab_model:
    st.markdown(
        """
        #### Model Architecture

        **Model Type:** Binary Classification (UP / DOWN)

        **Current Status:** The app loads the shared trained model from
        `ml/model/all_tickers_model.joblib`. If that file is missing, it falls back to a
        heuristic DummyClassifier.

        **Recommended approaches (Part 1):**
        - **Logistic Regression** — Simple baseline, fast training
        - **Random Forest** — Captures non-linear feature interactions
        - **Gradient Boosting (XGBoost / LightGBM)** — State-of-the-art for tabular data
        - **SVM** — Effective in high-dimensional feature spaces

        **Input:** 20 numerical features (normalized technical indicators)

        **Output:** Binary prediction (0 = DOWN, 1 = UP) + class probabilities

        **Deployment Path:** `ml/model/all_tickers_model.joblib`

        **How the app uses it:**
        1. `utils.model.load_model()` loads `all_tickers_model.joblib`
        2. The selected ticker is encoded through ticker dummy columns
        3. All app pages use that same loaded model for inference
        """
    )

with tab_eval:
    st.markdown(
        """
        #### Evaluation Methodology

        **Train-Test Split:** Temporal split using 80% train / 20% test to avoid
        look-ahead bias. We never shuffle time-series data randomly.

        **Metrics Used:**
        - **Accuracy** — Overall correctness
        - **Precision & Recall** — Per-class performance
        - **F1 Score** — Balance between precision and recall
        - **AUC-ROC** — Threshold-independent discrimination ability
        - **Confusion Matrix** — Detailed breakdown of predictions vs actuals

        **Cross-Validation:** Time-series cross-validation with expanding window to
        respect temporal ordering.

        **Important Note:** Stock prediction is inherently noisy. Even models with
        ~50% accuracy can be profitable with proper risk management. The goal is not
        perfect prediction but consistent edge over random.
        """
    )
