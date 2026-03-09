"""
📈 Go Live – Real-Time Predictions & Trading Signals
Displays market data, runs ETL + model, and shows predictions.
"""

import base64
import os

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Go Live | AutoTrader", page_icon="📈", layout="wide")

from utils.style import inject_custom_css
from utils.config import TICKERS, TICKER_LIST, MODEL_FEATURES
from utils.data_helpers import load_price_data, load_processed_data, get_latest_features, get_prediction_history
from utils.model import load_model
from utils.charts import (
    candlestick_chart, price_with_ma_chart, rsi_chart, macd_chart,
    prediction_timeline, confidence_gauge, prediction_distribution_chart,
    accuracy_over_time_chart, returns_distribution_chart,
)

inject_custom_css()

# ── Sidebar Controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Go Live Controls")
    st.markdown("---")

    selected_ticker = st.selectbox(
        "Select Company",
        TICKER_LIST,
        format_func=lambda t: f"{TICKERS[t]['icon']} {t} — {TICKERS[t]['name']}",
    )

    # Show selected company's header logo below the dropdown
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

    lookback_days = st.slider(
        "Historical Data (trading days)",
        min_value=60, max_value=504, value=252, step=20,
        help="Number of trading days to load for analysis.",
    )

    pred_window = st.slider(
        "Prediction History (days)",
        min_value=10, max_value=120, value=60, step=5,
        help="Number of days to show in the prediction timeline.",
    )

    st.markdown("---")
    api_key = st.session_state.get("simfin_api_key", "")
    if api_key:
        st.success("Live data from SimFin ✓")
    else:
        st.info("Demo mode — synthetic data")

# ── Page Header ──────────────────────────────────────────────────────
ticker_info = TICKERS[selected_ticker]


def _load_ticker_logo(image_path: str, size: int = 72) -> str:
    """Return a base64 <img> tag for the company logo, or empty string if not found."""
    abs_path = os.path.join(os.path.dirname(__file__), "..", image_path)
    if os.path.isfile(abs_path):
        with open(abs_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = abs_path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return (
            f'<img src="data:{mime};base64,{data}" '
            f'style="width:{size}px;height:{size}px;object-fit:contain;margin-bottom:0.5rem;" />'
        )
    return f'<span style="font-size:3rem;">{ticker_info["icon"]}</span>'


logo_html = _load_ticker_logo(ticker_info.get("header_image", ""), 72)

st.markdown(
    f"""
    <div style="margin-bottom: 0.5rem;">
        {logo_html}
        <div>
            <h1 style="margin: 0; font-size: 1.8rem;">{selected_ticker}</h1>
            <p style="color: #94a3b8 !important; margin: 0; font-size: 1rem;">
                {ticker_info['name']} · {ticker_info['sector']}
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load Data ────────────────────────────────────────────────────────
with st.spinner("Loading market data & running ETL pipeline..."):
    raw_df = load_price_data(selected_ticker, days=lookback_days, api_key=api_key)
    processed_df = load_processed_data(selected_ticker, days=lookback_days, api_key=api_key)
    model = load_model(selected_ticker)

# ── Latest Prediction ────────────────────────────────────────────────
st.markdown("---")

features = get_latest_features(selected_ticker, api_key=api_key)

if not features.empty:
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    prob_up = probabilities[1]
    prob_down = probabilities[0]
    direction = "UP" if prediction == 1 else "DOWN"
    confidence = abs(prob_up - 0.5) * 2

    # Signal determination
    if prediction == 1 and confidence > 0.3:
        signal, signal_class = "BUY", "signal-up"
    elif prediction == 0 and confidence > 0.3:
        signal, signal_class = "SELL", "signal-down"
    else:
        signal, signal_class = "HOLD", "signal-hold"

    # Signal Row
    col_signal, col_gauge1, col_gauge2 = st.columns([1.2, 1, 1])

    with col_signal:
        st.markdown("### 🎯 Today's Signal")

        badge_class = "badge-up" if direction == "UP" else "badge-down"
        st.markdown(
            f"""
            <div class="{signal_class}" style="padding: 1.5rem; margin-top: 0.5rem;">
                <p style="font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;
                          color: #94a3b8 !important; margin-bottom: 0.5rem; text-transform: uppercase;
                          letter-spacing: 2px;">
                    Next-Day Prediction
                </p>
                <span class="prediction-badge {badge_class}">{direction}</span>
                <p style="font-family: 'Orbitron', sans-serif; font-size: 1.8rem;
                          margin-top: 1rem; margin-bottom: 0.3rem;">
                    Signal: <span style="color: {'#00ff88' if signal == 'BUY' else '#ff3366' if signal == 'SELL' else '#fbbf24'};">
                    {signal}</span>
                </p>
                <p style="color: #94a3b8 !important; font-size: 0.85rem;">
                    Confidence: {confidence:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_gauge1:
        st.plotly_chart(confidence_gauge(prob_up, "UP"), use_container_width=True)

    with col_gauge2:
        st.plotly_chart(confidence_gauge(prob_down, "DOWN"), use_container_width=True)

    st.markdown("")

# ── Key Price Metrics ────────────────────────────────────────────────
if not processed_df.empty:
    latest = processed_df.iloc[-1]
    prev = processed_df.iloc[-2] if len(processed_df) > 1 else latest

    price_change = latest["price"] - prev["price"]
    price_change_pct = price_change / prev["price"] * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last Close", f"${latest['price']:.2f}", f"{price_change_pct:+.2f}%")
    m2.metric("RSI (14)", f"{latest['rsi_14']:.1f}")
    m3.metric("Volatility (10d)", f"{latest['volatility_10d']:.4f}")
    m4.metric("MACD", f"{latest['macd']:.5f}")
    m5.metric("ATR (14)", f"{latest['atr_14']:.4f}")

st.markdown("---")

# ── Price Charts ─────────────────────────────────────────────────────
st.markdown("### 📊 Market Data Analysis")

tab_candle, tab_tech, tab_indicators = st.tabs([
    "🕯️ Candlestick", "📉 Technical Analysis", "📏 Indicators"
])

with tab_candle:
    fig = candlestick_chart(processed_df, selected_ticker)
    st.plotly_chart(fig, use_container_width=True)

with tab_tech:
    fig = price_with_ma_chart(processed_df, selected_ticker)
    st.plotly_chart(fig, use_container_width=True)

with tab_indicators:
    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.plotly_chart(rsi_chart(processed_df), use_container_width=True)
    with col_macd:
        st.plotly_chart(macd_chart(processed_df), use_container_width=True)

st.markdown("---")

# ── Prediction History ───────────────────────────────────────────────
st.markdown("### 🔮 Prediction History")

pred_history = get_prediction_history(selected_ticker, model, n_days=pred_window, api_key=api_key)

if not pred_history.empty:
    # Summary metrics
    total_preds = len(pred_history)
    up_preds = (pred_history["prediction"] == 1).sum()
    down_preds = (pred_history["prediction"] == 0).sum()
    avg_confidence = pred_history["confidence"].mean()

    # Accuracy vs actual
    if "actual" in pred_history.columns and pred_history["actual"].notna().any():
        valid = pred_history.dropna(subset=["actual"])
        accuracy = (valid["prediction"] == valid["actual"]).mean()
    else:
        accuracy = None

    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pc1.metric("Total Predictions", total_preds)
    pc2.metric("UP Signals", f"{up_preds} ({up_preds/total_preds:.0%})")
    pc3.metric("DOWN Signals", f"{down_preds} ({down_preds/total_preds:.0%})")
    pc4.metric("Avg Confidence", f"{avg_confidence:.1%}")
    if accuracy is not None:
        pc5.metric("Accuracy", f"{accuracy:.1%}")
    else:
        pc5.metric("Accuracy", "N/A")

    st.markdown("")

    # Prediction timeline chart
    st.plotly_chart(
        prediction_timeline(pred_history, selected_ticker),
        use_container_width=True,
    )

    # Sub-charts
    col_dist, col_acc = st.columns(2)
    with col_dist:
        st.plotly_chart(
            prediction_distribution_chart(pred_history),
            use_container_width=True,
        )
    with col_acc:
        st.plotly_chart(
            accuracy_over_time_chart(pred_history),
            use_container_width=True,
        )

st.markdown("---")

# ── Feature Snapshot ─────────────────────────────────────────────────
st.markdown("### 🧮 Latest Feature Values")

if not features.empty:
    feature_data = features.iloc[0]
    avail_features = [f for f in MODEL_FEATURES if f in feature_data.index]

    # Display as a formatted table
    feat_df = pd.DataFrame({
        "Feature": avail_features,
        "Value": [f"{feature_data[f]:.6f}" for f in avail_features],
    })

    # Split into columns
    n = len(feat_df)
    mid = n // 2
    fc1, fc2 = st.columns(2)
    with fc1:
        st.dataframe(feat_df.iloc[:mid], use_container_width=True, hide_index=True)
    with fc2:
        st.dataframe(feat_df.iloc[mid:], use_container_width=True, hide_index=True)

st.markdown("---")

# ── Returns Analysis ─────────────────────────────────────────────────
st.markdown("### 📈 Returns Analysis")
st.plotly_chart(returns_distribution_chart(processed_df), use_container_width=True)

# ── Raw Data Explorer ────────────────────────────────────────────────
with st.expander("📋 View Raw Data", expanded=False):
    tab_raw, tab_processed = st.tabs(["Raw Prices", "Processed Features"])
    with tab_raw:
        st.dataframe(raw_df.tail(50), use_container_width=True, hide_index=True)
    with tab_processed:
        st.dataframe(processed_df.tail(50), use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("")
st.caption(
    "⚠️ Predictions are generated by a machine learning model and should not be considered financial advice. "
    "Past performance does not guarantee future results."
)
