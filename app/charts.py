"""
Chart Helpers – Plotly chart builders with the futuristic dark theme.

All charts follow the app's dark theme with neon accents.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Theme Constants ──────────────────────────────────────────────────
BG_COLOR = "rgba(10, 14, 23, 0)"
PAPER_COLOR = "rgba(10, 14, 23, 0)"
GRID_COLOR = "rgba(30, 41, 59, 0.5)"
TEXT_COLOR = "#94a3b8"
CYAN = "#00d4ff"
GREEN = "#00ff88"
RED = "#ff3366"
PURPLE = "#a855f7"
YELLOW = "#fbbf24"
FONT = "Rajdhani, sans-serif"

_LAYOUT_DEFAULTS = dict(
    font=dict(family=FONT, color=TEXT_COLOR, size=13),
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    hoverlabel=dict(bgcolor="#1a1f2e", font_size=12, font_family=FONT),
)


def _apply_defaults(fig, height=400, **kwargs):
    """Apply default layout styling."""
    fig.update_layout(**_LAYOUT_DEFAULTS, height=height, **kwargs)
    return fig


# ── 1. Candlestick Chart ─────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """
    OHLCV candlestick chart with volume bars.

    Expects columns: date, open, high, low, close, volume.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
        name="Price",
    ), row=1, col=1)

    # Volume bars
    colors = [GREEN if c >= o else RED for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], marker_color=colors,
        opacity=0.4, name="Volume", showlegend=False,
    ), row=2, col=1)

    fig = _apply_defaults(fig, height=500, title=f"{ticker} Price & Volume")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    return fig


# ── 2. Price with Moving Averages ────────────────────────────────────

def price_with_ma_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Price line with SMA overlays and Bollinger Bands."""
    fig = go.Figure()

    # Bollinger Bands (filled)
    if "bb_upper" in df.columns:
        bb_upper = df["price"] * (1 + df["bb_upper"])
        bb_lower = df["price"] * (1 + df["bb_lower"])
        fig.add_trace(go.Scatter(
            x=df["date"], y=bb_upper, mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=bb_lower, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(168, 85, 247, 0.1)",
            name="Bollinger Bands",
        ))

    # Price
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"], mode="lines",
        line=dict(color=CYAN, width=2), name="Price",
    ))

    # SMAs
    for period, color in [(5, YELLOW), (20, PURPLE), (50, GREEN)]:
        col = f"sma_{period}"
        if col in df.columns:
            sma_abs = df["price"] * (1 + df[col])
            fig.add_trace(go.Scatter(
                x=df["date"], y=sma_abs, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"SMA {period}",
            ))

    return _apply_defaults(fig, height=420, title=f"{ticker} Technical Analysis")


# ── 3. RSI Chart ─────────────────────────────────────────────────────

def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI-14 chart with overbought/oversold zones."""
    fig = go.Figure()

    # Overbought / Oversold zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255, 51, 102, 0.08)",
                  line_width=0, annotation_text="Overbought",
                  annotation_position="top left")
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0, 255, 136, 0.08)",
                  line_width=0, annotation_text="Oversold",
                  annotation_position="bottom left")

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rsi_14"], mode="lines",
        line=dict(color=PURPLE, width=2), name="RSI 14",
    ))

    fig.add_hline(y=70, line_dash="dash", line_color=RED, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color=TEXT_COLOR, opacity=0.3)

    return _apply_defaults(fig, height=250, title="RSI (14)")


# ── 4. MACD Chart ────────────────────────────────────────────────────

def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD line, signal line, and histogram."""
    fig = go.Figure()

    macd_hist = df["macd"] - df["macd_signal"]
    colors = [GREEN if v >= 0 else RED for v in macd_hist]

    fig.add_trace(go.Bar(
        x=df["date"], y=macd_hist, marker_color=colors,
        opacity=0.5, name="Histogram",
    ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd"], mode="lines",
        line=dict(color=CYAN, width=2), name="MACD",
    ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd_signal"], mode="lines",
        line=dict(color=YELLOW, width=1.5, dash="dot"), name="Signal",
    ))

    return _apply_defaults(fig, height=250, title="MACD")


# ── 5. Prediction Timeline ──────────────────────────────────────────

def prediction_timeline(pred_df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Price chart with prediction markers (UP=green, DOWN=red)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["price"], mode="lines",
        line=dict(color=CYAN, width=2), name="Price",
    ))

    # UP predictions
    up_mask = pred_df["prediction"] == 1
    fig.add_trace(go.Scatter(
        x=pred_df.loc[up_mask, "date"], y=pred_df.loc[up_mask, "price"],
        mode="markers", marker=dict(color=GREEN, size=6, symbol="triangle-up"),
        name="Predict UP",
    ))

    # DOWN predictions
    down_mask = pred_df["prediction"] == 0
    fig.add_trace(go.Scatter(
        x=pred_df.loc[down_mask, "date"], y=pred_df.loc[down_mask, "price"],
        mode="markers", marker=dict(color=RED, size=6, symbol="triangle-down"),
        name="Predict DOWN",
    ))

    return _apply_defaults(fig, height=400, title=f"{ticker} Prediction History")


# ── 6. Confidence Gauge ─────────────────────────────────────────────

def confidence_gauge(probability: float, direction: str = "UP") -> go.Figure:
    """Semicircular gauge showing prediction confidence."""
    color = GREEN if direction == "UP" else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=36, family="Orbitron, sans-serif", color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_COLOR, tickfont=dict(size=10)),
            bar=dict(color=color),
            bgcolor="rgba(26, 31, 46, 0.8)",
            borderwidth=2,
            bordercolor="rgba(30, 41, 59, 0.8)",
            steps=[
                dict(range=[0, 40], color="rgba(255, 51, 102, 0.15)"),
                dict(range=[40, 60], color="rgba(251, 191, 36, 0.15)"),
                dict(range=[60, 100], color="rgba(0, 255, 136, 0.15)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=probability * 100),
        ),
        title=dict(text=f"Predict {direction}", font=dict(size=16, family="Rajdhani, sans-serif")),
    ))
    return _apply_defaults(fig, height=250)


# ── 7. Feature Importance ────────────────────────────────────────────

def feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    fi_df = fi_df.sort_values("importance", ascending=True).tail(15)

    colors = [CYAN if v > fi_df["importance"].median() else PURPLE for v in fi_df["importance"]]

    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in fi_df["importance"]],
        textposition="outside", textfont=dict(size=11),
    ))

    return _apply_defaults(fig, height=450, title="Feature Importance (Top 15)")


# ── 8. Confusion Matrix ─────────────────────────────────────────────

def confusion_matrix_chart(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Interactive confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = ["DOWN (0)", "UP (1)"]

    # Normalize for display
    cm_pct = cm.astype(float) / cm.sum() * 100

    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=text, texttemplate="%{text}",
        colorscale=[[0, "rgba(10, 14, 23, 0.8)"], [1, CYAN]],
        showscale=False,
    ))

    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )

    return _apply_defaults(fig, height=350, title="Confusion Matrix")


# ── 9. Prediction Distribution ───────────────────────────────────────

def prediction_distribution_chart(pred_df: pd.DataFrame) -> go.Figure:
    """Distribution of UP vs DOWN probabilities."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pred_df["prob_up"], name="P(UP)",
        marker_color=GREEN, opacity=0.6, nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=pred_df["prob_down"], name="P(DOWN)",
        marker_color=RED, opacity=0.6, nbinsx=30,
    ))

    fig.update_layout(barmode="overlay")
    return _apply_defaults(fig, height=300, title="Prediction Probability Distribution")


# ── 10. Backtesting Portfolio Chart ──────────────────────────────────

def portfolio_chart(
    backtest_results: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame = None,
    dates: pd.Series = None,
) -> go.Figure:
    """
    Multi-line portfolio value chart comparing strategies.

    Args:
        backtest_results: Dict of {strategy_name: backtest_df}
        benchmark: Optional benchmark (buy-and-hold) DataFrame.
        dates: Date series for x-axis.
    """
    fig = go.Figure()
    colors = [CYAN, PURPLE, YELLOW, GREEN]

    for i, (name, bt_df) in enumerate(backtest_results.items()):
        x = dates[:len(bt_df)] if dates is not None else bt_df["step"]
        fig.add_trace(go.Scatter(
            x=x, y=bt_df["portfolio_value"], mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
            name=name,
        ))

    if benchmark is not None:
        x = dates[:len(benchmark)] if dates is not None else benchmark["step"]
        fig.add_trace(go.Scatter(
            x=x, y=benchmark["portfolio_value"], mode="lines",
            line=dict(color=TEXT_COLOR, width=1.5, dash="dash"),
            name="Benchmark (Buy & Hold)",
        ))

    return _apply_defaults(fig, height=420, title="Portfolio Performance Comparison")


# ── 11. Drawdown Chart ──────────────────────────────────────────────

def drawdown_chart(
    backtest_df: pd.DataFrame,
    dates: pd.Series = None,
    strategy_name: str = "",
) -> go.Figure:
    """Drawdown chart showing portfolio dips from peak."""
    cummax = backtest_df["portfolio_value"].cummax()
    drawdown = (backtest_df["portfolio_value"] - cummax) / cummax * 100

    x = dates[:len(drawdown)] if dates is not None else backtest_df["step"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=drawdown, mode="lines", fill="tozeroy",
        fillcolor="rgba(255, 51, 102, 0.2)",
        line=dict(color=RED, width=1.5),
        name="Drawdown",
    ))

    return _apply_defaults(fig, height=250, title=f"Drawdown – {strategy_name}")


# ── 12. Trade Actions Scatter ────────────────────────────────────────

def trade_actions_chart(
    backtest_df: pd.DataFrame,
    dates: pd.Series = None,
) -> go.Figure:
    """Price chart with BUY/SELL markers from strategy."""
    x = dates[:len(backtest_df)] if dates is not None else backtest_df["step"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=backtest_df["price"], mode="lines",
        line=dict(color=CYAN, width=1.5), name="Price",
    ))

    buys = backtest_df[backtest_df["action"] == "BUY"]
    sells = backtest_df[backtest_df["action"] == "SELL"]

    buy_x = dates.iloc[buys.index] if dates is not None else buys["step"]
    sell_x = dates.iloc[sells.index] if dates is not None else sells["step"]

    fig.add_trace(go.Scatter(
        x=buy_x, y=buys["price"], mode="markers",
        marker=dict(color=GREEN, size=10, symbol="triangle-up", line=dict(width=1, color="white")),
        name="BUY",
    ))

    fig.add_trace(go.Scatter(
        x=sell_x, y=sells["price"], mode="markers",
        marker=dict(color=RED, size=10, symbol="triangle-down", line=dict(width=1, color="white")),
        name="SELL",
    ))

    return _apply_defaults(fig, height=380, title="Trade Actions")


# ── 13. Accuracy Over Time ──────────────────────────────────────────

def accuracy_over_time_chart(pred_df: pd.DataFrame, window: int = 20) -> go.Figure:
    """Rolling accuracy of predictions over time."""
    if "actual" not in pred_df.columns:
        return go.Figure()

    correct = (pred_df["prediction"] == pred_df["actual"]).astype(int)
    rolling_acc = correct.rolling(window=window, min_periods=5).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=rolling_acc, mode="lines",
        line=dict(color=CYAN, width=2),
        fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.1)",
        name=f"Rolling {window}-day Accuracy",
    ))

    fig.add_hline(y=50, line_dash="dash", line_color=YELLOW, opacity=0.5,
                  annotation_text="50% baseline")

    return _apply_defaults(fig, height=300, title=f"Model Accuracy (Rolling {window}-day)")


# ── 14. Returns Distribution ────────────────────────────────────────

def returns_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Distribution of daily returns with normal curve overlay."""
    returns = df["return_1d"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=50, marker_color=CYAN, opacity=0.7,
        name="Daily Returns",
    ))

    # Normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    from scipy.stats import norm
    mu, std = returns.mean(), returns.std()
    y_norm = norm.pdf(x_range, mu, std) * len(returns) * (returns.max() - returns.min()) / 50
    fig.add_trace(go.Scatter(
        x=x_range, y=y_norm, mode="lines",
        line=dict(color=YELLOW, width=2, dash="dash"),
        name="Normal Dist.",
    ))

    return _apply_defaults(fig, height=300, title="Daily Returns Distribution")
