"""
📊 Backtesting – Strategy Simulation & Performance Analysis
Interactive backtesting simulator comparing ML-based strategies vs benchmarks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os


APP_DIR = os.path.dirname(os.path.abspath(__file__))  # or os.path.dirname(os.path.dirname(...)) for pages
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
st.set_page_config(page_title="Backtesting | AutoTrader", page_icon="📊", layout="wide")

from utils.style import inject_custom_css
from utils.config import TICKERS, TICKER_LIST, MODEL_FEATURES, INITIAL_CAPITAL
from utils.data_helpers import load_processed_data
from utils.model import load_model
from utils.trading_strategy import (
    strategy_buy_and_hold, strategy_buy_and_sell,
    compute_strategy_metrics, benchmark_buy_and_hold,
)
from utils.charts import (
    portfolio_chart, drawdown_chart, trade_actions_chart,
)

inject_custom_css()

# ── Sidebar Controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Backtesting Controls")
    st.markdown("---")

    selected_ticker = st.selectbox(
        "Select Company",
        TICKER_LIST,
        format_func=lambda t: f"{t} — {TICKERS[t]['name']}",
        key="bt_ticker",
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
    st.markdown("#### Strategy Parameters")

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1_000, max_value=1_000_000, value=int(INITIAL_CAPITAL),
        step=10_000,
    )

    transaction_cost_bps = st.slider(
        "Transaction Cost (bps)",
        min_value=0, max_value=100, value=10, step=5,
        format="%d bps",
        help="Cost per transaction in basis points (10 bps = 0.1%).",
    )
    transaction_cost = transaction_cost_bps / 10_000.0

    backtest_days = st.slider(
        "Backtest Period (trading days)",
        min_value=60, max_value=504, value=252, step=20,
    )

    st.markdown("---")
    st.markdown("#### Select Strategies")

    run_bah = st.checkbox("Buy & Hold", value=True)
    run_bas = st.checkbox("Buy & Sell", value=True)
    run_benchmark = st.checkbox("Benchmark", value=True)

    st.markdown("---")
    run_backtest = st.button("🚀 Run Backtest", use_container_width=True)

    api_key = st.session_state.get("api_key_stored", "")

# ── Page Header ──────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem;">Backtesting Simulator</h1>
        <p style="color: #94a3b8; font-size: 1rem;">
            Test and compare trading strategies against historical data
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Strategy Explanation ─────────────────────────────────────────────
st.markdown("### Strategy Definitions")
st.markdown("")

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        """
                <div class="glass-card" style="min-height: 220px;">
                    <h4 style="font-family: 'Orbitron', sans-serif; color: #00d4ff !important; font-size: 0.85rem;">
                        📈 Buy & Hold
                    </h4>
                    <p style="color: #94a3b8 !important; font-size: 0.85rem; line-height: 1.7;">
                        Buy on a <span style="color: #00ff88;">▲ UP prediction</span>.<br>
                        Wait 1 trading week before the next possible buy.<br>
                        Never sells during the backtest.<br><br>
                        <em>Long-only accumulation; 5-day cooldown.</em>
                    </p>
                </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
        <div class="glass-card" style="min-height: 220px;">
            <h4 style="font-family: 'Orbitron', sans-serif; color: #a855f7 !important; font-size: 0.85rem;">
                🔄 Buy & Sell (ML)
            </h4>
            <p style="color: #94a3b8 !important; font-size: 0.85rem; line-height: 1.7;">
                <span style="color: #00ff88;">▲ P(UP) ≥ 50%</span> → Buy shares<br>
                <span style="color: #ff3366;">▼ P(UP) &lt; 50%</span> → Sell shares<br>
                <span style="color: #fbbf24;">⏱ Force sell after 3 days</span> → Rotate<br><br>
                <em>More trades; sells on every bearish signal.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    st.markdown(
        """
                <div class="glass-card" style="min-height: 220px;">
                    <h4 style="font-family: 'Orbitron', sans-serif; color: #94a3b8 !important; font-size: 0.85rem;">
                        📊 Benchmark
                    </h4>
                    <p style="color: #94a3b8 !important; font-size: 0.85rem; line-height: 1.7;">
                        Buy once on day 1 and hold the entire period.<br>
                        No predictions or active trading.<br><br>
                        <em>Baseline to evaluate if the ML model adds value over simply holding.</em>
                    </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Run Backtest ─────────────────────────────────────────────────────
# Fingerprint current params — re-run whenever any parameter changes
_bt_params = (selected_ticker, backtest_days, initial_capital,
              transaction_cost, run_bah, run_bas, run_benchmark)
_needs_rerun = (
    run_backtest
    or "bt_results" not in st.session_state
    or st.session_state.get("bt_params") != _bt_params
)

if _needs_rerun:
    with st.spinner("Running backtest simulation..."):
        # Load data & model
        processed_df = load_processed_data(selected_ticker, days=backtest_days, api_key=api_key)
        model = load_model(selected_ticker)

        if processed_df.empty or len(processed_df) < 30:
            st.error("Not enough data to run backtest. Try increasing the backtest period.")
            st.stop()

        # Generate predictions and probabilities
        avail_features = [f for f in MODEL_FEATURES if f in processed_df.columns]
        X = processed_df[avail_features]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # P(UP)
        prices = processed_df["price"].values
        dates = processed_df["date"]

        results = {}
        metrics_dict = {}

        if run_bah:
            bt_bah = strategy_buy_and_hold(
                predictions, prices, initial_capital,
                transaction_cost=transaction_cost,
            )
            results["Buy & Hold"] = bt_bah
            metrics_dict["Buy & Hold"] = compute_strategy_metrics(bt_bah, initial_capital)

        if run_bas:
            bt_bas = strategy_buy_and_sell(
                probabilities, prices, initial_capital,
                transaction_cost=transaction_cost,
            )
            results["Buy & Sell"] = bt_bas
            metrics_dict["Buy & Sell"] = compute_strategy_metrics(bt_bas, initial_capital)

        bench = None
        if run_benchmark:
            bench = benchmark_buy_and_hold(prices, initial_capital)
            metrics_dict["Benchmark"] = compute_strategy_metrics(bench, initial_capital)

        # Store in session state
        st.session_state["bt_results"] = results
        st.session_state["bt_metrics"] = metrics_dict
        st.session_state["bt_benchmark"] = bench
        st.session_state["bt_dates"] = dates
        st.session_state["bt_predictions"] = predictions
        st.session_state["bt_prices"] = prices
        st.session_state["bt_params"] = _bt_params

# ── Load cached results ──────────────────────────────────────────────
if "bt_results" in st.session_state:
    results = st.session_state["bt_results"]
    metrics_dict = st.session_state["bt_metrics"]
    bench = st.session_state["bt_benchmark"]
    dates = st.session_state["bt_dates"]

    # ── Portfolio Performance Chart ──────────────────────────────────
    st.markdown("### Portfolio Performance")
    st.markdown("")

    fig = portfolio_chart(results, bench, dates)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Strategy Comparison Metrics ──────────────────────────────────
    st.markdown("### Strategy Comparison")
    st.markdown("")

    # Build comparison DataFrame
    comp_data = []
    for name, m in metrics_dict.items():
        comp_data.append({
            "Strategy": name,
            "Total Return": f"{m['total_return']:+.2%}",
            "Annualized Return": f"{m['annualized_return']:+.2%}",
            "Sharpe Ratio": f"{m['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{m['max_drawdown']:.2%}",
            "Win Rate": f"{m['win_rate']:.1%}" if m['total_trades'] > 0 else "N/A",
            "Total Trades": m["total_trades"],
            "Final Value": f"${m['final_value']:,.2f}",
        })

    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("")

    # Summary metrics row
    best_strategy = max(metrics_dict.items(), key=lambda x: x[1]["total_return"])
    best_name, best_m = best_strategy

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("🏆 Best Strategy", best_name)
    bm2.metric("Total Return", f"{best_m['total_return']:+.2%}")
    bm3.metric("Sharpe Ratio", f"{best_m['sharpe_ratio']:.2f}")
    bm4.metric("Final Value", f"${best_m['final_value']:,.2f}")

    st.markdown("---")

    # ── Detailed Strategy Analysis ───────────────────────────────────
    st.markdown("### Detailed Strategy Analysis")

    # Strategy selector for detailed view
    strategy_names = list(results.keys())
    if strategy_names:
        selected_strategy = st.selectbox(
            "Select strategy for detailed view",
            strategy_names,
            key="detail_strategy",
        )

        bt_df = results[selected_strategy]

        # Metrics for selected strategy
        sm = metrics_dict[selected_strategy]

        dm1, dm2, dm3, dm4, dm5, dm6 = st.columns(6)
        dm1.metric("Total Return", f"{sm['total_return']:+.2%}")
        dm2.metric("Ann. Return", f"{sm['annualized_return']:+.2%}")
        dm3.metric("Sharpe", f"{sm['sharpe_ratio']:.2f}")
        dm4.metric("Max Drawdown", f"{sm['max_drawdown']:.2%}")
        dm5.metric("Win Rate", f"{sm['win_rate']:.1%}" if sm['total_trades'] > 0 else "N/A")
        dm6.metric("Trades", sm["total_trades"])

        # Charts
        tab_trades, tab_drawdown, tab_actions = st.tabs([
            "📈 Equity Curve",
            "📉 Drawdown Analysis",
            "🎯 Trade Actions",
        ])

        with tab_trades:
            import plotly.graph_objects as pgo
            fig = pgo.Figure()
            fig.add_trace(pgo.Scatter(
                x=dates[:len(bt_df)], y=bt_df["portfolio_value"],
                mode="lines", fill="tozeroy",
                line=dict(color="#00d4ff", width=2),
                fillcolor="rgba(0, 212, 255, 0.1)",
                name="Portfolio Value",
            ))
            fig.add_hline(y=initial_capital, line_dash="dash",
                         line_color="#fbbf24", opacity=0.5,
                         annotation_text="Initial Capital")
            fig.update_layout(
                height=400,
                font=dict(family="Rajdhani, sans-serif", color="#94a3b8"),
                paper_bgcolor="rgba(10, 14, 23, 0)",
                plot_bgcolor="rgba(10, 14, 23, 0)",
                xaxis=dict(gridcolor="rgba(30, 41, 59, 0.5)"),
                yaxis=dict(gridcolor="rgba(30, 41, 59, 0.5)", title="Portfolio Value ($)"),
                title=f"Equity Curve — {selected_strategy}",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_drawdown:
            st.plotly_chart(
                drawdown_chart(bt_df, dates, selected_strategy),
                use_container_width=True,
            )

        with tab_actions:
            st.plotly_chart(
                trade_actions_chart(bt_df, dates),
                use_container_width=True,
            )

        st.markdown("---")

        # ── Trade Log ────────────────────────────────────────────────
        st.markdown("### Trade Log")

        trades_only = bt_df[bt_df["action"].isin(["BUY", "SELL"])].copy()
        if not trades_only.empty:
            trades_only["date"] = dates.iloc[trades_only.index].values
            trades_display = trades_only[["date", "action", "price", "shares", "cash", "portfolio_value"]]
            trades_display = trades_display.rename(columns={
                "date": "Date", "action": "Action", "price": "Price",
                "shares": "Shares Held", "cash": "Cash",
                "portfolio_value": "Portfolio Value",
            })
            st.dataframe(trades_display, use_container_width=True, hide_index=True)
        else:
            st.info("No trades were executed with this strategy.")

    st.markdown("---")

    # ── Monthly Returns Heatmap ──────────────────────────────────────
    st.markdown("### Monthly Returns Analysis")

    if strategy_names:
        bt_monthly = results[strategy_names[0]].copy()
        bt_monthly["date"] = dates[:len(bt_monthly)].values

        bt_monthly["month"] = pd.to_datetime(bt_monthly["date"]).dt.to_period("M")
        monthly_returns = bt_monthly.groupby("month").apply(
            lambda g: (g["portfolio_value"].iloc[-1] / g["portfolio_value"].iloc[0] - 1) * 100
            if len(g) > 0 else 0
        )

        if len(monthly_returns) > 1:
            import plotly.graph_objects as pgo

            months = [str(m) for m in monthly_returns.index]
            values = monthly_returns.values
            colors = ["#00ff88" if v >= 0 else "#ff3366" for v in values]

            fig = pgo.Figure(pgo.Bar(
                x=months, y=values, marker_color=colors, opacity=0.8,
                text=[f"{v:+.1f}%" for v in values], textposition="outside",
                textfont=dict(size=10),
            ))
            fig.update_layout(
                height=350,
                font=dict(family="Rajdhani, sans-serif", color="#94a3b8"),
                paper_bgcolor="rgba(10, 14, 23, 0)",
                plot_bgcolor="rgba(10, 14, 23, 0)",
                xaxis=dict(gridcolor="rgba(30, 41, 59, 0.5)", title="Month"),
                yaxis=dict(gridcolor="rgba(30, 41, 59, 0.5)", title="Return (%)"),
                title=f"Monthly Returns — {strategy_names[0]}",
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click **Run Backtest** in the sidebar to start the simulation.")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("")
st.caption(
    "⚠️ Backtesting results are hypothetical and based on historical data. "
    "Past performance does not guarantee future results. Transaction costs "
    "and slippage are approximated."
)
