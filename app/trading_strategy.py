"""
Trading Strategy Module (Section 1.3 – Optional / Bonus)

Implements two trading strategies and a backtesting engine:
  1. Buy-and-Hold: Buy on UP signal, hold on DOWN, sell at profit target
  2. Buy-and-Sell: Buy on UP signal, sell on DOWN signal

The backtester simulates a portfolio over historical data using model predictions.
"""

import pandas as pd
import numpy as np
import logging

from utils.config import INITIAL_CAPITAL, TRANSACTION_COST

logger = logging.getLogger(__name__)


def strategy_buy_and_hold(
    predictions: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    profit_target: float = 0.10,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Buy-and-Hold Strategy:
      - If prediction = UP (1) and no position → BUY
      - If prediction = DOWN (0) → HOLD current position
      - If unrealized profit >= profit_target → SELL

    Args:
        predictions: Array of model predictions (0 or 1).
        prices: Array of stock prices corresponding to predictions.
        initial_capital: Starting capital in dollars.
        profit_target: Fraction of profit at which to sell (e.g., 0.10 = 10%).
        transaction_cost: Transaction cost as fraction of trade value.

    Returns:
        DataFrame with columns: date_idx, price, prediction, action, shares,
        cash, portfolio_value, daily_return.
    """
    n = len(predictions)
    cash = initial_capital
    shares = 0
    buy_price = 0.0
    records = []

    for i in range(n):
        price = prices[i]
        pred = predictions[i]
        action = "HOLD"

        # Check profit target for selling
        if shares > 0 and buy_price > 0:
            unrealized = (price - buy_price) / buy_price
            if unrealized >= profit_target:
                # SELL all shares
                revenue = shares * price * (1 - transaction_cost)
                cash += revenue
                action = "SELL"
                shares = 0
                buy_price = 0.0

        # Buy signal
        if pred == 1 and shares == 0 and cash > price:
            # Buy as many shares as possible
            affordable = int(cash / (price * (1 + transaction_cost)))
            if affordable > 0:
                cost = affordable * price * (1 + transaction_cost)
                cash -= cost
                shares += affordable
                buy_price = price
                action = "BUY"

        portfolio_value = cash + shares * price
        records.append({
            "step": i,
            "price": price,
            "prediction": int(pred),
            "action": action,
            "shares": shares,
            "cash": round(cash, 2),
            "portfolio_value": round(portfolio_value, 2),
        })

    df = pd.DataFrame(records)
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df


def strategy_buy_and_sell(
    predictions: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Buy-and-Sell Strategy:
      - If prediction = UP (1) and no position → BUY
      - If prediction = DOWN (0) and holding → SELL

    More active trading approach.

    Returns:
        DataFrame with portfolio simulation results.
    """
    n = len(predictions)
    cash = initial_capital
    shares = 0
    records = []

    for i in range(n):
        price = prices[i]
        pred = predictions[i]
        action = "HOLD"

        if pred == 1 and shares == 0 and cash > price:
            # BUY
            affordable = int(cash / (price * (1 + transaction_cost)))
            if affordable > 0:
                cost = affordable * price * (1 + transaction_cost)
                cash -= cost
                shares += affordable
                action = "BUY"

        elif pred == 0 and shares > 0:
            # SELL all
            revenue = shares * price * (1 - transaction_cost)
            cash += revenue
            shares = 0
            action = "SELL"

        portfolio_value = cash + shares * price
        records.append({
            "step": i,
            "price": price,
            "prediction": int(pred),
            "action": action,
            "shares": shares,
            "cash": round(cash, 2),
            "portfolio_value": round(portfolio_value, 2),
        })

    df = pd.DataFrame(records)
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df


def compute_strategy_metrics(backtest_df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """
    Compute performance metrics for a backtest result.

    Returns dict with: total_return, annualized_return, sharpe_ratio,
    max_drawdown, win_rate, total_trades, final_value.
    """
    final_value = backtest_df["portfolio_value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Daily returns
    daily_rets = backtest_df["daily_return"]
    n_days = len(daily_rets)

    # Annualized return (approx 252 trading days)
    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # Sharpe ratio (annualized, assuming rf=0)
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cummax = backtest_df["portfolio_value"].cummax()
    drawdown = (backtest_df["portfolio_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    # Trade stats (benchmark has no "action" column)
    if "action" not in backtest_df.columns:
        n_trades = 0
        win_rate = 0.0
    else:
        trades = backtest_df[backtest_df["action"].isin(["BUY", "SELL"])]
        n_trades = len(trades)

        sell_trades = backtest_df[backtest_df["action"] == "SELL"]
        buy_trades = backtest_df[backtest_df["action"] == "BUY"]

        wins = 0
        for _, sell_row in sell_trades.iterrows():
            prev_buys = buy_trades[buy_trades["step"] < sell_row["step"]]
            if not prev_buys.empty:
                buy_price = prev_buys.iloc[-1]["price"]
                if sell_row["price"] > buy_price:
                    wins += 1

        win_rate = wins / max(len(sell_trades), 1)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": n_trades,
        "final_value": final_value,
        "initial_capital": initial_capital,
        "n_days": n_days,
    }


def benchmark_buy_and_hold(
    prices: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """
    Simple benchmark: buy at the start and hold the entire period.
    Used for comparison against the ML-based strategies.
    """
    shares = int(initial_capital / prices[0])
    leftover_cash = initial_capital - shares * prices[0]
    records = []
    for i, price in enumerate(prices):
        pv = leftover_cash + shares * price
        records.append({
            "step": i,
            "price": price,
            "portfolio_value": round(pv, 2),
        })
    df = pd.DataFrame(records)
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df
