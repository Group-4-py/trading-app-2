"""
Trading Strategy Module (Section 1.3 – Optional / Bonus)

Implements two trading strategies and a backtesting engine:
  1. Buy-and-Hold (ML): Buys on UP signal, sells on price exits or signal flip.
     Conservative — uses stop-loss, trailing stop, and profit target.
     Fewer trades; stays in market during bullish runs.

  2. Buy-and-Sell (ML): Active strategy using probability signals.
     Sells when model turns bearish OR max hold days reached.
     More trades; forces rotation to act on every model signal.

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
    transaction_cost: float = TRANSACTION_COST,
    n_buy_ins: int = 12,
) -> pd.DataFrame:
    """
    Buy-and-Hold Strategy (simple DCA):
      - Look for all UP prediction days across the full backtest
      - Spread capital across up to n_buy_ins equal buy-ins
      - Execute those buy-ins on evenly spaced bullish entry points
      - Never sell — accumulates shares over time

    This keeps the strategy passive while avoiding only a handful of early
    buys. The result is a simple dollar-cost-averaging style accumulation
    plan with more than 4 buy-ins by default.

    Args:
        predictions: Binary model predictions (0=DOWN, 1=UP).
        prices: Stock prices per step.
        initial_capital: Starting capital.
        transaction_cost: Per-trade cost as fraction of value.
        n_buy_ins: Target number of equal buy-ins to spread capital across.

    Returns:
        DataFrame with portfolio simulation results.
    """
    n = len(predictions)
    cash = initial_capital
    shares = 0
    records = []

    up_steps = np.flatnonzero(np.asarray(predictions).astype(int) == 1)
    if len(up_steps) > 0:
        planned_buy_count = min(max(n_buy_ins, 1), len(up_steps))
        dca_positions = np.linspace(0, len(up_steps) - 1, num=planned_buy_count, dtype=int)
        buy_steps = set(up_steps[np.unique(dca_positions)].tolist())
    else:
        planned_buy_count = 0
        buy_steps = set()

    tranche_size = initial_capital / planned_buy_count if planned_buy_count > 0 else 0.0
    buys_executed = 0

    for i in range(n):
        price = prices[i]
        pred = int(predictions[i])
        action = "HOLD"

        if i in buy_steps and cash >= price:
            # Use the remaining cash on the final DCA entry to avoid leaving
            # a small uninvested balance stranded by transaction costs.
            deploy = cash if buys_executed == planned_buy_count - 1 else min(tranche_size, cash)
            affordable = int(deploy / (price * (1 + transaction_cost)))
            if affordable > 0:
                cost = affordable * price * (1 + transaction_cost)
                cash -= cost
                shares += affordable
                buys_executed += 1
                action = "BUY"

        portfolio_value = cash + shares * price
        records.append({
            "step": i,
            "price": price,
            "prediction": pred,
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
    probabilities: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.50,
    transaction_cost: float = TRANSACTION_COST,
    max_hold_days: int = 3,
) -> pd.DataFrame:
    """
    Active Buy-and-Sell Strategy using model probabilities:
      - BUY when P(UP) >= buy_threshold and no current position
      - SELL when P(UP) < sell_threshold (model turns bearish)
      - SELL after max_hold_days regardless (forces rotation)
      - SELL if unrealized loss >= 2% (hard stop-loss)
      - After SELL, waits one bar before re-entering (prevents same-day flip)

    This generates significantly more trades than Buy-and-Hold because it
    responds to every bearish probability signal and forces regular rotation.

    Args:
        probabilities: P(UP) probability per step from model.predict_proba().
        prices: Stock prices per step.
        initial_capital: Starting capital.
        buy_threshold: Min P(UP) to buy (default 0.50).
        sell_threshold: Max P(UP) to hold; sell if below (default 0.50).
        transaction_cost: Per-trade cost as fraction of value.
        max_hold_days: Force sell after this many days in position.

    Returns:
        DataFrame with portfolio simulation results.
    """
    n = len(probabilities)
    cash = initial_capital
    shares = 0
    hold_days = 0
    buy_price = 0.0
    records = []

    for i in range(n):
        price = prices[i]
        prob_up = float(probabilities[i])
        action = "HOLD"

        if shares > 0:
            hold_days += 1
            unrealized = (price - buy_price) / buy_price if buy_price > 0 else 0

            sell = False
            if prob_up < sell_threshold:
                sell = True
            elif hold_days >= max_hold_days:
                sell = True
            elif unrealized <= -0.02:
                sell = True

            if sell:
                revenue = shares * price * (1 - transaction_cost)
                cash += revenue
                shares = 0
                hold_days = 0
                buy_price = 0.0
                action = "SELL"

        # elif: prevents buying on the same bar we just sold
        elif prob_up >= buy_threshold and cash > price:
            affordable = int(cash / (price * (1 + transaction_cost)))
            if affordable > 0:
                cost = affordable * price * (1 + transaction_cost)
                cash -= cost
                shares += affordable
                buy_price = price
                hold_days = 0
                action = "BUY"

        portfolio_value = cash + shares * price
        records.append({
            "step": i,
            "price": price,
            "prediction": int(prob_up >= 0.5),
            "prob_up": round(prob_up, 4),
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

    daily_rets = backtest_df["daily_return"]
    n_days = len(daily_rets)

    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    cummax = backtest_df["portfolio_value"].cummax()
    drawdown = (backtest_df["portfolio_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    if "action" not in backtest_df.columns:
        n_trades = 0
        win_rate = 0.0
    else:
        buy_trades = backtest_df[backtest_df["action"] == "BUY"].copy()
        sell_trades = backtest_df[backtest_df["action"] == "SELL"].copy()
        n_trades = len(buy_trades) + len(sell_trades)

        wins = 0
        for _, sell_row in sell_trades.iterrows():
            prev_buys = buy_trades[buy_trades["step"] < sell_row["step"]]
            if not prev_buys.empty:
                entry_price = prev_buys.iloc[-1]["price"]
                if sell_row["price"] > entry_price:
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
