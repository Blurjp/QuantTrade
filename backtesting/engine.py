"""
Backtest execution helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(value: float) -> float:
    return float(value) if pd.notna(value) and np.isfinite(value) else 0.0


def _manual_portfolio(price_df: pd.DataFrame, signal_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = price_df.merge(signal_df, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["position"] = merged["position"].fillna(0).astype(int)
    merged["position"] = merged["position"].shift(1).fillna(0).astype(int)
    merged["asset_return"] = merged["price_close"].pct_change().fillna(0.0)
    merged["returns"] = merged["position"] * merged["asset_return"]
    merged["equity_curve"] = (1.0 + merged["returns"]).cumprod()
    merged["drawdown"] = merged["equity_curve"] / merged["equity_curve"].cummax() - 1.0

    returns = merged["returns"]
    trade_days = merged["position"].diff().abs().fillna(0)
    turnover = float(trade_days.sum())
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else math.inf
    std = returns.std()
    sharpe = float(np.sqrt(252) * returns.mean() / std) if pd.notna(std) and std > 0 else 0.0
    periods = max(len(merged), 1)
    cagr = float(merged["equity_curve"].iloc[-1] ** (252 / periods) - 1) if periods > 1 else 0.0
    summary = {
        "total_return": _finite(merged["equity_curve"].iloc[-1] - 1.0),
        "max_drawdown": _finite(merged["drawdown"].min()),
        "sharpe": _finite(sharpe),
        "win_rate": _finite((returns > 0).mean()) if len(returns) > 0 else 0.0,
        "profit_factor": _finite(profit_factor),
        "cagr": _finite(cagr),
        "turnover": _finite(turnover),
        "trade_count": int((trade_days > 0).sum()),
    }
    return merged, summary


def run_backtest(price_df: pd.DataFrame, signal_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged, summary = _manual_portfolio(price_df, signal_df)

    try:
        import vectorbt as vbt
    except ModuleNotFoundError:
        return merged, summary

    close = merged.set_index("date")["price_close"]
    entries = merged.set_index("date")["position"] > 0
    exits = merged.set_index("date")["position"] <= 0
    short_entries = merged.set_index("date")["position"] < 0
    short_exits = merged.set_index("date")["position"] >= 0
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=1.0,
        fees=0.0,
        freq="1D",
    )
    try:
        summary.update({
            "total_return": _finite(portfolio.total_return()),
            "max_drawdown": _finite(portfolio.max_drawdown()),
            "sharpe": _finite(portfolio.sharpe_ratio()) if pd.notna(portfolio.sharpe_ratio()) else summary["sharpe"],
            "trade_count": int(portfolio.trades.count()),
        })
        merged["equity_curve"] = portfolio.value().values
        merged["drawdown"] = portfolio.drawdown().values
    except Exception:
        return merged, summary
    return merged, summary
