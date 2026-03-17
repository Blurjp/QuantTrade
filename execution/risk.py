"""
Risk management utilities.

Position sizing, stop loss calculation, and risk metrics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategies.base import TradeCandidate, Direction


@dataclass
class RiskMetrics:
    """
    Risk metrics for a trade or portfolio.
    """
    # Position risk
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    # Risk amounts
    risk_amount: float = 0.0  # Dollar amount at risk
    risk_pct: float = 0.0  # Percentage of position at risk

    # Portfolio risk
    portfolio_var_pct: float = 0.0  # Value at Risk (%)
    max_drawdown_pct: float = 0.0  # Maximum drawdown
    beta: float = 1.0  # Beta to market

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "risk_amount": self.risk_amount,
            "risk_pct": self.risk_pct,
            "portfolio_var_pct": self.portfolio_var_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "beta": self.beta,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
        }


class PositionSizer:
    """
    Calculate optimal position sizes based on risk parameters.

    Methods:
    - Fixed fractional: Risk % of portfolio per trade
    - Kelly criterion: Optimal size based on win rate and payoff
    - Risk parity: Equalize risk contribution across positions
    - Volatility targeting: Scale position by inverse volatility
    """

    def __init__(
        self,
        method: str = "fixed_fractional",
        portfolio_value: float = 100000.0,
        risk_per_trade: float = 0.02,  # 2% risk per trade
    ):
        """
        Initialize position sizer.

        Args:
            method: Sizing method ('fixed_fractional', 'kelly', 'risk_parity', 'vol_target')
            portfolio_value: Total portfolio value
            risk_per_trade: Target risk per trade (as % of portfolio)
        """
        self.method = method
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade

    def calculate_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate position size in dollars.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win amount (for Kelly)
            avg_loss: Average loss amount (for Kelly)
            volatility: Asset volatility (for vol targeting)

        Returns:
            Position size in dollars
        """
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            return 0.0

        if self.method == "fixed_fractional":
            # Risk fixed % of portfolio
            risk_amount = self.portfolio_value * self.risk_per_trade
            shares = risk_amount / risk_per_share
            return shares * entry_price

        elif self.method == "kelly":
            # Kelly criterion: f = (bp - q) / b
            # where b = avg_win / avg_loss, p = win_rate, q = 1-p
            if win_rate is None or avg_win is None or avg_loss is None:
                # Fall back to fixed fractional
                return self._fixed_fractional(entry_price, stop_loss_price)

            if avg_loss == 0:
                return 0.0

            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_frac = (b * p - q) / b
            kelly_frac = max(0, min(kelly_frac, 0.25))  # Cap at 25%

            risk_amount = self.portfolio_value * kelly_frac
            shares = risk_amount / risk_per_share
            return shares * entry_price

        elif self.method == "vol_target":
            # Scale by inverse volatility
            if volatility is None or volatility == 0:
                return self._fixed_fractional(entry_price, stop_loss_price)

            # Target 15% annual volatility
            target_vol = 0.15
            scale = target_vol / volatility
            scale = max(0.5, min(scale, 2.0))  # Cap scaling

            base_size = self._fixed_fractional(entry_price, stop_loss_price)
            return base_size * scale

        else:
            return self._fixed_fractional(entry_price, stop_loss_price)

    def _fixed_fractional(
        self,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """Fixed fractional position sizing."""
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_amount = self.portfolio_value * self.risk_per_trade
        shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
        return shares * entry_price


def calculate_stop_loss(
    entry_price: float,
    direction: Direction,
    method: str = "percentage",
    stop_pct: float = 0.05,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0,
) -> float:
    """
    Calculate stop loss price.

    Args:
        entry_price: Entry price
        direction: Trade direction
        method: Stop method ('percentage', 'atr', 'support_resistance')
        stop_pct: Stop percentage (for percentage method)
        atr: Average True Range (for ATR method)
        atr_multiplier: ATR multiplier

    Returns:
        Stop loss price
    """
    if method == "percentage":
        if direction == Direction.LONG:
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)

    elif method == "atr" and atr is not None:
        if direction == Direction.LONG:
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)

    else:
        # Default to percentage
        if direction == Direction.LONG:
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)


def calculate_take_profit(
    entry_price: float,
    stop_loss_price: float,
    direction: Direction,
    reward_risk_ratio: float = 2.0,
) -> float:
    """
    Calculate take profit price based on risk-reward ratio.

    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        direction: Trade direction
        reward_risk_ratio: Target reward:risk ratio

    Returns:
        Take profit price
    """
    risk = abs(entry_price - stop_loss_price)
    reward = risk * reward_risk_ratio

    if direction == Direction.LONG:
        return entry_price + reward
    else:
        return entry_price - reward


def calculate_portfolio_risk(
    positions: List[Dict[str, Any]],
    prices: Dict[str, float],
    correlation_matrix: Optional[pd.DataFrame] = None,
) -> RiskMetrics:
    """
    Calculate portfolio-level risk metrics.

    Args:
        positions: List of position dicts {ticker, direction, size, entry_price}
        prices: Current prices {ticker: price}
        correlation_matrix: Asset correlation matrix

    Returns:
        RiskMetrics object with portfolio risk
    """
    if not positions:
        return RiskMetrics()

    # Calculate portfolio value
    total_value = sum(p.get("size", 0) * prices.get(p["ticker"], 0) for p in positions)

    # Calculate weighted beta (assuming beta=1 for all stocks)
    weights = np.array([
        (p.get("size", 0) * prices.get(p["ticker"], 0)) / total_value
        for p in positions
    ])

    # Simplified portfolio metrics
    metrics = RiskMetrics(
        portfolio_var_pct = 0.15,  # Assumed 15% VaR
        max_drawdown_pct = 0.10,  # Assumed 10% max DD
        beta = 1.0,
    )

    # If we have correlation matrix, calculate portfolio variance
    if correlation_matrix is not None and len(weights) > 1:
        try:
            # Portfolio variance = w' * Sigma * w
            cov_matrix = correlation_matrix  # Simplified (using corr as proxy for cov)
            portfolio_var = float(weights.T @ cov_matrix.values @ weights)
            metrics.portfolio_var_pct = np.sqrt(portfolio_var) * 0.15  # Scale to ~15%
        except Exception:
            pass

    return metrics


def check_risk_limits(
    trades: List[TradeCandidate],
    current_positions: Dict[str, Dict[str, Any]],
    max_portfolio_risk: float = 0.20,
    max_single_risk: float = 0.05,
) -> Tuple[bool, List[str]]:
    """
    Check if trades violate risk limits.

    Args:
        trades: Proposed trades
        current_positions: Current positions
        max_portfolio_risk: Max portfolio risk %
        max_single_risk: Max single position risk %

    Returns:
        Tuple of (is_safe, list_of_violations)
    """
    violations = []

    # Check single position risks
    for trade in trades:
        if trade.size_pct > max_single_risk:
            violations.append(
                f"{trade.ticker}: {trade.size_pct:.1%} exceeds max {max_single_risk:.1%}"
            )

    # Check total new exposure
    total_new_exposure = sum(t.size_pct for t in trades)
    if total_new_exposure > max_portfolio_risk:
        violations.append(
            f"Total new exposure {total_new_exposure:.1%} exceeds max {max_portfolio_risk:.1%}"
        )

    # Check for concentration
    ticker_exposure = defaultdict(float)
    for trade in trades:
        ticker_exposure[trade.ticker] += trade.size_pct

    for ticker, exposure in ticker_exposure.items():
        if exposure > max_single_risk * 2:  # Allow 2x normal size if split into multiple trades
            violations.append(
                f"{ticker}: Total exposure {exposure:.1%} exceeds limit"
            )

    return len(violations) == 0, violations


__all__ = [
    "RiskMetrics",
    "PositionSizer",
    "calculate_stop_loss",
    "calculate_take_profit",
    "calculate_portfolio_risk",
    "check_risk_limits",
]
