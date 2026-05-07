"""
Portfolio rules - position sizing and risk limits.

Enforces portfolio-level constraints:
- Maximum position size per asset
- Maximum exposure per strategy
- Maximum total exposure
- Correlation-based position limits
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np

from strategies.base import TradeCandidate, Direction, AssetType


@dataclass
class PositionLimit:
    """
    Position limit for a specific asset or group.
    """
    pattern: str  # Ticker pattern or "ALL"
    max_size_pct: float  # Maximum position size (% of portfolio)
    max_total_exposure: float = 0.20  # Maximum total exposure to this group

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "max_size_pct": self.max_size_pct,
            "max_total_exposure": self.max_total_exposure,
        }


@dataclass
class PortfolioConstraints:
    """
    Portfolio-wide constraints.

    Enforced on all trade candidates.
    """
    # Position constraints
    max_single_position_pct: float = 0.05  # Max 5% in one position
    max_total_exposure: float = 0.95  # Max 95% invested (keep 5% cash)

    # Strategy constraints
    max_per_strategy_pct: float = 0.15  # Max 15% per strategy
    max_same_direction_pct: float = 0.25  # Max 25% in same direction

    # Asset class constraints
    max_equity_pct: float = 0.60  # Max 60% in equities
    max_commodity_pct: float = 0.30  # Max 30% in commodities
    max_etf_pct: float = 0.50  # Max 50% in ETFs

    # Correlation constraints (simplified)
    allow_same_ticker_opposite: bool = True  # Allow long and short same ticker
    require_diversification: bool = True  # Prefer diversified positions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_single_position_pct": self.max_single_position_pct,
            "max_total_exposure": self.max_total_exposure,
            "max_per_strategy_pct": self.max_per_strategy_pct,
            "max_same_direction_pct": self.max_same_direction_pct,
            "max_equity_pct": self.max_equity_pct,
            "max_commodity_pct": self.max_commodity_pct,
            "max_etf_pct": self.max_etf_pct,
            "allow_same_ticker_opposite": self.allow_same_ticker_opposite,
            "require_diversification": self.require_diversification,
        }


class PortfolioRules:
    """
    Enforces portfolio-level rules on trade candidates.

    Filters and adjusts trades to ensure portfolio constraints are met.

    Example:
        >>> rules = PortfolioRules()
        >>> trades = [TradeCandidate(...), ...]
        >>> filtered = rules.apply_rules(trades, current_positions={})
        >>> for trade in filtered:
        ...     print(f"{trade.ticker}: {trade.size_pct*100:.1f}%")
    """

    def __init__(
        self,
        constraints: Optional[PortfolioConstraints] = None,
        position_limits: Optional[List[PositionLimit]] = None,
    ):
        """
        Initialize portfolio rules.

        Args:
            constraints: Portfolio-wide constraints
            position_limits: Specific asset position limits
        """
        self.constraints = constraints or PortfolioConstraints()
        self.position_limits = position_limits or []
        self._limit_index = {limit.pattern: limit for limit in self.position_limits}

    def apply_rules(
        self,
        trades: List[TradeCandidate],
        current_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[TradeCandidate]:
        """
        Apply portfolio rules to trade candidates.

        Filters and adjusts trades to ensure constraints are met.

        Args:
            trades: List of proposed trade candidates
            current_positions: Current positions {ticker: {direction, size_pct}}

        Returns:
            Filtered and adjusted list of trade candidates
        """
        current_positions = current_positions or {}

        # Step 1: Apply single position limits
        trades = self._apply_single_position_limits(trades)

        # Step 2: Apply strategy limits
        trades = self._apply_strategy_limits(trades)

        # Step 3: Apply direction limits
        trades = self._apply_direction_limits(trades)

        # Step 4: Apply asset class limits
        trades = self._apply_asset_class_limits(trades, current_positions)

        # Step 5: Check for duplicates and conflicting positions
        trades = self._resolve_conflicts(trades, current_positions)

        # Step 6: Apply total exposure limit
        trades = self._apply_total_exposure_limit(trades, current_positions)

        return trades

    def get_current_exposure(
        self,
        trades: List[TradeCandidate],
        current_positions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Calculate current portfolio exposure by category.

        Returns:
            Dict with exposure by strategy, direction, asset_class
        """
        exposure = {
            "by_strategy": defaultdict(float),
            "by_direction": defaultdict(float),
            "by_asset_class": defaultdict(float),
            "total_long": 0.0,
            "total_short": 0.0,
            "total_net": 0.0,
        }

        # Add current positions
        for ticker, pos in current_positions.items():
            direction = pos.get("direction", "long")
            size = pos.get("size_pct", 0)
            strategy = pos.get("strategy", "unknown")
            asset_type = pos.get("asset_type", "equity")

            exposure["by_strategy"][strategy] += size
            exposure["by_direction"][direction] += size
            exposure["by_asset_class"][asset_type] += size

            if direction == "long":
                exposure["total_long"] += size
            else:
                exposure["total_short"] += size

        # Add new trades
        for trade in trades:
            exposure["by_strategy"][trade.strategy] += trade.size_pct
            exposure["by_direction"][trade.direction.value] += trade.size_pct
            exposure["by_asset_class"][trade.asset_type.value] += trade.size_pct

            if trade.direction == Direction.LONG:
                exposure["total_long"] += trade.size_pct
            else:
                exposure["total_short"] += trade.size_pct

        exposure["total_net"] = exposure["total_long"] - exposure["total_short"]

        return exposure

    def _apply_single_position_limits(
        self,
        trades: List[TradeCandidate],
    ) -> List[TradeCandidate]:
        """Apply maximum single position size limits."""
        filtered = []

        for trade in trades:
            # Check specific limits
            limit = self._limit_index.get(trade.ticker)
            if limit:
                max_size = min(limit.max_size_pct, self.constraints.max_single_position_pct)
            else:
                max_size = self.constraints.max_single_position_pct

            if trade.size_pct > max_size:
                # Cap the size
                trade = TradeCandidate(
                    **{k: v for k, v in trade.to_dict().items() if k != "size_pct"},
                    size_pct=max_size,
                )

            filtered.append(trade)

        return filtered

    def _apply_strategy_limits(
        self,
        trades: List[TradeCandidate],
    ) -> List[TradeCandidate]:
        """Apply per-strategy position limits."""
        # Group by strategy
        by_strategy = defaultdict(list)
        for trade in trades:
            by_strategy[trade.strategy].append(trade)

        filtered = []

        for strategy, strategy_trades in by_strategy.items():
            total_size = sum(t.size_pct for t in strategy_trades)

            if total_size > self.constraints.max_per_strategy_pct:
                # Scale down proportionally
                scale = self.constraints.max_per_strategy_pct / total_size

                for trade in strategy_trades:
                    scaled_trade = TradeCandidate(
                        **{k: v for k, v in trade.to_dict().items() if k != "size_pct"},
                        size_pct=trade.size_pct * scale,
                    )
                    filtered.append(scaled_trade)
            else:
                filtered.extend(strategy_trades)

        return filtered

    def _apply_direction_limits(
        self,
        trades: List[TradeCandidate],
    ) -> List[TradeCandidate]:
        """Apply same-direction position limits."""
        # Group by direction
        long_trades = [t for t in trades if t.direction == Direction.LONG]
        short_trades = [t for t in trades if t.direction == Direction.SHORT]

        filtered = []

        # Scale long trades
        long_total = sum(t.size_pct for t in long_trades)
        if long_total > self.constraints.max_same_direction_pct:
            scale = self.constraints.max_same_direction_pct / long_total
            for trade in long_trades:
                filtered.append(TradeCandidate(
                    **{k: v for k, v in trade.to_dict().items() if k != "size_pct"},
                    size_pct=trade.size_pct * scale,
                ))
        else:
            filtered.extend(long_trades)

        # Scale short trades
        short_total = sum(t.size_pct for t in short_trades)
        if short_total > self.constraints.max_same_direction_pct:
            scale = self.constraints.max_same_direction_pct / short_total
            for trade in short_trades:
                filtered.append(TradeCandidate(
                    **{k: v for k, v in trade.to_dict().items() if k != "size_pct"},
                    size_pct=trade.size_pct * scale,
                ))
        else:
            filtered.extend(short_trades)

        return filtered

    def _apply_asset_class_limits(
        self,
        trades: List[TradeCandidate],
        current_positions: Dict[str, Dict[str, Any]],
    ) -> List[TradeCandidate]:
        """Apply asset class exposure limits."""
        exposure = self.get_current_exposure(trades, current_positions)

        filtered = []

        for trade in trades:
            asset_class = trade.asset_type.value
            current_exp = exposure["by_asset_class"][asset_class]

            # Get limit for this asset class
            if asset_class == "equity":
                limit = self.constraints.max_equity_pct
            elif asset_class == "commodity_futures":
                limit = self.constraints.max_commodity_pct
            elif asset_class == "etf":
                limit = self.constraints.max_etf_pct
            else:
                limit = 0.20  # Default 20%

            if current_exp < limit:
                filtered.append(trade)
            # Else: skip this trade (asset class limit reached)

        return filtered

    def _resolve_conflicts(
        self,
        trades: List[TradeCandidate],
        current_positions: Dict[str, Dict[str, Any]],
    ) -> List[TradeCandidate]:
        """Resolve duplicate and conflicting positions."""
        # Group by ticker
        by_ticker: Dict[str, List[TradeCandidate]] = defaultdict(list)
        for trade in trades:
            by_ticker[trade.ticker].append(trade)

        filtered = []

        for ticker, ticker_trades in by_ticker.items():
            current = current_positions.get(ticker)

            if current:
                current_dir = current.get("direction", "long")
                current_dir = Direction.LONG if current_dir == "long" else Direction.SHORT

                # Check if we have conflicting new trades
                directions = set(t.direction for t in ticker_trades)

                if len(directions) > 1:
                    # Mixed signals - cancel out
                    continue

                new_dir = list(directions)[0]

                if new_dir == current_dir:
                    # Same direction - add to position
                    filtered.extend(ticker_trades)
                elif self.constraints.allow_same_ticker_opposite:
                    # Opposite direction - allow as separate position
                    filtered.extend(ticker_trades)
                else:
                    # Close existing position first
                    filtered.extend(ticker_trades)
            else:
                # No existing position
                if len(set(t.direction for t in ticker_trades)) == 1:
                    # All same direction - add
                    filtered.extend(ticker_trades)
                else:
                    # Mixed directions - cancel out
                    pass

        return filtered

    def _apply_total_exposure_limit(
        self,
        trades: List[TradeCandidate],
        current_positions: Dict[str, Dict[str, Any]],
    ) -> List[TradeCandidate]:
        """Apply total portfolio exposure limit."""
        exposure = self.get_current_exposure(trades, current_positions)
        total_exposure = exposure["total_long"] + exposure["total_short"]

        if total_exposure <= self.constraints.max_total_exposure:
            return trades

        # Scale down all trades
        scale = self.constraints.max_total_exposure / total_exposure

        filtered = []
        for trade in trades:
            filtered.append(TradeCandidate(
                **{k: v for k, v in trade.to_dict().items() if k != "size_pct"},
                size_pct=trade.size_pct * scale,
            ))

        return filtered


def apply_portfolio_rules(
    trades: List[TradeCandidate],
    current_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    constraints: Optional[PortfolioConstraints] = None,
) -> List[TradeCandidate]:
    """
    Convenience function to apply portfolio rules.

    Args:
        trades: Proposed trade candidates
        current_positions: Current portfolio positions
        constraints: Portfolio constraints

    Returns:
        Filtered and adjusted trade candidates
    """
    rules = PortfolioRules(constraints=constraints)
    return rules.apply_rules(trades, current_positions)


__all__ = [
    "PortfolioRules",
    "PortfolioConstraints",
    "PositionLimit",
    "apply_portfolio_rules",
]
