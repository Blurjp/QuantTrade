"""
Trade mapper - ResearchSignal to TradeCandidate conversion.

This is where "economic observation" becomes "tradable idea".

Key responsibilities:
- Map regional signals to specific tickers
- Determine trade direction and size
- Set risk parameters (stop loss, take profit)
- Aggregate signals from multiple regions
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from strategies.base import ResearchSignal, TradeCandidate, Direction, AssetType


@dataclass
class TradingRule:
    """
    Rule for converting a signal to a trade.

    Defines how a research signal maps to a tradable position.
    """
    region_pattern: str  # Region name pattern this applies to
    tickers: List[str]  # Tickers to trade
    asset_type: AssetType
    direction_map: Dict[str, Direction]  # Map signal direction to trade direction

    # Position sizing
    base_size_pct: float = 0.02  # Base position size (% of portfolio)
    max_size_pct: float = 0.05  # Maximum position size

    # Risk parameters
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    horizon_days: int = 20

    # Confidence scaling
    scale_by_confidence: bool = True
    min_confidence: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "region_pattern": self.region_pattern,
            "tickers": self.tickers,
            "asset_type": self.asset_type.value,
            "direction_map": {k: v.value for k, v in self.direction_map.items()},
            "base_size_pct": self.base_size_pct,
            "max_size_pct": self.max_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "horizon_days": self.horizon_days,
            "scale_by_confidence": self.scale_by_confidence,
            "min_confidence": self.min_confidence,
        }


class TradeMapper:
    """
    Maps research signals to trade candidates.

    Uses trading rules to convert ResearchSignal objects into
    TradeCandidate objects that can be executed.

    Example:
        >>> mapper = TradeMapper()
        >>> mapper.add_default_rules()
        >>> signals = [ResearchSignal(...), ...]
        >>> trades = mapper.map_signals(signals)
        >>> for trade in trades:
        ...     print(f"{trade.ticker}: {trade.direction.value} {trade.size_pct*100}%")
    """

    def __init__(self, rules: Optional[List[TradingRule]] = None):
        """
        Initialize the trade mapper.

        Args:
            rules: Optional list of trading rules
        """
        self.rules: List[TradingRule] = rules or []
        self._rule_index: Dict[str, List[TradingRule]] = defaultdict(list)

        # Index rules by region pattern
        for rule in self.rules:
            self._rule_index[rule.region_pattern].append(rule)

    def add_rule(self, rule: TradingRule) -> "TradeMapper":
        """
        Add a trading rule.

        Args:
            rule: Trading rule to add

        Returns:
            Self (for chaining)
        """
        self.rules.append(rule)
        self._rule_index[rule.region_pattern].append(rule)
        return self

    def add_default_rules(self) -> "TradeMapper":
        """
        Add default trading rules for common strategies.

        Returns:
            Self
        """
        # Auto inventory rules
        self.add_rule(TradingRule(
            region_pattern="*auto*",
            tickers=["CARZ", "F", "GM", "STLA"],
            asset_type=AssetType.EQUITY,
            direction_map={"long": Direction.SHORT, "short": Direction.SHORT},  # High inventory = short
            base_size_pct=0.02,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
            horizon_days=30,
        ))

        # Chokepoint rules
        self.add_rule(TradingRule(
            region_pattern="*hormuz*",
            tickers=["DRYS", "SBLK", "TNK", "NAT"],  # Shipping
            asset_type=AssetType.EQUITY,
            direction_map={"long": Direction.LONG, "short": Direction.SHORT},
            base_size_pct=0.02,
            stop_loss_pct=0.06,
            take_profit_pct=0.12,
            horizon_days=14,
        ))

        # Oil storage rules
        self.add_rule(TradingRule(
            region_pattern="*cushing*",
            tickers=["USO", "XLE", "XOM", "CVX"],
            asset_type=AssetType.ETF,
            direction_map={"long": Direction.SHORT, "short": Direction.SHORT},  # High storage = bearish
            base_size_pct=0.02,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            horizon_days=21,
        ))

        # Brazil soy rules
        self.add_rule(TradingRule(
            region_pattern="*brazil_soy*",
            tickers=["SOYB", "SOYB", "DBA", "JJG"],  # Soy ETFs
            asset_type=AssetType.ETF,
            direction_map={"long": Direction.LONG, "short": Direction.SHORT},
            base_size_pct=0.02,
            stop_loss_pct=0.06,
            take_profit_pct=0.12,
            horizon_days=30,
        ))

        return self

    def map_signal(self, signal: ResearchSignal) -> List[TradeCandidate]:
        """
        Map a single research signal to trade candidates.

        Args:
            signal: ResearchSignal to map

        Returns:
            List of TradeCandidate objects (may be empty)
        """
        # Check if signal meets minimum confidence
        if signal.confidence < 0.3:
            return []

        # Find matching rules
        matching_rules = self._find_matching_rules(signal.region)

        if not matching_rules:
            # Default: no tickers, no trade
            return []

        trades = []
        for rule in matching_rules:
            # Determine trade direction
            signal_dir = signal.direction.value
            trade_dir = rule.direction_map.get(signal_dir, Direction.FLAT)

            if trade_dir == Direction.FLAT:
                continue

            # Calculate position size
            size_pct = self._calculate_position_size(signal, rule)

            # Create trade for each ticker
            for ticker in rule.tickers:
                trade = TradeCandidate(
                    strategy=signal.strategy,
                    timestamp=signal.timestamp,
                    ticker=ticker,
                    asset_type=rule.asset_type,
                    direction=trade_dir,
                    horizon_days=rule.horizon_days,
                    size_pct=size_pct,
                    stop_loss_pct=rule.stop_loss_pct,
                    take_profit_pct=rule.take_profit_pct,
                    rationale=self._build_rationale(signal, trade_dir, ticker),
                    source_signals=[f"{signal.region}_{signal.timestamp}"],
                    probability=signal.confidence,
                )
                trades.append(trade)

        return trades

    def map_signals(self, signals: List[ResearchSignal]) -> List[TradeCandidate]:
        """
        Map multiple research signals to trade candidates.

        Args:
            signals: List of ResearchSignal objects

        Returns:
            List of TradeCandidate objects
        """
        all_trades = []

        for signal in signals:
            trades = self.map_signal(signal)
            all_trades.extend(trades)

        # Aggregate by ticker (combine signals for same ticker)
        aggregated = self._aggregate_trades(all_trades)

        return aggregated

    def aggregate_signals(
        self,
        signals: List[ResearchSignal],
        group_by: str = "ticker",
    ) -> List[TradeCandidate]:
        """
        Aggregate multiple signals before mapping to trades.

        This combines signals from multiple regions for the same underlying thesis.

        Args:
            signals: List of ResearchSignal objects
            group_by: How to group ("ticker" or "strategy")

        Returns:
            List of aggregated TradeCandidate objects
        """
        # Group signals by strategy
        by_strategy = defaultdict(list)
        for signal in signals:
            by_strategy[signal.strategy].append(signal)

        # Create aggregated signal for each strategy
        aggregated_signals = []
        for strategy, strat_signals in by_strategy.items():
            agg_signal = self._aggregate_signals_by_strategy(strat_signals)
            aggregated_signals.append(agg_signal)

        # Map aggregated signals to trades
        return self.map_signals(aggregated_signals)

    def _find_matching_rules(self, region: str) -> List[TradingRule]:
        """Find trading rules that match a region."""
        matching = []

        for pattern, rules in self._rule_index.items():
            # Simple pattern matching (can be extended)
            if pattern == "*" or pattern in region.lower() or region.lower() in pattern.lower():
                matching.extend(rules)

        return matching

    def _calculate_position_size(
        self,
        signal: ResearchSignal,
        rule: TradingRule,
    ) -> float:
        """Calculate position size based on signal and rule."""
        size = rule.base_size_pct

        # Scale by confidence if enabled
        if rule.scale_by_confidence:
            size = size * signal.confidence

        # Scale by signal strength
        size = size * min(signal.strength / 2.0, 1.5)  # Cap at 1.5x

        # Clamp to max size
        size = min(size, rule.max_size_pct)

        return size

    def _build_rationale(
        self,
        signal: ResearchSignal,
        direction: Direction,
        ticker: str,
    ) -> str:
        """Build human-readable rationale for the trade."""
        return f"{signal.strategy}: {signal.thesis}. Trade {direction.value} {ticker}."

    def _aggregate_signals_by_strategy(
        self,
        signals: List[ResearchSignal],
    ) -> ResearchSignal:
        """Aggregate multiple signals of the same strategy."""
        if not signals:
            raise ValueError("Cannot aggregate empty signal list")

        # Use the most recent timestamp
        latest = max(signals, key=lambda s: s.timestamp)

        # Average the values
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_quality = np.mean([s.data_quality for s in signals])

        # Weighted vote for direction
        long_votes = sum(s.confidence for s in signals if s.direction == Direction.LONG)
        short_votes = sum(s.confidence for s in signals if s.direction == Direction.SHORT)

        if long_votes > short_votes:
            direction = Direction.LONG
        elif short_votes > long_votes:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # Combine regions
        regions = ", ".join(set(s.region for s in signals))

        return ResearchSignal(
            strategy=signals[0].strategy,
            timestamp=latest.timestamp,
            region=f"aggregated({len(signals)} regions)",
            direction=direction,
            strength=avg_strength,
            confidence=avg_confidence,
            data_quality=avg_quality,
            sample_count=sum(s.sample_count for s in signals),
            coverage_ratio=np.mean([s.coverage_ratio for s in signals]),
            thesis=f"Aggregated signal from {len(signals)} regions: {regions}",
        )

    def _aggregate_trades(
        self,
        trades: List[TradeCandidate],
    ) -> List[TradeCandidate]:
        """Aggregate trades for the same ticker."""
        # Group by ticker and direction
        groups = defaultdict(list)
        for trade in trades:
            key = (trade.ticker, trade.direction)
            groups[key].append(trade)

        aggregated = []
        for (ticker, direction), group_trades in groups.items():
            # Combine parameters
            total_size = sum(t.size_pct for t in group_trades)
            avg_probability = np.mean([t.probability for t in group_trades if t.probability])

            # Combine source signals
            all_sources = []
            for t in group_trades:
                all_sources.extend(t.source_signals)

            # Use the most aggressive risk params
            min_stop = min(t.stop_loss_pct for t in group_trades)
            max_profit = max(t.take_profit_pct for t in group_trades)

            aggregated_trade = TradeCandidate(
                strategy=group_trades[0].strategy,
                timestamp=datetime.now().isoformat(),
                ticker=ticker,
                asset_type=group_trades[0].asset_type,
                direction=direction,
                horizon_days=int(np.mean([t.horizon_days for t in group_trades])),
                size_pct=min(total_size, 0.10),  # Cap at 10%
                stop_loss_pct=min_stop,
                take_profit_pct=max_profit,
                rationale=f"Aggregated from {len(group_trades)} signals",
                source_signals=list(set(all_sources)),
                probability=avg_probability,
            )
            aggregated.append(aggregated_trade)

        return aggregated


def create_default_mapper() -> TradeMapper:
    """
    Create a trade mapper with default rules.

    Returns:
        TradeMapper with pre-configured rules
    """
    mapper = TradeMapper()
    mapper.add_default_rules()
    return mapper


__all__ = [
    "TradeMapper",
    "TradingRule",
    "create_default_mapper",
]
