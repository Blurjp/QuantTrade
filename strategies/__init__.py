"""
QuantTrade strategies module.

All trading strategies implement the BaseStrategy protocol defined in base.py.
This ensures consistent integration with backtesting, paper trading, and dashboard.

Usage:
    from strategies.base import BaseStrategy, ResearchSignal, TradeCandidate

    # Each strategy is in its own subdirectory
    from strategies.auto_inventory import AutoInventoryStrategy
    from strategies.chokepoint import ChokepointStrategy
"""

from strategies.base import (
    # Protocol
    BaseStrategy,
    # Data classes
    ResearchSignal,
    TradeCandidate,
    StrategyConfig,
    # Enums
    Direction,
    AssetType,
)

__all__ = [
    "BaseStrategy",
    "ResearchSignal",
    "TradeCandidate",
    "StrategyConfig",
    "Direction",
    "AssetType",
]
