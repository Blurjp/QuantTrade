"""
Asset and portfolio schemas.

Defines the structure for tradable assets and portfolio positions.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class AssetType(Enum):
    """Asset types."""
    EQUITY = "equity"
    ETF = "etf"
    COMMODITY_FUTURES = "commodity_futures"
    CRYPTO = "crypto"
    FOREX = "forex"
    BOND = "bond"
    INDEX = "index"


class PositionDirection(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class AssetSchema:
    """
    Base schema for a tradable asset.
    """
    ticker: str
    name: str
    asset_type: AssetType

    # Optional fields
    exchange: Optional[str] = None
    currency: Optional[str] = "USD"
    description: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None

    # Trading constraints
    min_order_size: Optional[float] = None
    max_position_pct: Optional[float] = None  # Max % of portfolio

    # Data sources
    yahoo_symbol: Optional[str] = None
    bloomberg_symbol: Optional[str] = None
    factset_symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "asset_type": self.asset_type.value if isinstance(self.asset_type, AssetType) else self.asset_type,
            "exchange": self.exchange,
            "currency": self.currency,
            "description": self.description,
            "sector": self.sector,
            "country": self.country,
            "min_order_size": self.min_order_size,
            "max_position_pct": self.max_position_pct,
            "yahoo_symbol": self.yahoo_symbol or self.ticker,
            "bloomberg_symbol": self.bloomberg_symbol,
            "factset_symbol": self.factset_symbol,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetSchema":
        """Create from dictionary."""
        asset_type = data.get("asset_type", "equity")
        if isinstance(asset_type, str):
            try:
                asset_type = AssetType(asset_type)
            except ValueError:
                asset_type = AssetType.EQUITY

        return cls(
            ticker=data["ticker"],
            name=data.get("name", data["ticker"]),
            asset_type=asset_type,
            exchange=data.get("exchange"),
            currency=data.get("currency", "USD"),
            description=data.get("description"),
            sector=data.get("sector"),
            country=data.get("country"),
            min_order_size=data.get("min_order_size"),
            max_position_pct=data.get("max_position_pct"),
            yahoo_symbol=data.get("yahoo_symbol"),
            bloomberg_symbol=data.get("bloomberg_symbol"),
            factset_symbol=data.get("factset_symbol"),
        )


@dataclass
class PositionSchema:
    """
    Schema for a portfolio position.
    """
    ticker: str
    direction: PositionDirection
    quantity: float
    entry_price: float
    entry_date: str

    # Optional fields
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Metadata
    source_region: Optional[str] = None
    source_strategy: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "direction": self.direction.value if isinstance(self.direction, PositionDirection) else self.direction,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "source_region": self.source_region,
            "source_strategy": self.source_strategy,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionSchema":
        """Create from dictionary."""
        direction = data.get("direction", "long")
        if isinstance(direction, str):
            try:
                direction = PositionDirection(direction.lower())
            except ValueError:
                direction = PositionDirection.LONG

        return cls(
            ticker=data["ticker"],
            direction=direction,
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            entry_date=data["entry_date"],
            current_price=data.get("current_price"),
            market_value=data.get("market_value"),
            unrealized_pnl=data.get("unrealized_pnl"),
            unrealized_pnl_pct=data.get("unrealized_pnl_pct"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            source_region=data.get("source_region"),
            source_strategy=data.get("source_strategy"),
            notes=data.get("notes"),
        )


@dataclass
class PortfolioSnapshotSchema:
    """
    Schema for a portfolio value snapshot.
    """
    date: str
    total_value: float
    cash: float
    positions_value: float

    # Optional breakdown
    positions: Optional[List[PositionSchema]] = None
    daily_pnl: Optional[float] = None
    daily_pnl_pct: Optional[float] = None

    # Performance metrics
    peak_value: Optional[float] = None
    drawdown: Optional[float] = None
    drawdown_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "positions": [p.to_dict() for p in self.positions] if self.positions else [],
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "peak_value": self.peak_value,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
        }


@dataclass
class TradeExecutionSchema:
    """
    Schema for a trade execution.
    """
    ticker: str
    action: str  # OPEN_LONG, OPEN_SHORT, CLOSE
    quantity: float
    price: float
    timestamp: str

    # Context
    source_region: Optional[str] = None
    source_strategy: Optional[str] = None
    signal_strength: Optional[float] = None
    confidence: Optional[str] = None

    # Commission
    commission: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
            "source_region": self.source_region,
            "source_strategy": self.source_strategy,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "commission": self.commission,
        }


__all__ = [
    "AssetSchema",
    "PositionSchema",
    "PortfolioSnapshotSchema",
    "TradeExecutionSchema",
    "AssetType",
    "PositionDirection",
]
