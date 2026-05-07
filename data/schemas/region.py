"""
Region configuration schemas.

Defines the structure for monitoring regions.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class MonitoringType(Enum):
    """Types of monitoring."""
    CHOKEPOINT = "chokepoint"
    PORT_LOGISTICS = "port_logistics"
    AUTO_INVENTORY = "auto_inventory"
    OIL_STORAGE = "oil_storage"
    AGRICULTURE = "agriculture"
    AGRICULTURAL = "agricultural"
    POWER_PLANT = "power_plant"


@dataclass
class InstrumentSchema:
    """
    Tradable instrument schema.
    """
    ticker: str
    name: str
    asset_type: str  # equity, etf, commodity, etc.

    # Optional fields
    exchange: Optional[str] = None
    currency: Optional[str] = "USD"
    enabled_for_backtest: bool = True
    enabled_for_trading: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "asset_type": self.asset_type,
            "exchange": self.exchange,
            "currency": self.currency,
            "enabled_for_backtest": self.enabled_for_backtest,
            "enabled_for_trading": self.enabled_for_trading,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentSchema":
        """Create from dictionary."""
        return cls(
            ticker=data["ticker"],
            name=data.get("name", data["ticker"]),
            asset_type=data.get("asset_type", "equity"),
            exchange=data.get("exchange"),
            currency=data.get("currency", "USD"),
            enabled_for_backtest=data.get("enabled_for_backtest", True),
            enabled_for_trading=data.get("enabled_for_trading", True),
        )


@dataclass
class RegionSchema:
    """
    Base schema for region configuration.
    """
    id: str
    name: str
    type: MonitoringType
    aoi_file: str
    instruments: List[InstrumentSchema]

    # Optional fields
    active: bool = True
    meta_group: Optional[str] = None
    meta_weight: float = 1.0
    confirmations_required: int = 1

    # Additional metadata
    description: Optional[str] = None
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value if isinstance(self.type, MonitoringType) else self.type,
            "aoi_file": self.aoi_file,
            "instruments": [i.to_dict() for i in self.instruments],
            "active": self.active,
            "meta_group": self.meta_group,
            "meta_weight": self.meta_weight,
            "confirmations_required": self.confirmations_required,
            "description": self.description,
            "location": self.location,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionSchema":
        """Create from dictionary."""
        # Convert type string to enum
        monitoring_type = data.get("type", "unknown")
        if isinstance(monitoring_type, str):
            try:
                monitoring_type = MonitoringType(monitoring_type)
            except ValueError:
                monitoring_type = MonitoringType.AGRICULTURE  # Default

        # Parse instruments
        instruments = []
        for instr_data in data.get("instruments", []):
            if isinstance(instr_data, dict):
                instruments.append(InstrumentSchema.from_dict(instr_data))
            elif isinstance(instr_data, str):
                instruments.append(InstrumentSchema(
                    ticker=instr_data,
                    name=instr_data,
                    asset_type="equity",
                ))

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            type=monitoring_type,
            aoi_file=data["aoi_file"],
            instruments=instruments,
            active=data.get("active", True),
            meta_group=data.get("meta_group"),
            meta_weight=float(data.get("meta_weight", 1.0)),
            confirmations_required=int(data.get("confirmations_required", 1)),
            description=data.get("description"),
            location=data.get("location"),
            metadata=data.get("metadata"),
        )


@dataclass
class MetaGroupSchema:
    """
    Schema for meta signal groups.
    """
    id: str
    label: str
    type: str = "meta_signal"
    instruments: List[Dict[str, Any]] = field(default_factory=list)

    # Bias descriptions
    bullish_bias: str = "Bullish prices"
    bearish_bias: str = "Bearish prices"
    neutral_bias: str = "Mixed regional signal"

    # Trading settings
    portfolio_trade: bool = True
    confirmations_required: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "instruments": self.instruments,
            "bullish_bias": self.bullish_bias,
            "bearish_bias": self.bearish_bias,
            "neutral_bias": self.neutral_bias,
            "portfolio_trade": self.portfolio_trade,
            "confirmations_required": self.confirmations_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaGroupSchema":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data.get("label", data["id"]),
            type=data.get("type", "meta_signal"),
            instruments=data.get("instruments", []),
            bullish_bias=data.get("bullish_bias", "Bullish prices"),
            bearish_bias=data.get("bearish_bias", "Bearish prices"),
            neutral_bias=data.get("neutral_bias", "Mixed regional signal"),
            portfolio_trade=data.get("portfolio_trade", True),
            confirmations_required=int(data.get("confirmations_required", 1)),
        )


__all__ = [
    "RegionSchema",
    "MetaGroupSchema",
    "InstrumentSchema",
    "MonitoringType",
]
