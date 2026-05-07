"""
Execution models — broker-neutral runtime objects.

These models are the interface between the trading decision layer
and any specific broker adapter. All broker-specific mapping happens
inside the broker adapter, never in the service or scheduler.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderClass(str, Enum):
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    OPG = "opg"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"
    EXPIRED = "expired"


class PositionIntent(str, Enum):
    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"


@dataclass
class OrderIntent:
    """
    Desired action before broker submission.

    Exactly one of quantity or notional must be > 0.
    For bracket orders, set order_class=BRACKET and provide
    stop_loss_stop and take_profit_limit.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce
    client_order_id: str
    created_at: datetime
    position_intent: PositionIntent = PositionIntent.OPEN_POSITION
    order_class: OrderClass = OrderClass.SIMPLE
    quantity: Optional[float] = None
    notional: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_limit: Optional[float] = None
    stop_loss_stop: Optional[float] = None
    stop_loss_limit: Optional[float] = None
    asset_class: str = "us_equity"
    rationale: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        has_qty = self.quantity is not None and self.quantity > 0
        has_notional = self.notional is not None and self.notional > 0
        if not has_qty and not has_notional:
            raise ValueError(
                "OrderIntent must have exactly one of quantity or notional > 0"
            )
        if has_qty and has_notional:
            raise ValueError(
                "OrderIntent quantity and notional are mutually exclusive"
            )

    def notional_value(self) -> float:
        if self.notional is not None and self.notional > 0:
            return self.notional
        if self.quantity is not None and self.limit_price is not None:
            return self.quantity * self.limit_price
        return 0.0


@dataclass
class OrderResult:
    client_order_id: str
    status: OrderStatus
    broker_order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    legs: Optional[List["OrderResult"]] = None
    raw_response: Optional[dict] = None


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class BrokerAccount:
    equity: float
    cash: float
    buying_power: float
    initial_margin: float
    maintenance_margin: float
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool


@dataclass
class FillEvent:
    broker_order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    details: dict = field(default_factory=dict)
