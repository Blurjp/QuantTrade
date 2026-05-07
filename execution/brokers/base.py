"""
Broker client protocol.

All broker adapters must implement this interface.
The ExecutionService calls only these methods — no broker-specific
logic should leak into the service or scheduler layers.
"""

from typing import List, Optional, Protocol, runtime_checkable

from execution.models import (
    BrokerAccount,
    BrokerPosition,
    FillEvent,
    OrderIntent,
    OrderResult,
)


@runtime_checkable
class BrokerClient(Protocol):
    def submit_order(self, intent: OrderIntent) -> OrderResult: ...
    def cancel_order(self, broker_order_id: str) -> OrderResult: ...
    def get_open_orders(self) -> List[OrderResult]: ...
    def get_positions(self) -> List[BrokerPosition]: ...
    def get_account(self) -> BrokerAccount: ...
    def get_fills(self, since: Optional[str] = None) -> List[FillEvent]: ...
    def is_market_open(self) -> bool: ...
