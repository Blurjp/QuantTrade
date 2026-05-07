"""
Shadow broker client.

Logs and simulates accepted orders without making any live broker calls.
Used in shadow mode (Phase 1) and as the default when EXECUTION_MODE
is not 'live'.

All simulated fills use the limit_price (if provided) or a price
stored in intent.metadata['price'] as a proxy for market price.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from execution.models import (
    BrokerAccount,
    BrokerPosition,
    FillEvent,
    OrderClass,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class ShadowBrokerClient:
    """
    Simulates a broker by logging orders to disk and memory.

    Never contacts any external API. Suitable for shadow mode
    testing and integration tests.
    """

    def __init__(self, output_dir: str = "outputs/execution/shadow"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._orders: Dict[str, OrderResult] = {}
        self._fills: List[FillEvent] = []
        self._positions: Dict[str, BrokerPosition] = {}

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        broker_order_id = f"shadow-{uuid.uuid4().hex[:12]}"

        price = intent.limit_price
        if price is None:
            price = intent.metadata.get("price", 0.0)

        qty = intent.quantity or 0.0
        if qty == 0 and intent.notional and price and price > 0:
            qty = intent.notional / price

        result = OrderResult(
            client_order_id=intent.client_order_id,
            status=OrderStatus.FILLED if price and price > 0 else OrderStatus.ACCEPTED,
            broker_order_id=broker_order_id,
            filled_qty=qty if price and price > 0 else 0.0,
            filled_avg_price=price if price and price > 0 else None,
            submitted_at=datetime.now(timezone.utc),
            filled_at=datetime.now(timezone.utc) if price and price > 0 else None,
        )

        if intent.order_class == OrderClass.BRACKET and result.status == OrderStatus.FILLED:
            result.legs = self._create_bracket_legs(intent, broker_order_id)

        self._orders[intent.client_order_id] = result

        if result.status == OrderStatus.FILLED:
            self._update_shadow_position(intent, qty, price)
            fill = FillEvent(
                broker_order_id=broker_order_id,
                fill_id=f"fill-{uuid.uuid4().hex[:8]}",
                symbol=intent.symbol,
                side=intent.side,
                quantity=qty,
                price=price,
                timestamp=datetime.now(timezone.utc),
            )
            self._fills.append(fill)

        self._persist_order(intent, result)

        logger.info(
            "SHADOW order %s: %s %s %s qty=%.2f price=%.2f status=%s",
            intent.client_order_id,
            intent.side.value,
            intent.symbol,
            intent.order_class.value,
            qty,
            price,
            result.status.value,
        )

        return result

    def cancel_order(self, broker_order_id: str) -> OrderResult:
        for cid, result in self._orders.items():
            if result.broker_order_id == broker_order_id:
                if result.status in (OrderStatus.PENDING, OrderStatus.ACCEPTED):
                    result.status = OrderStatus.CANCELED
                return result
        return OrderResult(
            client_order_id="unknown",
            status=OrderStatus.REJECTED,
            broker_order_id=broker_order_id,
            rejection_reason="order_not_found",
        )

    def get_open_orders(self) -> List[OrderResult]:
        return [
            r for r in self._orders.values()
            if r.status in (OrderStatus.PENDING, OrderStatus.ACCEPTED)
        ]

    def get_positions(self) -> List[BrokerPosition]:
        return list(self._positions.values())

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(
            equity=100000.0,
            cash=100000.0 - sum(p.market_value for p in self._positions.values()),
            buying_power=200000.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False,
        )

    def get_fills(self, since: Optional[str] = None) -> List[FillEvent]:
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f.timestamp.isoformat() >= since]

    def is_market_open(self) -> bool:
        return True

    def _create_bracket_legs(
        self, intent: OrderIntent, parent_id: str
    ) -> List[OrderResult]:
        legs = []
        if intent.stop_loss_stop:
            legs.append(OrderResult(
                client_order_id=f"{intent.client_order_id}-sl",
                status=OrderStatus.ACCEPTED,
                broker_order_id=f"{parent_id}-sl",
                raw_response={"stop_price": intent.stop_loss_stop},
            ))
        if intent.take_profit_limit:
            legs.append(OrderResult(
                client_order_id=f"{intent.client_order_id}-tp",
                status=OrderStatus.ACCEPTED,
                broker_order_id=f"{parent_id}-tp",
                raw_response={"limit_price": intent.take_profit_limit},
            ))
        return legs

    def _update_shadow_position(
        self, intent: OrderIntent, qty: float, price: float
    ):
        symbol = intent.symbol
        existing = self._positions.get(symbol)

        if intent.side == OrderSide.BUY:
            if existing and existing.side == "long":
                new_qty = existing.qty + qty
                new_avg = (
                    (existing.avg_entry_price * existing.qty + price * qty) / new_qty
                )
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol,
                    qty=new_qty,
                    side="long",
                    avg_entry_price=new_avg,
                    current_price=price,
                    market_value=new_qty * price,
                    unrealized_pnl=(price - new_avg) * new_qty,
                    unrealized_pnl_pct=(price - new_avg) / new_avg if new_avg else 0,
                )
            else:
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol,
                    qty=qty,
                    side="long",
                    avg_entry_price=price,
                    current_price=price,
                    market_value=qty * price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                )
        elif intent.side == OrderSide.SELL:
            if existing and existing.side == "long":
                remaining = existing.qty - qty
                if remaining <= 0:
                    del self._positions[symbol]
                else:
                    existing.qty = remaining
                    existing.market_value = remaining * price
                    existing.current_price = price
            else:
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol,
                    qty=qty,
                    side="short",
                    avg_entry_price=price,
                    current_price=price,
                    market_value=qty * price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                )

    def _persist_order(self, intent: OrderIntent, result: OrderResult):
        record = {
            "intent": {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "order_type": intent.order_type.value,
                "order_class": intent.order_class.value,
                "quantity": intent.quantity,
                "notional": intent.notional,
                "client_order_id": intent.client_order_id,
                "position_intent": intent.position_intent.value,
                "rationale": intent.rationale[:200],
            },
            "result": {
                "status": result.status.value,
                "broker_order_id": result.broker_order_id,
                "filled_qty": result.filled_qty,
                "filled_avg_price": result.filled_avg_price,
                "rejection_reason": result.rejection_reason,
            },
        }
        log_file = self.output_dir / f"shadow_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
