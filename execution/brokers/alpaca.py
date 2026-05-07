"""
Alpaca broker adapter — live trading via Alpaca Markets API.

Uses alpaca-py (TradingClient) to submit orders, query positions,
fetch fills, and check account/market state. All broker-specific
mapping happens here; the ExecutionService never touches Alpaca types.

Environment variables:
- ALPACA_API_KEY:    Required. API key from Alpaca dashboard.
- ALPACA_SECRET_KEY: Required. Secret key from Alpaca dashboard.
- ALPACA_PAPER:      Optional. "true" (default) uses paper trading API.

Safety: this adapter is only instantiated when EXECUTION_MODE=live
AND LIVE_TRADING_ENABLED=true AND BROKER=alpaca. All three must be set.
"""

import logging
import os
from datetime import datetime, timezone
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
    PositionIntent,
    TimeInForce,
)

logger = logging.getLogger(__name__)

_ALPACA_STATUS_MAP: Dict[str, OrderStatus] = {
    "new": OrderStatus.ACCEPTED,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.ACCEPTED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "held": OrderStatus.ACCEPTED,
    "stopped": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.PENDING,
    "pending_replace": OrderStatus.PENDING,
    "pending_review": OrderStatus.PENDING,
    "replaced": OrderStatus.ACCEPTED,
    "calculated": OrderStatus.ACCEPTED,
}

_ALPACA_POSITION_INTENT_MAP = {
    (OrderSide.BUY, PositionIntent.OPEN_POSITION): "buy_to_open",
    (OrderSide.BUY, PositionIntent.CLOSE_POSITION): "buy_to_close",
    (OrderSide.SELL, PositionIntent.OPEN_POSITION): "sell_to_open",
    (OrderSide.SELL, PositionIntent.CLOSE_POSITION): "sell_to_close",
}


def _map_status(alpaca_status: str) -> OrderStatus:
    return _ALPACA_STATUS_MAP.get(alpaca_status, OrderStatus.PENDING)


class AlpacaBrokerClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
    ):
        from alpaca.trading.client import TradingClient

        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")

        if paper is None:
            paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self._paper = paper

        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are required. "
                "Set them in environment or pass to constructor."
            )

        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )

        mode = "paper" if self._paper else "live"
        logger.info(
            "AlpacaBrokerClient initialized (%s mode, key=%s...%s)",
            mode,
            self._api_key[:4],
            self._api_key[-4:],
        )

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import (
            OrderSide as AlpacaSide,
            TimeInForce as AlpacaTIF,
            OrderClass as AlpacaOrderClass,
            PositionIntent as AlpacaPositionIntent,
        )

        side = AlpacaSide.BUY if intent.side == OrderSide.BUY else AlpacaSide.SELL
        tif = AlpacaTIF.DAY if intent.time_in_force == TimeInForce.DAY else AlpacaTIF.GTC

        pos_intent_str = _ALPACA_POSITION_INTENT_MAP.get(
            (intent.side, intent.position_intent)
        )
        alpaca_pos_intent = None
        if pos_intent_str:
            alpaca_pos_intent = AlpacaPositionIntent(pos_intent_str)

        common_kwargs = {
            "symbol": intent.symbol,
            "side": side,
            "time_in_force": tif,
            "client_order_id": intent.client_order_id,
            "order_class": AlpacaOrderClass.SIMPLE,
        }

        if intent.quantity is not None and intent.quantity > 0:
            common_kwargs["qty"] = str(intent.quantity)
        elif intent.notional is not None and intent.notional > 0:
            common_kwargs["notional"] = str(intent.notional)

        if alpaca_pos_intent is not None:
            common_kwargs["position_intent"] = alpaca_pos_intent

        if intent.order_class == OrderClass.BRACKET:
            common_kwargs["order_class"] = AlpacaOrderClass.BRACKET
            if intent.take_profit_limit is not None:
                common_kwargs["take_profit"] = {"limit_price": str(intent.take_profit_limit)}
            if intent.stop_loss_stop is not None:
                sl_kwargs = {"stop_price": str(intent.stop_loss_stop)}
                if intent.stop_loss_limit is not None:
                    sl_kwargs["limit_price"] = str(intent.stop_loss_limit)
                common_kwargs["stop_loss"] = sl_kwargs

        try:
            if intent.order_type == OrderType.LIMIT:
                if intent.limit_price is None:
                    return OrderResult(
                        client_order_id=intent.client_order_id,
                        status=OrderStatus.REJECTED,
                        rejection_reason="limit_order_requires_limit_price",
                    )
                common_kwargs["limit_price"] = str(intent.limit_price)
                order_req = LimitOrderRequest(**common_kwargs)
            elif intent.order_type == OrderType.STOP:
                return self._submit_stop_order(intent, common_kwargs, AlpacaSide, AlpacaTIF)
            elif intent.order_type == OrderType.STOP_LIMIT:
                return self._submit_stop_limit_order(intent, common_kwargs)
            else:
                order_req = MarketOrderRequest(**common_kwargs)

            resp = self._client.submit_order(order_data=order_req)
            return self._alpaca_order_to_result(resp)

        except Exception as e:
            error_msg = str(e)
            logger.error("Alpaca submit failed for %s: %s", intent.symbol, error_msg)
            return OrderResult(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"alpaca_error: {error_msg[:200]}",
            )

    def _submit_stop_order(self, intent, common_kwargs, AlpacaSide, AlpacaTIF):
        if intent.stop_price is None:
            return OrderResult(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="stop_order_requires_stop_price",
            )
        try:
            from alpaca.trading.requests import StopOrderRequest
            common_kwargs["stop_price"] = str(intent.stop_price)
            order_req = StopOrderRequest(**common_kwargs)
            resp = self._client.submit_order(order_data=order_req)
            return self._alpaca_order_to_result(resp)
        except Exception as e:
            return OrderResult(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"alpaca_error: {str(e)[:200]}",
            )

    def _submit_stop_limit_order(self, intent, common_kwargs):
        if intent.stop_price is None or intent.limit_price is None:
            return OrderResult(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="stop_limit_order_requires_stop_and_limit_price",
            )
        try:
            from alpaca.trading.requests import LimitOrderRequest
            common_kwargs["stop_price"] = str(intent.stop_price)
            common_kwargs["limit_price"] = str(intent.limit_price)
            order_req = LimitOrderRequest(**common_kwargs)
            resp = self._client.submit_order(order_data=order_req)
            return self._alpaca_order_to_result(resp)
        except Exception as e:
            return OrderResult(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"alpaca_error: {str(e)[:200]}",
            )

    def cancel_order(self, broker_order_id: str) -> OrderResult:
        try:
            self._client.cancel_order_by_id(broker_order_id)
            return OrderResult(
                client_order_id="",
                status=OrderStatus.CANCELED,
                broker_order_id=broker_order_id,
            )
        except Exception as e:
            return OrderResult(
                client_order_id="",
                status=OrderStatus.REJECTED,
                broker_order_id=broker_order_id,
                rejection_reason=str(e)[:200],
            )

    def get_open_orders(self) -> List[OrderResult]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(request=req)
            return [self._alpaca_order_to_result(o) for o in orders]
        except Exception as e:
            logger.error("Alpaca get_open_orders failed: %s", e)
            return []

    def get_positions(self) -> List[BrokerPosition]:
        try:
            positions = self._client.get_all_positions()
            result = []
            for p in positions:
                result.append(BrokerPosition(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    side=p.side,
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price) if p.current_price else 0.0,
                    market_value=float(p.market_value) if p.market_value else 0.0,
                    unrealized_pnl=float(p.unrealized_pl) if p.unrealized_pl else 0.0,
                    unrealized_pnl_pct=float(p.unrealized_plpc) if p.unrealized_plpc else 0.0,
                ))
            return result
        except Exception as e:
            logger.error("Alpaca get_positions failed: %s", e)
            return []

    def get_account(self) -> BrokerAccount:
        try:
            acct = self._client.get_account()
            return BrokerAccount(
                equity=float(acct.equity),
                cash=float(acct.cash),
                buying_power=float(acct.buying_power),
                initial_margin=float(acct.initial_margin) if acct.initial_margin else 0.0,
                maintenance_margin=float(acct.maintenance_margin) if acct.maintenance_margin else 0.0,
                pattern_day_trader=bool(acct.pattern_day_trader),
                trading_blocked=bool(acct.trading_blocked),
                account_blocked=bool(acct.account_blocked),
            )
        except Exception as e:
            logger.error("Alpaca get_account failed: %s", e)
            return BrokerAccount(
                equity=0, cash=0, buying_power=0,
                initial_margin=0, maintenance_margin=0,
                pattern_day_trader=False,
                trading_blocked=True,
                account_blocked=True,
            )

    def get_fills(self, since: Optional[str] = None) -> List[FillEvent]:
        try:
            activities = self._client.get_activities(
                activity_types="FILL",
                after=since,
            )
            fills = []
            for a in activities:
                raw_ts = getattr(a, "timestamp", None) or getattr(a, "transaction_time", None)
                fill_ts = datetime.now(timezone.utc)
                if raw_ts:
                    try:
                        fill_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass
                fills.append(FillEvent(
                    broker_order_id=getattr(a, "order_id", ""),
                    fill_id=getattr(a, "id", ""),
                    symbol=getattr(a, "symbol", ""),
                    side=OrderSide.BUY if getattr(a, "side", "") == "buy" else OrderSide.SELL,
                    quantity=float(getattr(a, "qty", 0)),
                    price=float(getattr(a, "price", 0)),
                    timestamp=fill_ts,
                ))
            return fills
        except Exception as e:
            logger.error("Alpaca get_fills failed: %s", e)
            return []

    def is_market_open(self) -> bool:
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception:
            return False

    def _alpaca_order_to_result(self, order) -> OrderResult:
        status = _map_status(getattr(order, "status", "pending"))
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        filled_avg_price = getattr(order, "filled_avg_price", None)
        if filled_avg_price is not None:
            filled_avg_price = float(filled_avg_price)

        raw = {}
        if hasattr(order, "id"):
            raw["id"] = str(order.id)
        if hasattr(order, "status"):
            raw["status"] = str(order.status)
        if hasattr(order, "submitted_at"):
            raw["submitted_at"] = str(order.submitted_at)
        if hasattr(order, "filled_at"):
            raw["filled_at"] = str(order.filled_at)

        legs = None
        if hasattr(order, "legs") and order.legs:
            legs = []
            for leg in order.legs:
                legs.append(OrderResult(
                    client_order_id=getattr(leg, "client_order_id", ""),
                    status=_map_status(getattr(leg, "status", "pending")),
                    broker_order_id=str(getattr(leg, "id", "")),
                    raw_response={"id": str(getattr(leg, "id", ""))},
                ))

        submitted_at = None
        if hasattr(order, "submitted_at") and order.submitted_at:
            submitted_at = order.submitted_at
            if isinstance(submitted_at, str):
                submitted_at = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))

        filled_at = None
        if hasattr(order, "filled_at") and order.filled_at:
            filled_at = order.filled_at
            if isinstance(filled_at, str):
                filled_at = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))

        return OrderResult(
            client_order_id=getattr(order, "client_order_id", ""),
            status=status,
            broker_order_id=str(getattr(order, "id", "")),
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            submitted_at=submitted_at,
            filled_at=filled_at,
            rejection_reason=None,
            legs=legs,
            raw_response=raw,
        )
