"""
Tests for AlpacaBrokerClient adapter.

All tests mock the alpaca TradingClient to avoid needing real credentials.
Tests verify correct mapping between execution models and Alpaca API types.
"""

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from execution.models import (
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


def _mock_alpaca_order(
    id="ord-123",
    client_order_id="test-coid",
    status="filled",
    filled_qty="10",
    filled_avg_price="50.0",
    submitted_at="2026-05-07T14:00:00Z",
    filled_at="2026-05-07T14:00:01Z",
    legs=None,
):
    return SimpleNamespace(
        id=id,
        client_order_id=client_order_id,
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        submitted_at=submitted_at,
        filled_at=filled_at,
        legs=legs,
    )


def _mock_alpaca_position(
    symbol="XLE",
    qty="40",
    side="long",
    avg_entry_price="50.0",
    current_price="51.0",
    market_value="2040.0",
    unrealized_pl="40.0",
    unrealized_plpc="0.02",
):
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        side=side,
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        market_value=market_value,
        unrealized_pl=unrealized_pl,
        unrealized_plpc=unrealized_plpc,
    )


def _mock_alpaca_account(
    equity="100000",
    cash="95000",
    buying_power="200000",
    initial_margin="5000",
    maintenance_margin="3000",
    pattern_day_trader=False,
    trading_blocked=False,
    account_blocked=False,
):
    return SimpleNamespace(
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        pattern_day_trader=pattern_day_trader,
        trading_blocked=trading_blocked,
        account_blocked=account_blocked,
    )


def _mock_alpaca_clock(is_open=True):
    return SimpleNamespace(is_open=is_open)


@pytest.fixture
def alpaca_client():
    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "testkey123",
        "ALPACA_SECRET_KEY": "testsecret456",
        "ALPACA_PAPER": "true",
    }):
        mock_tc = MagicMock()
        with patch("alpaca.trading.client.TradingClient", return_value=mock_tc):
            from execution.brokers.alpaca import AlpacaBrokerClient
            client = AlpacaBrokerClient()
            return client


def _intent(
    symbol="XLE",
    side=OrderSide.BUY,
    quantity=10.0,
    notional=None,
    order_class=OrderClass.SIMPLE,
    stop_loss_stop=None,
    take_profit_limit=None,
    position_intent=PositionIntent.OPEN_POSITION,
    coid="test-coid-001",
):
    return OrderIntent(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=coid,
        created_at=datetime.now(timezone.utc),
        quantity=quantity,
        notional=notional,
        order_class=order_class,
        stop_loss_stop=stop_loss_stop,
        take_profit_limit=take_profit_limit,
        position_intent=position_intent,
        metadata={"price": 50.0},
    )


class TestAlpacaSubmitOrder:
    def test_market_buy_qty(self, alpaca_client):
        alpaca_client._client.submit_order.return_value = _mock_alpaca_order(
            client_order_id="test-coid-001",
            status="filled",
            filled_qty="10",
            filled_avg_price="50.0",
        )
        intent = _intent(quantity=10.0)
        result = alpaca_client.submit_order(intent)

        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 10.0
        assert result.filled_avg_price == 50.0
        assert result.broker_order_id == "ord-123"

        call_kwargs = alpaca_client._client.submit_order.call_args
        order_data = call_kwargs[1]["order_data"] if "order_data" in (call_kwargs[1] or {}) else call_kwargs[0][0]
        assert order_data.symbol == "XLE"

    def test_market_buy_notional(self, alpaca_client):
        alpaca_client._client.submit_order.return_value = _mock_alpaca_order(
            status="accepted",
            filled_qty="0",
            filled_avg_price=None,
        )
        intent = _intent(quantity=None, notional=1000.0)
        result = alpaca_client.submit_order(intent)

        assert result.status == OrderStatus.ACCEPTED

    def test_sell_order(self, alpaca_client):
        alpaca_client._client.submit_order.return_value = _mock_alpaca_order(
            status="filled",
            filled_qty="10",
        )
        intent = _intent(
            side=OrderSide.SELL,
            quantity=10.0,
            position_intent=PositionIntent.CLOSE_POSITION,
        )
        result = alpaca_client.submit_order(intent)
        assert result.status == OrderStatus.FILLED

    def test_submit_rejected_returns_rejection(self, alpaca_client):
        alpaca_client._client.submit_order.side_effect = Exception("insufficient funds")
        intent = _intent(quantity=10.0)
        result = alpaca_client.submit_order(intent)

        assert result.status == OrderStatus.REJECTED
        assert "insufficient funds" in result.rejection_reason


class TestAlpacaBracketOrders:
    def test_bracket_order_submits_with_legs(self, alpaca_client):
        mock_leg_sl = _mock_alpaca_order(
            id="ord-123-sl",
            status="accepted",
            filled_qty="0",
            filled_avg_price=None,
        )
        mock_leg_tp = _mock_alpaca_order(
            id="ord-123-tp",
            status="accepted",
            filled_qty="0",
            filled_avg_price=None,
        )
        alpaca_client._client.submit_order.return_value = _mock_alpaca_order(
            status="filled",
            filled_qty="10",
            filled_avg_price="100.0",
            legs=[mock_leg_sl, mock_leg_tp],
        )

        intent = _intent(
            quantity=10.0,
            order_class=OrderClass.BRACKET,
            stop_loss_stop=90.0,
            take_profit_limit=120.0,
        )
        result = alpaca_client.submit_order(intent)

        assert result.status == OrderStatus.FILLED
        assert result.legs is not None
        assert len(result.legs) == 2


class TestAlpacaStatusMapping:
    def test_known_statuses(self, alpaca_client):
        from execution.brokers.alpaca import _map_status

        assert _map_status("filled") == OrderStatus.FILLED
        assert _map_status("new") == OrderStatus.ACCEPTED
        assert _map_status("accepted") == OrderStatus.ACCEPTED
        assert _map_status("partially_filled") == OrderStatus.PARTIALLY_FILLED
        assert _map_status("rejected") == OrderStatus.REJECTED
        assert _map_status("canceled") == OrderStatus.CANCELED
        assert _map_status("expired") == OrderStatus.EXPIRED

    def test_unknown_status_defaults_pending(self):
        from execution.brokers.alpaca import _map_status

        assert _map_status("unknown_status") == OrderStatus.PENDING


class TestAlpacaGetPositions:
    def test_returns_broker_positions(self, alpaca_client):
        alpaca_client._client.get_all_positions.return_value = [
            _mock_alpaca_position(symbol="XLE", qty="40", side="long"),
            _mock_alpaca_position(symbol="SOYB", qty="50", side="long"),
        ]
        positions = alpaca_client.get_positions()

        assert len(positions) == 2
        assert positions[0].symbol == "XLE"
        assert positions[0].qty == 40.0
        assert positions[0].side == "long"

    def test_empty_positions(self, alpaca_client):
        alpaca_client._client.get_all_positions.return_value = []
        assert alpaca_client.get_positions() == []

    def test_error_returns_empty(self, alpaca_client):
        alpaca_client._client.get_all_positions.side_effect = Exception("API error")
        assert alpaca_client.get_positions() == []


class TestAlpacaGetAccount:
    def test_returns_broker_account(self, alpaca_client):
        alpaca_client._client.get_account.return_value = _mock_alpaca_account()
        account = alpaca_client.get_account()

        assert account.equity == 100000.0
        assert account.buying_power == 200000.0
        assert not account.trading_blocked

    def test_error_returns_blocked_account(self, alpaca_client):
        alpaca_client._client.get_account.side_effect = Exception("timeout")
        account = alpaca_client.get_account()

        assert account.trading_blocked is True
        assert account.account_blocked is True


class TestAlpacaCancelOrder:
    def test_successful_cancel(self, alpaca_client):
        result = alpaca_client.cancel_order("ord-123")
        assert result.status == OrderStatus.CANCELED
        assert result.broker_order_id == "ord-123"

    def test_failed_cancel(self, alpaca_client):
        alpaca_client._client.cancel_order_by_id.side_effect = Exception("not found")
        result = alpaca_client.cancel_order("ord-999")
        assert result.status == OrderStatus.REJECTED


class TestAlpacaIsMarketOpen:
    def test_market_open(self, alpaca_client):
        alpaca_client._client.get_clock.return_value = _mock_alpaca_clock(is_open=True)
        assert alpaca_client.is_market_open() is True

    def test_market_closed(self, alpaca_client):
        alpaca_client._client.get_clock.return_value = _mock_alpaca_clock(is_open=False)
        assert alpaca_client.is_market_open() is False

    def test_error_returns_false(self, alpaca_client):
        alpaca_client._client.get_clock.side_effect = Exception("timeout")
        assert alpaca_client.is_market_open() is False


class TestAlpacaInit:
    def test_missing_credentials_raises(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}, clear=False):
            with patch("alpaca.trading.client.TradingClient"):
                from execution.brokers.alpaca import AlpacaBrokerClient
                with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
                    AlpacaBrokerClient()

    def test_paper_mode_by_default(self):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "key",
            "ALPACA_SECRET_KEY": "secret",
        }):
            with patch("alpaca.trading.client.TradingClient") as MockTC:
                from execution.brokers.alpaca import AlpacaBrokerClient
                client = AlpacaBrokerClient()
                _, kwargs = MockTC.call_args
                assert kwargs["paper"] is True

    def test_live_mode_when_paper_false(self):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "key",
            "ALPACA_SECRET_KEY": "secret",
            "ALPACA_PAPER": "false",
        }):
            with patch("alpaca.trading.client.TradingClient") as MockTC:
                from execution.brokers.alpaca import AlpacaBrokerClient
                client = AlpacaBrokerClient()
                _, kwargs = MockTC.call_args
                assert kwargs["paper"] is False
