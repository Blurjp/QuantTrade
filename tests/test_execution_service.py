"""
Tests for the execution service layer.

Covers:
- ExecutionService.submit() with shadow broker
- Risk gate: HALT_TRADING, duplicate client_order_id, max notional, order TTL
- EXECUTION_MODE=shadow overrides BROKER=alpaca (no Alpaca adapter imported)
- Bracket order model support
- Ledger persistence (orders, fills, risk_decisions)
- scheduler_service.py has no ungated open_position in auto-trade paths
- One-order-per-instrument-per-day policy (make_client_order_id semantics)
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from execution.models import (
    OrderClass,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from execution.brokers.shadow import ShadowBrokerClient
from execution.ledger import OrderLedger
from execution.service import ExecutionService


@pytest.fixture
def service(tmp_path):
    db = tmp_path / "test_orders.sqlite"
    halt = tmp_path / "HALT_TRADING"
    return ExecutionService(
        ledger_path=str(db),
        halt_trading_path=str(halt),
    )


def _intent(
    symbol="XLE",
    side=OrderSide.BUY,
    notional=1000.0,
    quantity=None,
    coid="test-coid-001",
    created_at=None,
    price=50.0,
    order_class=OrderClass.SIMPLE,
    stop_loss_stop=None,
    take_profit_limit=None,
    position_intent=PositionIntent.OPEN_POSITION,
):
    return OrderIntent(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=coid,
        created_at=created_at or datetime.now(timezone.utc),
        notional=notional,
        quantity=quantity,
        order_class=order_class,
        stop_loss_stop=stop_loss_stop,
        take_profit_limit=take_profit_limit,
        position_intent=position_intent,
        metadata={"price": price},
    )


class TestExecutionServiceSubmit:
    def test_basic_buy_fills(self, service):
        intent = _intent(symbol="XLE", notional=1000.0, price=58.85)
        result = service.submit(intent)

        assert result.status == OrderStatus.FILLED
        assert result.filled_qty > 0
        assert result.filled_avg_price == 58.85
        assert result.broker_order_id is not None

    def test_basic_sell_fills(self, service):
        intent_buy = _intent(
            symbol="XLE", notional=1000.0, price=50.0, coid="buy-001"
        )
        service.submit(intent_buy)

        intent_sell = _intent(
            symbol="XLE",
            side=OrderSide.SELL,
            notional=500.0,
            price=51.0,
            coid="sell-001",
            position_intent=PositionIntent.CLOSE_POSITION,
        )
        result = service.submit(intent_sell)

        assert result.status == OrderStatus.FILLED
        assert result.filled_qty > 0

    def test_quantity_based_order(self, service):
        intent = _intent(quantity=10.0, notional=None, price=100.0, coid="qty-001")
        result = service.submit(intent)

        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 10.0
        assert result.filled_avg_price == 100.0

    def test_order_persisted_in_ledger(self, service):
        intent = _intent(coid="persist-001", symbol="SOYB")
        service.submit(intent)

        order = service.ledger.get_order("persist-001")
        assert order is not None
        assert order["symbol"] == "SOYB"
        assert order["status"] == "filled"

    def test_fill_recorded_in_ledger(self, service):
        intent = _intent(coid="fill-test-001", symbol="WEAT", price=25.0)
        service.submit(intent)

        fills = service.ledger.get_recent_fills(limit=10)
        fill_symbols = [f["symbol"] for f in fills]
        assert "WEAT" in fill_symbols


class TestRiskGateHaltTrading:
    def test_halt_trading_blocks_order(self, service, tmp_path):
        halt_path = tmp_path / "HALT_TRADING"
        halt_path.touch()

        intent = _intent(coid="halt-001")
        result = service.submit(intent)

        assert result.status == OrderStatus.REJECTED
        assert result.rejection_reason == "halt_trading"

    def test_halt_trading_records_risk_decision(self, service, tmp_path):
        halt_path = tmp_path / "HALT_TRADING"
        halt_path.touch()

        intent = _intent(coid="halt-rd-001")
        service.submit(intent)

        row = service.ledger._conn.execute(
            "SELECT * FROM risk_decisions WHERE client_order_id = ?",
            ("halt-rd-001",),
        ).fetchone()
        assert row is not None
        assert row["approved"] == 0
        assert row["reason"] == "halt_trading"


class TestRiskGateDuplicate:
    def test_duplicate_client_order_id_rejected_after_fill(self, service):
        intent1 = _intent(coid="dup-001")
        result1 = service.submit(intent1)
        assert result1.status == OrderStatus.FILLED

        intent2 = _intent(coid="dup-001")
        result2 = service.submit(intent2)
        assert result2.status == OrderStatus.REJECTED
        assert result2.rejection_reason == "duplicate_client_order_id"

    def test_duplicate_client_order_id_rejected_after_rejection(self, service, tmp_path):
        halt_path = tmp_path / "HALT_TRADING"
        halt_path.touch()

        intent1 = _intent(coid="dup-rej-001")
        result1 = service.submit(intent1)
        assert result1.status == OrderStatus.REJECTED

        halt_path.unlink()
        intent2 = _intent(coid="dup-rej-001")
        result2 = service.submit(intent2)
        assert result2.status == OrderStatus.REJECTED
        assert result2.rejection_reason == "duplicate_client_order_id"

    def test_different_coid_accepted(self, service):
        intent1 = _intent(coid="uniq-001")
        intent2 = _intent(coid="uniq-002")
        r1 = service.submit(intent1)
        r2 = service.submit(intent2)
        assert r1.status == OrderStatus.FILLED
        assert r2.status == OrderStatus.FILLED


class TestRiskGateMaxNotional:
    def test_max_notional_exceeded(self, service):
        with patch.dict(os.environ, {"MAX_ORDER_NOTIONAL": "500"}):
            intent = _intent(notional=1000.0, coid="maxn-001")
            result = service.submit(intent)
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "max_notional_exceeded"

    def test_max_notional_at_limit_passes(self, service):
        with patch.dict(os.environ, {"MAX_ORDER_NOTIONAL": "5000"}):
            intent = _intent(notional=5000.0, coid="maxn-002")
            result = service.submit(intent)
            assert result.status == OrderStatus.FILLED


class TestRiskGateTTL:
    def test_expired_order_rejected(self, service):
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        with patch.dict(os.environ, {"ORDER_TTL_MINUTES": "120"}):
            intent = _intent(coid="ttl-001", created_at=old_time)
            result = service.submit(intent)
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "order_expired"

    def test_fresh_order_accepted(self, service):
        with patch.dict(os.environ, {"ORDER_TTL_MINUTES": "120"}):
            intent = _intent(coid="ttl-002")
            result = service.submit(intent)
            assert result.status == OrderStatus.FILLED


class TestRiskGateRejectedLogged:
    def test_rejected_intent_logged_in_risk_decisions(self, service, tmp_path):
        halt_path = tmp_path / "HALT_TRADING"
        halt_path.touch()

        intent = _intent(coid="audit-001")
        service.submit(intent)

        row = service.ledger._conn.execute(
            "SELECT * FROM risk_decisions WHERE client_order_id = ?",
            ("audit-001",),
        ).fetchone()
        assert row is not None
        assert row["approved"] == 0
        assert row["reason"] == "halt_trading"

        order = service.ledger.get_order("audit-001")
        assert order is not None
        assert order["status"] == "rejected"


class TestRiskGateDailyNotionalCap:
    def test_daily_cap_blocks_order(self, service):
        with patch.dict(os.environ, {"MAX_DAILY_NOTIONAL": "2500"}):
            service.submit(_intent(coid="dcap-001", notional=1000.0))
            service.submit(_intent(coid="dcap-002", notional=1000.0))
            result = service.submit(_intent(coid="dcap-003", notional=1000.0))
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "daily_notional_cap_exceeded"

    def test_daily_cap_allows_under_limit(self, service):
        with patch.dict(os.environ, {"MAX_DAILY_NOTIONAL": "15000"}):
            r1 = service.submit(_intent(coid="dcap-ok-001", notional=1000.0))
            r2 = service.submit(_intent(coid="dcap-ok-002", notional=1000.0))
            assert r1.status == OrderStatus.FILLED
            assert r2.status == OrderStatus.FILLED


class TestRiskGateShortSelling:
    def test_open_short_rejected_by_default(self, service):
        with patch.dict(os.environ, {"ALLOW_SHORT_SELLING": ""}):
            intent = _intent(
                coid="short-blocked-001",
                side=OrderSide.SELL,
                position_intent=PositionIntent.OPEN_POSITION,
            )
            result = service.submit(intent)
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "short_selling_not_allowed"

    def test_close_long_sell_allowed_without_flag(self, service):
        with patch.dict(os.environ, {"ALLOW_SHORT_SELLING": ""}):
            service.submit(_intent(coid="buy-for-close-001"))
            intent = _intent(
                coid="close-long-001",
                side=OrderSide.SELL,
                position_intent=PositionIntent.CLOSE_POSITION,
            )
            result = service.submit(intent)
            assert result.status == OrderStatus.FILLED

    def test_open_short_allowed_with_flag(self, service):
        with patch.dict(os.environ, {"ALLOW_SHORT_SELLING": "true"}):
            intent = _intent(
                coid="short-allowed-001",
                side=OrderSide.SELL,
                position_intent=PositionIntent.OPEN_POSITION,
            )
            result = service.submit(intent)
            assert result.status == OrderStatus.FILLED


class TestRiskGatePositionConcentration:
    def test_concentration_blocks_additional_order(self, service):
        with patch.dict(os.environ, {"MAX_POSITION_PCT": "0.01"}):
            service.submit(
                _intent(coid="conc-001", symbol="XLE", notional=1000.0, price=50.0)
            )
            result = service.submit(
                _intent(coid="conc-002", symbol="XLE", notional=1000.0, price=50.0)
            )
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "position_concentration_exceeded"

    def test_different_symbol_under_concentration(self, service):
        with patch.dict(os.environ, {"MAX_POSITION_PCT": "0.01"}):
            r1 = service.submit(
                _intent(coid="conc-diff-001", symbol="XLE", notional=1000.0)
            )
            r2 = service.submit(
                _intent(coid="conc-diff-002", symbol="SOYB", notional=1000.0)
            )
            assert r1.status == OrderStatus.FILLED
            assert r2.status == OrderStatus.FILLED


class TestShadowOverridesAlpaca:
    def test_shadow_mode_ignores_broker_alpaca(self, tmp_path):
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        with patch.dict(os.environ, {"BROKER": "alpaca", "EXECUTION_MODE": "shadow"}):
            svc = ExecutionService(
                ledger_path=str(db),
                execution_mode="shadow",
                halt_trading_path=str(halt),
            )
            assert isinstance(svc.broker, ShadowBrokerClient)

    def test_shadow_mode_does_not_import_alpaca(self, tmp_path):
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        with patch.dict(os.environ, {"BROKER": "alpaca"}):
            svc = ExecutionService(
                ledger_path=str(db),
                execution_mode="shadow",
                halt_trading_path=str(halt),
            )
            broker_module = type(svc.broker).__module__
            assert "alpaca" not in broker_module

    def test_paper_mode_uses_shadow(self, tmp_path):
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        svc = ExecutionService(
            ledger_path=str(db),
            execution_mode="paper",
            halt_trading_path=str(halt),
        )
        assert isinstance(svc.broker, ShadowBrokerClient)


class TestLiveModeRequiresExplicitEnable:
    """
    Live mode needs two switches: EXECUTION_MODE=live AND
    LIVE_TRADING_ENABLED=true. __init__ raises on invalid BROKER;
    _check_risk blocks if LIVE_TRADING_ENABLED is missing.
    """

    def test_live_mode_init_raises_without_broker(self, tmp_path):
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        with patch.dict(os.environ, {"BROKER": ""}, clear=False):
            with pytest.raises(RuntimeError, match="EXECUTION_MODE=live"):
                ExecutionService(
                    ledger_path=str(db),
                    execution_mode="live",
                    halt_trading_path=str(halt),
                )

    def test_live_mode_init_raises_with_unknown_broker(self, tmp_path):
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        with patch.dict(os.environ, {"BROKER": "ibkr"}, clear=False):
            with pytest.raises(RuntimeError, match="EXECUTION_MODE=live"):
                ExecutionService(
                    ledger_path=str(db),
                    execution_mode="live",
                    halt_trading_path=str(halt),
                )

    def test_live_mode_rejects_without_enable_flag(self, tmp_path):
        """
        Even if __init__ somehow succeeded with a live broker,
        _check_risk rejects when LIVE_TRADING_ENABLED != 'true'.
        Test by constructing service with a shadow broker but live mode,
        then verifying _check_risk blocks submit.
        """
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": ""}, clear=False):
            svc = ExecutionService(
                ledger_path=str(db),
                execution_mode="shadow",
                halt_trading_path=str(halt),
            )
            svc.execution_mode = "live"
            intent = _intent(coid="live-no-enable-001")
            result = svc.submit(intent)
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "live_trading_not_enabled"

    def test_shadow_mode_does_not_require_enable_flag(self, tmp_path):
        db = tmp_path / "test_orders.sqlite"
        halt = tmp_path / "HALT_TRADING"
        svc = ExecutionService(
            ledger_path=str(db),
            execution_mode="shadow",
            halt_trading_path=str(halt),
        )
        intent = _intent(coid="shadow-no-enable-001")
        result = svc.submit(intent)
        assert result.status == OrderStatus.FILLED


class TestBracketOrders:
    def test_bracket_order_with_legs(self, service):
        intent = _intent(
            coid="bracket-001",
            notional=1000.0,
            price=100.0,
            order_class=OrderClass.BRACKET,
            stop_loss_stop=90.0,
            take_profit_limit=120.0,
        )
        result = service.submit(intent)

        assert result.status == OrderStatus.FILLED
        assert result.legs is not None
        assert len(result.legs) == 2

        leg_ids = [leg.broker_order_id for leg in result.legs]
        assert any("-sl" in lid for lid in leg_ids)
        assert any("-tp" in lid for lid in leg_ids)

    def test_bracket_order_stored_in_ledger(self, service):
        intent = _intent(
            coid="bracket-ledger-001",
            notional=1000.0,
            price=100.0,
            order_class=OrderClass.BRACKET,
            stop_loss_stop=90.0,
            take_profit_limit=120.0,
        )
        service.submit(intent)

        order = service.ledger.get_order("bracket-ledger-001")
        assert order is not None
        assert order["order_class"] == "bracket"
        assert order["stop_loss_stop"] == 90.0
        assert order["take_profit_limit"] == 120.0


class TestPositionIntent:
    def test_sell_without_existing_long_is_open_short(self, service):
        with patch.dict(os.environ, {"ALLOW_SHORT_SELLING": "true"}):
            intent_sell = _intent(
                side=OrderSide.SELL,
                notional=500.0,
                price=50.0,
                coid="short-001",
                position_intent=PositionIntent.OPEN_POSITION,
            )
            result = service.submit(intent_sell)
            assert result.status == OrderStatus.FILLED

            positions = {p.symbol: p for p in service.broker.get_positions()}
            assert "XLE" in positions
            assert positions["XLE"].side == "short"

    def test_sell_with_existing_long_is_close(self, service):
        _buy = _intent(notional=1000.0, price=50.0, coid="buy-close-001")
        service.submit(_buy)

        intent_sell = _intent(
            side=OrderSide.SELL,
            notional=500.0,
            price=51.0,
            coid="close-001",
            position_intent=PositionIntent.CLOSE_POSITION,
        )
        result = service.submit(intent_sell)
        assert result.status == OrderStatus.FILLED


class TestQuantityNotionalMutualExclusion:
    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            OrderIntent(
                symbol="XLE",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id="bad-001",
                created_at=datetime.now(timezone.utc),
                quantity=10.0,
                notional=1000.0,
            )

    def test_neither_set_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            OrderIntent(
                symbol="XLE",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id="bad-002",
                created_at=datetime.now(timezone.utc),
                quantity=None,
                notional=None,
            )


class TestMakeClientOrderId:
    def test_deterministic(self):
        id1 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        id2 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        assert id1 == id2

    def test_different_inputs_produce_different_ids(self):
        id1 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        id2 = ExecutionService.make_client_order_id("fb", "XLE", "long", "20260504")
        assert id1 != id2

    def test_format(self):
        coid = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        assert coid.startswith("qt-bayes-XLE-long-")

    def test_same_ticker_same_direction_same_day_produces_same_id(self):
        id1 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        id2 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        assert id1 == id2

    def test_different_direction_same_day_produces_different_id(self):
        id_long = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        id_short = ExecutionService.make_client_order_id("bayes", "XLE", "short", "20260504")
        assert id_long != id_short

    def test_different_day_same_ticker_produces_different_id(self):
        id_d1 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        id_d2 = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260505")
        assert id_d1 != id_d2


class TestOneOrderPerInstrumentPerDayPolicy:
    """
    make_client_order_id is deterministic per (prefix, symbol, direction, date).
    This means the scheduler can submit at most one order per instrument per
    direction per day — duplicate submissions are caught by the idempotency gate.
    """

    def test_same_instrument_direction_day_is_duplicate(self, service):
        coid = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")

        intent1 = _intent(coid=coid, symbol="XLE")
        r1 = service.submit(intent1)
        assert r1.status == OrderStatus.FILLED

        intent2 = _intent(coid=coid, symbol="XLE")
        r2 = service.submit(intent2)
        assert r2.status == OrderStatus.REJECTED
        assert r2.rejection_reason == "duplicate_client_order_id"

    def test_different_instrument_same_day_is_allowed(self, service):
        coid_xle = ExecutionService.make_client_order_id("bayes", "XLE", "long", "20260504")
        coid_soyb = ExecutionService.make_client_order_id("bayes", "SOYB", "long", "20260504")

        r1 = service.submit(_intent(coid=coid_xle, symbol="XLE"))
        r2 = service.submit(_intent(coid=coid_soyb, symbol="SOYB"))
        assert r1.status == OrderStatus.FILLED
        assert r2.status == OrderStatus.FILLED


class TestSchedulerNoDirectOpenPosition:
    def test_no_ungated_open_position_in_auto_trade_paths(self):
        """
        portfolio.open_position() may only appear inside a conditional block
        that checks a successful ExecutionService.submit() result — specifically
        inside `if result.status.value in ("accepted", "filled"):`.
        Any other occurrence is a violation of the Phase 0 mandate.
        """
        scheduler_path = Path(__file__).parent.parent / "scheduler_service.py"
        content = scheduler_path.read_text()
        lines = content.splitlines()

        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "portfolio.open_position(" not in stripped:
                continue

            found_gate = False
            for j in range(max(0, i - 20), i):
                ctx = lines[j].strip()
                if 'result.status.value in ("accepted", "filled")' in ctx:
                    found_gate = True
                    break

            if not found_gate:
                violations.append((i, stripped))

        assert len(violations) == 0, (
            f"Found ungated portfolio.open_position() at lines: "
            f"{violations}. All auto-trade open_position calls must be "
            f"gated by a successful ExecutionService.submit() result."
        )

    def test_no_execution_svc_none_bypass(self):
        """
        The 'elif not execution_svc' / 'else: result = None' fallback path
        that calls portfolio.open_position() directly must not exist.
        """
        scheduler_path = Path(__file__).parent.parent / "scheduler_service.py"
        content = scheduler_path.read_text()

        assert "elif not execution_svc" not in content, (
            "Found 'elif not execution_svc' bypass in scheduler_service.py. "
            "Phase 0 requires fail-closed: if ExecutionService is unavailable, "
            "auto-trading must be skipped entirely."
        )
