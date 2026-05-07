"""
Tests for the reconciler and market hours risk gate.

Covers:
- Reconciler: stranded pending orders (no submitted_at)
  - Old stranded orders are cancelled
  - Recent stranded orders are resubmitted
  - Resubmit failures are reported
- Reconciler: drift detection (accepted orders gone from broker)
- Market hours check in _check_risk for live mode
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from execution.brokers.shadow import ShadowBrokerClient
from execution.ledger import OrderLedger
from execution.models import (
    OrderClass,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from execution.reconciler import Reconciler, ReconcileReport
from execution.service import ExecutionService


@pytest.fixture
def tmp_ledger(tmp_path):
    db = tmp_path / "test_reconcile.sqlite"
    return OrderLedger(db_path=str(db))


@pytest.fixture
def service(tmp_path):
    db = tmp_path / "test_reconcile.sqlite"
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
    coid="reconcile-coid-001",
    created_at=None,
    price=50.0,
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
        metadata={"price": price},
    )


def _insert_stranded_order(ledger, coid, symbol="XLE", created_at=None, age_minutes=10):
    now = datetime.now(timezone.utc)
    created = created_at or (now - timedelta(minutes=age_minutes))

    intent = OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=coid,
        created_at=created,
        quantity=10.0,
        limit_price=50.0,
        metadata={"price": 50.0},
    )
    ledger.insert_order(intent, broker="shadow")
    return intent


class TestReconcileStrandedOrders:
    def test_old_stranded_order_is_cancelled(self, service, tmp_ledger):
        coid = "strand-old-001"
        _insert_stranded_order(service.ledger, coid, age_minutes=30)

        reconciler = Reconciler(service, stranded_ttl_seconds=300)
        report = reconciler.run()

        order = service.ledger.get_order(coid)
        assert order["status"] == "canceled"
        assert report.stranded_found == 1
        assert report.stranded_cancelled == 1

    def test_recent_stranded_order_is_resubmitted(self, service):
        coid = "strand-recent-001"
        _insert_stranded_order(service.ledger, coid, age_minutes=1)

        reconciler = Reconciler(service, stranded_ttl_seconds=300)
        report = reconciler.run()

        order = service.ledger.get_order(coid)
        assert order["status"] == "filled"
        assert order["submitted_at"] is not None
        assert report.stranded_found == 1
        assert report.stranded_resubmitted == 1

    def test_no_stranded_orders_clean_report(self, service):
        intent = _intent(coid="normal-001", price=50.0)
        service.submit(intent)

        reconciler = Reconciler(service)
        report = reconciler.run()

        assert report.stranded_found == 0
        assert report.stranded_cancelled == 0
        assert report.stranded_resubmitted == 0
        assert not report.has_alert

    def test_multiple_stranded_orders_mixed(self, service):
        _insert_stranded_order(service.ledger, "strand-mix-old", age_minutes=20)
        _insert_stranded_order(service.ledger, "strand-mix-new", age_minutes=1)
        _insert_stranded_order(service.ledger, "strand-mix-very-old", age_minutes=60)

        reconciler = Reconciler(service, stranded_ttl_seconds=300)
        report = reconciler.run()

        assert report.stranded_found == 3
        assert report.stranded_cancelled == 2
        assert report.stranded_resubmitted == 1

    def test_resubmit_failure_reported(self, service):
        coid = "strand-fail-001"
        _insert_stranded_order(service.ledger, coid, age_minutes=1)

        service.broker.submit_order = MagicMock(
            side_effect=Exception("broker connection refused")
        )

        reconciler = Reconciler(service, stranded_ttl_seconds=300)
        report = reconciler.run()

        assert report.stranded_found == 1
        assert report.strand_resubmit_failed == 1
        assert report.has_alert

    def test_reconciliation_run_logged(self, service):
        _insert_stranded_order(service.ledger, "strand-log-001", age_minutes=30)

        reconciler = Reconciler(service, stranded_ttl_seconds=300)
        report = reconciler.run()

        rows = service.ledger._conn.execute(
            "SELECT * FROM reconciliation_runs"
        ).fetchall()
        assert len(rows) == 1
        run = dict(rows[0])
        assert run["status"] == "ok"
        assert run["orders_drift"] == 0


class TestReconcileDriftDetection:
    def test_drifted_order_updated_to_expired(self, service):
        intent = _intent(coid="drift-001", price=50.0)
        service.ledger.insert_order(intent, broker="shadow")
        service.ledger.update_submitted_at(intent.client_order_id)
        service.ledger.update_order_status(
            intent.client_order_id,
            "accepted",
            broker_order_id="shadow-drift-123",
        )

        service.broker.get_open_orders = MagicMock(return_value=[])
        service.broker.is_market_open = MagicMock(return_value=True)

        reconciler = Reconciler(service)
        report = reconciler.run()

        order = service.ledger.get_order("drift-001")
        assert order["status"] == "expired"
        assert report.drift_found == 1
        assert report.drift_updated == 1

    def test_no_drift_when_order_still_open_at_broker(self, service):
        intent = _intent(coid="nodrift-001", price=50.0)
        service.ledger.insert_order(intent, broker="shadow")
        service.ledger.update_submitted_at(intent.client_order_id)
        service.ledger.update_order_status(
            intent.client_order_id,
            "accepted",
            broker_order_id="shadow-nodrift-123",
        )

        service.broker.get_open_orders = MagicMock(return_value=[
            OrderResult(
                client_order_id="nodrift-001",
                status=OrderStatus.ACCEPTED,
                broker_order_id="shadow-nodrift-123",
            )
        ])

        reconciler = Reconciler(service)
        report = reconciler.run()

        assert report.drift_found == 0
        assert report.drift_updated == 0


class TestMarketHoursRiskGate:
    def test_shadow_mode_ignores_market_hours(self, service):
        service.broker.is_market_open = MagicMock(return_value=False)
        service.execution_mode = "shadow"

        intent = _intent(coid="mh-shadow-001", price=50.0)
        result = service.submit(intent)
        assert result.status == OrderStatus.FILLED

    def test_live_mode_rejects_when_market_closed(self, tmp_path):
        db = tmp_path / "test_mh.sqlite"
        halt = tmp_path / "HALT_TRADING"

        mock_broker = MagicMock()
        mock_broker.is_market_open.return_value = False
        mock_broker.get_account.return_value = MagicMock(equity=100000.0)
        mock_broker.get_positions.return_value = []

        with patch.dict(os.environ, {
            "EXECUTION_MODE": "live",
            "LIVE_TRADING_ENABLED": "true",
            "BROKER": "alpaca",
        }):
            svc = ExecutionService.__new__(ExecutionService)
            svc.execution_mode = "live"
            svc.halt_trading_path = halt
            svc.ledger = OrderLedger(db_path=str(db))
            svc.broker = mock_broker

            intent = _intent(coid="mh-closed-001", price=50.0)
            result = svc.submit(intent)
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "market_closed"

    def test_live_mode_accepts_when_market_open(self, tmp_path):
        db = tmp_path / "test_mh2.sqlite"
        halt = tmp_path / "HALT_TRADING"

        mock_broker = MagicMock()
        mock_broker.is_market_open.return_value = True
        mock_broker.get_account.return_value = MagicMock(equity=100000.0)
        mock_broker.get_positions.return_value = []
        mock_broker.submit_order.return_value = OrderResult(
            client_order_id="mh-open-001",
            status=OrderStatus.FILLED,
            broker_order_id="broker-001",
            filled_qty=20.0,
            filled_avg_price=50.0,
            filled_at=datetime.now(timezone.utc),
        )

        with patch.dict(os.environ, {
            "EXECUTION_MODE": "live",
            "LIVE_TRADING_ENABLED": "true",
            "BROKER": "alpaca",
        }):
            svc = ExecutionService.__new__(ExecutionService)
            svc.execution_mode = "live"
            svc.halt_trading_path = halt
            svc.ledger = OrderLedger(db_path=str(db))
            svc.broker = mock_broker

            intent = _intent(coid="mh-open-001", price=50.0)
            result = svc.submit(intent)
            assert result.status == OrderStatus.FILLED

    def test_live_mode_rejects_when_market_status_unknown(self, tmp_path):
        db = tmp_path / "test_mh3.sqlite"
        halt = tmp_path / "HALT_TRADING"

        mock_broker = MagicMock()
        mock_broker.is_market_open.side_effect = Exception("API timeout")
        mock_broker.get_account.return_value = MagicMock(equity=100000.0)
        mock_broker.get_positions.return_value = []

        with patch.dict(os.environ, {
            "EXECUTION_MODE": "live",
            "LIVE_TRADING_ENABLED": "true",
            "BROKER": "alpaca",
        }):
            svc = ExecutionService.__new__(ExecutionService)
            svc.execution_mode = "live"
            svc.halt_trading_path = halt
            svc.ledger = OrderLedger(db_path=str(db))
            svc.broker = mock_broker

            intent = _intent(coid="mh-unknown-001", price=50.0)
            result = svc.submit(intent)
            assert result.status == OrderStatus.REJECTED
            assert result.rejection_reason == "market_status_unknown"
