"""
End-to-end shadow test — validates the full auto-trade pipeline
from signal through ExecutionService to portfolio position.

Simulates the scheduler's auto-trade loop without needing
satellite data, Bayesian fusion, or real prices.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from execution.models import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from execution.service import ExecutionService
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio


@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace with portfolio and execution service."""
    portfolio = MultiAssetPortfolio(
        initial_capital=100000,
        output_base=str(tmp_path / "outputs"),
    )
    svc = ExecutionService(
        ledger_path=str(tmp_path / "exec" / "orders.sqlite"),
        halt_trading_path=str(tmp_path / "HALT_TRADING"),
    )
    return {
        "portfolio": portfolio,
        "service": svc,
        "tmp": tmp_path,
    }


def _bayesian_decision(ticker, direction, position_value, confidence, region="test_region"):
    return {
        "ticker": ticker,
        "direction": direction,
        "position_value": position_value,
        "fused_confidence": confidence,
        "kelly_fraction": 0.05,
        "conflict": False,
        "sources": [
            {"type": "chokepoint", "confidence": confidence, "region": region},
        ],
    }


def _submit_and_open(svc, portfolio, ticker, direction, position_value, price, prefix="bayes"):
    """
    Mirrors the scheduler's auto-trade path:
    1. Build OrderIntent
    2. ExecutionService.submit()
    3. If accepted/filled → portfolio.open_position()
    """
    now = datetime.now(timezone.utc)
    side = OrderSide.BUY if direction == "long" else OrderSide.SELL

    broker_positions = {p.symbol: p for p in svc.broker.get_positions()}
    has_long = ticker in broker_positions and broker_positions[ticker].side == "long"
    if side == OrderSide.SELL and has_long:
        pos_intent = PositionIntent.CLOSE_POSITION
    else:
        pos_intent = PositionIntent.OPEN_POSITION

    intent = OrderIntent(
        symbol=ticker,
        side=side,
        notional=position_value,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=svc.make_client_order_id(
            prefix, ticker, direction, now.strftime("%Y%m%d"),
        ),
        created_at=now,
        position_intent=pos_intent,
        rationale=f"E2E test: {prefix} {direction} {ticker}",
        metadata={"price": price},
    )
    result = svc.submit(intent)

    if result.status.value in ("accepted", "filled"):
        portfolio.open_position(
            ticker=ticker,
            direction=direction,
            price=price,
            value=position_value,
            rationale=f"E2E: {prefix} {direction} {ticker}",
            asset_class="commodity",
        )
        return True, result
    return False, result


class TestE2EShadowSingleTrade:
    def test_single_buy_fills_and_creates_position(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        ok, result = _submit_and_open(
            svc, portfolio, "XLE", "long", 2000.0, 58.85
        )

        assert ok
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty > 0
        assert result.filled_avg_price == 58.85

        assert "XLE" in portfolio.positions
        pos = portfolio.positions["XLE"]
        assert pos.direction == "long"
        assert pos.entry_price == 58.85
        assert portfolio.cash == 100000 - 2000.0

    def test_ledger_records_fill(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        _submit_and_open(svc, portfolio, "SOYB", "long", 1000.0, 25.0)

        order = svc.ledger.get_order(
            svc.make_client_order_id("bayes", "SOYB", "long", datetime.now(timezone.utc).strftime("%Y%m%d"))
        )
        assert order is not None
        assert order["status"] == "filled"
        assert order["symbol"] == "SOYB"

        fills = svc.ledger.get_recent_fills(limit=10)
        assert any(f["symbol"] == "SOYB" for f in fills)

    def test_shadow_broker_tracks_position(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        _submit_and_open(svc, portfolio, "XLE", "long", 2000.0, 50.0)

        broker_pos = {p.symbol: p for p in svc.broker.get_positions()}
        assert "XLE" in broker_pos
        assert broker_pos["XLE"].side == "long"
        assert broker_pos["XLE"].qty == 40.0


class TestE2EShadowMultipleTrades:
    def test_three_buys_within_daily_cap(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        trades = [
            ("XLE", "long", 2000.0, 58.85),
            ("SOYB", "long", 1500.0, 25.0),
            ("WEAT", "long", 1000.0, 45.0),
        ]
        filled = 0
        for ticker, direction, value, price in trades:
            ok, _ = _submit_and_open(svc, portfolio, ticker, direction, value, price)
            if ok:
                filled += 1

        assert filled == 3
        assert len(portfolio.positions) == 3
        assert portfolio.cash == 100000 - 2000 - 1500 - 1000

    def test_daily_cap_blocks_fourth_trade(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        with patch.dict(os.environ, {"MAX_DAILY_NOTIONAL": "5000"}):
            for i in range(4):
                _submit_and_open(
                    svc, portfolio,
                    f"TICK{i}", "long", 2000.0, 50.0,
                    prefix=f"cap{i}",
                )

        assert len(portfolio.positions) == 2
        order_tick2 = svc.ledger.get_order(
            svc.make_client_order_id("cap2", "TICK2", "long", datetime.now(timezone.utc).strftime("%Y%m%d"))
        )
        assert order_tick2["status"] == "rejected"

    def test_duplicate_same_ticker_same_day_rejected(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        ok1, _ = _submit_and_open(svc, portfolio, "XLE", "long", 1000.0, 50.0)
        ok2, r2 = _submit_and_open(svc, portfolio, "XLE", "long", 1000.0, 50.0)

        assert ok1
        assert not ok2
        assert r2.rejection_reason == "duplicate_client_order_id"
        assert len(portfolio.positions) == 1


class TestE2EShadowRiskGates:
    def test_halt_trading_blocks_everything(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        Path(workspace["service"].halt_trading_path).touch()

        ok, result = _submit_and_open(svc, portfolio, "XLE", "long", 1000.0, 50.0)
        assert not ok
        assert result.rejection_reason == "halt_trading"
        assert len(portfolio.positions) == 0

    def test_short_sell_blocked_by_default(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        ok, result = _submit_and_open(svc, portfolio, "XLE", "short", 1000.0, 50.0)
        assert not ok
        assert result.rejection_reason == "short_selling_not_allowed"

    def test_short_sell_allowed_with_flag(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        with patch.dict(os.environ, {"ALLOW_SHORT_SELLING": "true"}):
            ok, result = _submit_and_open(svc, portfolio, "XLE", "short", 1000.0, 50.0)
            assert ok
            assert result.status == OrderStatus.FILLED

    def test_concentration_limit_blocks(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        with patch.dict(os.environ, {"MAX_POSITION_PCT": "0.01"}):
            ok1, _ = _submit_and_open(
                svc, portfolio, "XLE", "long", 2000.0, 50.0, prefix="conc1"
            )
            ok2, r2 = _submit_and_open(
                svc, portfolio, "XLE", "long", 2000.0, 50.0, prefix="conc2"
            )

        assert ok1
        assert not ok2
        assert r2.rejection_reason == "position_concentration_exceeded"


class TestE2EShadowClosePosition:
    def test_sell_reduces_broker_position(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        ok1, _ = _submit_and_open(
            svc, portfolio, "XLE", "long", 2000.0, 50.0, prefix="buy"
        )
        assert ok1

        with patch.dict(os.environ, {"ALLOW_SHORT_SELLING": "true"}):
            ok2, result = _submit_and_open(
                svc, portfolio, "XLE", "short", 1000.0, 51.0, prefix="sell"
            )
        assert ok2
        assert result.status == OrderStatus.FILLED

        broker_pos = {p.symbol: p for p in svc.broker.get_positions()}
        assert "XLE" in broker_pos
        assert broker_pos["XLE"].side == "long"
        assert broker_pos["XLE"].qty < 40.0


class TestE2EShadowReconciliation:
    def test_ledger_and_portfolio_agree(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        _submit_and_open(svc, portfolio, "XLE", "long", 2000.0, 50.0)
        _submit_and_open(svc, portfolio, "SOYB", "long", 1000.0, 25.0)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_notional = svc.ledger.get_daily_notional(today)
        assert daily_notional == 3000.0

        fills = svc.ledger.get_recent_fills(limit=10)
        assert len(fills) == 2

        assert set(portfolio.positions.keys()) == {"XLE", "SOYB"}
        assert portfolio.cash == 100000 - 3000.0

    def test_risk_decisions_all_recorded(self, workspace):
        svc = workspace["service"]
        portfolio = workspace["portfolio"]

        _submit_and_open(svc, portfolio, "XLE", "long", 1000.0, 50.0)
        Path(svc.halt_trading_path).touch()
        _submit_and_open(svc, portfolio, "SOYB", "long", 1000.0, 25.0)
        svc.halt_trading_path.unlink()

        rows = svc.ledger._conn.execute(
            "SELECT reason FROM risk_decisions ORDER BY id"
        ).fetchall()
        reasons = [r["reason"] for r in rows]
        assert "all_checks_passed" in reasons
        assert "halt_trading" in reasons
