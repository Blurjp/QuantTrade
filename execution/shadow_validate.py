"""
Shadow mode validation — runs the execution pipeline end-to-end with
simulated signals to verify signal→order→fill→ledger→portfolio flow.

Usage:
    python -m execution.shadow_validate
    python -m execution.shadow_validate --date 2026-04-15
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from execution.models import (
    OrderClass,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from execution.service import ExecutionService
from execution.reconciler import Reconciler

logger = logging.getLogger(__name__)


def _make_signal_intent(
    symbol: str,
    direction: str,
    region_id: str,
    date: str,
    price: float,
    notional: float = 1000.0,
) -> OrderIntent:
    side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL
    pos_intent = PositionIntent.OPEN_POSITION
    coid = ExecutionService.make_client_order_id(region_id, symbol, direction.lower(), date)

    return OrderIntent(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=coid,
        created_at=datetime.now(timezone.utc),
        notional=notional,
        order_class=OrderClass.BRACKET if direction == "LONG" else OrderClass.SIMPLE,
        stop_loss_stop=price * 0.92 if direction == "LONG" else None,
        take_profit_limit=price * 1.10 if direction == "LONG" else None,
        position_intent=pos_intent,
        rationale=f"Shadow validation: {region_id} {direction} {symbol}",
        metadata={"price": price},
    )


TEST_SIGNALS = [
    {"symbol": "XLE", "direction": "LONG", "region": "hormuz", "price": 85.0, "notional": 1000},
    {"symbol": "CORN", "direction": "LONG", "region": "brazil_soy", "price": 58.0, "notional": 800},
    {"symbol": "SOYB", "direction": "SHORT", "region": "argentina_pampas", "price": 30.0, "notional": 500},
    {"symbol": "WEAT", "direction": "LONG", "region": "usa_wheat_plains", "price": 52.0, "notional": 1200},
    {"symbol": "XLE", "direction": "LONG", "region": "hormuz", "price": 85.0, "notional": 1000},
    {"symbol": "XRT", "direction": "LONG", "region": "us_retail_walmart", "price": 82.0, "notional": 2000},
]


def run_shadow_validation(
    date_str: str = None,
    ledger_path: str = "outputs/execution/validation_test.sqlite",
) -> Dict:
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    results = {
        "date": date_str,
        "total_signals": len(TEST_SIGNALS),
        "submitted": 0,
        "filled": 0,
        "rejected": 0,
        "duplicate_rejected": 0,
        "risk_rejected": 0,
        "errors": [],
        "order_details": [],
    }

    halt_path = Path("outputs/execution/VALIDATION_HALT")
    halt_path.parent.mkdir(parents=True, exist_ok=True)

    svc = ExecutionService(
        ledger_path=ledger_path,
        execution_mode="shadow",
        halt_trading_path=str(halt_path),
    )

    logger.info("Shadow validation: submitting %d test signals", len(TEST_SIGNALS))

    for sig in TEST_SIGNALS:
        try:
            intent = _make_signal_intent(
                symbol=sig["symbol"],
                direction=sig["direction"],
                region_id=sig["region"],
                date=date_str,
                price=sig["price"],
                notional=sig["notional"],
            )
            result = svc.submit(intent)

            entry = {
                "coid": intent.client_order_id,
                "symbol": sig["symbol"],
                "direction": sig["direction"],
                "status": result.status.value,
                "reason": result.rejection_reason,
                "filled_qty": result.filled_qty,
                "filled_price": result.filled_avg_price,
            }
            results["order_details"].append(entry)

            if result.status == OrderStatus.FILLED:
                results["filled"] += 1
                results["submitted"] += 1
            elif result.status == OrderStatus.REJECTED:
                results["rejected"] += 1
                if "duplicate" in (result.rejection_reason or ""):
                    results["duplicate_rejected"] += 1
                else:
                    results["risk_rejected"] += 1
            else:
                results["submitted"] += 1

        except Exception as e:
            results["errors"].append(f"{sig['symbol']}: {e}")
            logger.error("Shadow validation error for %s: %s", sig["symbol"], e)

    logger.info("Running reconciler...")
    reconciler = Reconciler(svc)
    report = reconciler.run()
    results["reconciler"] = {
        "stranded_found": report.stranded_found,
        "stranded_cancelled": report.stranded_cancelled,
        "stranded_resubmitted": report.stranded_resubmitted,
        "drift_found": report.drift_found,
        "has_alert": report.has_alert,
    }

    results["ledger_orders"] = svc.ledger._conn.execute(
        "SELECT COUNT(*) as cnt FROM orders"
    ).fetchone()["cnt"]
    results["ledger_fills"] = svc.ledger._conn.execute(
        "SELECT COUNT(*) as cnt FROM fills"
    ).fetchone()["cnt"]
    results["ledger_risk"] = svc.ledger._conn.execute(
        "SELECT COUNT(*) as cnt FROM risk_decisions"
    ).fetchone()["cnt"]

    passed = (
        results["filled"] >= 3
        and results["duplicate_rejected"] >= 1
        and results["risk_rejected"] >= 0
        and len(results["errors"]) == 0
        and results["reconciler"]["has_alert"] is False
    )
    results["passed"] = passed

    print(f"\n{'='*60}")
    print(f"SHADOW VALIDATION {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*60}")
    print(f"  Signals:    {results['total_signals']}")
    print(f"  Filled:     {results['filled']}")
    print(f"  Rejected:   {results['rejected']} (dup={results['duplicate_rejected']}, risk={results['risk_rejected']})")
    print(f"  Errors:     {len(results['errors'])}")
    print(f"  Ledger:     {results['ledger_orders']} orders, {results['ledger_fills']} fills, {results['ledger_risk']} risk decisions")
    print(f"  Reconciler: stranded={results['reconciler']['stranded_found']}, drift={results['reconciler']['drift_found']}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--ledger", default="outputs/execution/validation_test.sqlite")
    args = parser.parse_args()

    results = run_shadow_validation(date_str=args.date, ledger_path=args.ledger)
    sys.exit(0 if results["passed"] else 1)
