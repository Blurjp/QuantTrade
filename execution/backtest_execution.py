"""
Backfill execution test — replays historical signals through the execution
layer to measure simulated PnL over time.

Reads daily_summary.json files from outputs/{date}/, extracts actionable
signals, generates OrderIntents, submits through ExecutionService in shadow
mode, and computes cumulative PnL.

Usage:
    python -m execution.backtest_execution --start 2026-03-01 --end 2026-03-24
    python -m execution.backtest_execution --last 30
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

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
from execution.service import ExecutionService

logger = logging.getLogger(__name__)

PRICES = {
    "XLE": 85.0, "CORN": 58.0, "SOYB": 30.0, "WEAT": 52.0,
    "XRT": 82.0, "USO": 42.0, "GLD": 195.0, "FXI": 28.0,
    "OIH": 38.0, "XOP": 55.0, "BNO": 22.0, "CAT": 340.0,
    "DE": 400.0, "WMT": 65.0, "COST": 780.0, "HD": 350.0,
}

INSTRUMENT_MAP = {
    "WTI": "USO", "Brent": "USO", "Crude Oil": "USO",
    "Corn": "CORN", "Soybeans": "SOYB", "Wheat": "WEAT",
    "Gold": "GLD", "Soybean Oil": "SOYB",
}


def _resolve_symbol(instrument: str) -> Optional[str]:
    if instrument in PRICES:
        return instrument
    return INSTRUMENT_MAP.get(instrument)


def _load_daily_summaries(output_base: str, start: str, end: str) -> List[Dict]:
    base = Path(output_base)
    summaries = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        ds = current.strftime("%Y-%m-%d")
        summary_path = base / ds / "daily_summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
                data["_date"] = ds
                summaries.append(data)
            except Exception:
                pass
        current += timedelta(days=1)
    return summaries


def _extract_signals(summary: Dict) -> List[Dict]:
    signals = []
    for region_id, sig in summary.get("signals", {}).items():
        action = sig.get("trading_action", "FLAT")
        if action == "FLAT":
            continue
        confidence = sig.get("confidence", "Low")
        if confidence == "Low":
            continue
        instruments = sig.get("instruments", [])
        for inst in instruments:
            symbol = _resolve_symbol(inst)
            if symbol:
                direction = "LONG" if action == "LONG" else "SHORT"
                signals.append({
                    "region": region_id,
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence,
                    "price": PRICES[symbol],
                })
    return signals


def run_backtest(
    start: str,
    end: str,
    output_base: str = "outputs",
    initial_capital: float = 100000.0,
) -> Dict:
    ledger_path = "outputs/execution/backtest_test.sqlite"
    halt_path = Path("outputs/execution/BACKTEST_HALT")
    halt_path.parent.mkdir(parents=True, exist_ok=True)

    svc = ExecutionService(
        ledger_path=ledger_path,
        execution_mode="shadow",
        halt_trading_path=str(halt_path),
    )

    os.environ["ORDER_TTL_MINUTES"] = "999999"

    summaries = _load_daily_summaries(output_base, start, end)
    if not summaries:
        print(f"No daily summaries found between {start} and {end}")
        return {"error": "no data"}

    total_pnl = 0.0
    total_filled = 0
    total_rejected = 0
    daily_results = []
    positions: Dict[str, Dict] = {}
    cash = initial_capital

    for summary in summaries:
        ds = summary["_date"]
        signals = _extract_signals(summary)
        day_filled = 0
        day_rejected = 0
        day_pnl = 0.0

        for sig in signals:
            side = OrderSide.BUY if sig["direction"] == "LONG" else OrderSide.SELL
            pos_intent = PositionIntent.OPEN_POSITION
            coid = svc.make_client_order_id(sig["region"], sig["symbol"], sig["direction"].lower(), ds)

            try:
                intent = OrderIntent(
                    symbol=sig["symbol"],
                    side=side,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=coid,
                    created_at=datetime.now(timezone.utc),
                    notional=500.0,
                    position_intent=pos_intent,
                    rationale=f"Backtest: {sig['region']} {sig['direction']}",
                    metadata={"price": sig["price"]},
                )
                result = svc.submit(intent)
            except Exception:
                continue

            if result.status == OrderStatus.FILLED:
                day_filled += 1
                qty = result.filled_qty
                price = result.filled_avg_price or sig["price"]

                if side == OrderSide.BUY:
                    cash -= qty * price
                    if sig["symbol"] in positions:
                        existing = positions[sig["symbol"]]
                        new_qty = existing["qty"] + qty
                        new_avg = (existing["avg_price"] * existing["qty"] + price * qty) / new_qty
                        positions[sig["symbol"]] = {"qty": new_qty, "avg_price": new_avg, "direction": "long"}
                    else:
                        positions[sig["symbol"]] = {"qty": qty, "avg_price": price, "direction": "long"}
                else:
                    if sig["symbol"] in positions:
                        pos = positions[sig["symbol"]]
                        realized = qty * (price - pos["avg_price"])
                        day_pnl += realized
                        cash += qty * price
                        pos["qty"] -= qty
                        if pos["qty"] <= 0:
                            del positions[sig["symbol"]]
            else:
                day_rejected += 1

        unrealized = sum(
            p["qty"] * (PRICES.get(s, 0) - p["avg_price"])
            for s, p in positions.items()
        )
        portfolio_value = cash + sum(p["qty"] * PRICES.get(s, 0) for s, p in positions.items())

        total_pnl += day_pnl
        total_filled += day_filled
        total_rejected += day_rejected

        daily_results.append({
            "date": ds,
            "signals": len(signals),
            "filled": day_filled,
            "rejected": day_rejected,
            "realized_pnl": round(day_pnl, 2),
            "portfolio_value": round(portfolio_value, 2),
            "open_positions": len(positions),
            "unrealized_pnl": round(unrealized, 2),
        })

    final_value = cash + sum(p["qty"] * PRICES.get(s, 0) for s, p in positions.items())
    total_return = (final_value - initial_capital) / initial_capital * 100

    report = {
        "period": f"{start} to {end}",
        "days_with_data": len(summaries),
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "total_realized_pnl": round(total_pnl, 2),
        "total_filled": total_filled,
        "total_rejected": total_rejected,
        "open_positions": len(positions),
        "daily_results": daily_results,
    }

    print(f"\n{'='*60}")
    print(f"EXECUTION BACKTEST: {start} → {end}")
    print(f"{'='*60}")
    print(f"  Days:          {report['days_with_data']}")
    print(f"  Capital:       ${initial_capital:,.0f} → ${final_value:,.2f}")
    print(f"  Return:        {total_return:+.2f}%")
    print(f"  Realized PnL:  ${total_pnl:+,.2f}")
    print(f"  Filled:        {total_filled}")
    print(f"  Rejected:      {total_rejected}")
    print(f"  Open Pos:      {len(positions)}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--last", type=int, help="Process last N days")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    if args.last:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=args.last)).strftime("%Y-%m-%d")
    else:
        start = args.start or "2026-03-01"
        end = args.end or "2026-03-24"

    result = run_backtest(start, end, output_base=args.output)
    if "error" not in result:
        out_path = Path("outputs/execution/backtest_result.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Full results saved to {out_path}")
