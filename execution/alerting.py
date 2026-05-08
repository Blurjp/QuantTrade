"""
Execution alerting — sends notifications for order events.

Hooks into the existing NotificationManager Discord channel.
Callers: ExecutionService (after submit), Reconciler (after run).

Alert types:
- ORDER_FILLED:    Order successfully filled
- ORDER_REJECTED:  Order rejected by risk gate or broker
- RECONCILER_ALERT: Stranded orders or drift detected
- DAILY_SUMMARY:   End-of-day execution summary
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ALERT_HISTORY_PATH = Path("outputs/execution/alert_history.json")


def _load_alert_history() -> List[Dict]:
    if ALERT_HISTORY_PATH.exists():
        try:
            return json.loads(ALERT_HISTORY_PATH.read_text())
        except Exception:
            pass
    return []


def _save_alert_history(history: List[Dict]):
    ALERT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_HISTORY_PATH.write_text(json.dumps(history[-200:], indent=2, default=str))


def _send_discord(embed: Dict) -> bool:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("EXECUTION_DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.debug("No Discord webhook configured for execution alerts")
        return False
    try:
        import requests
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        return resp.status_code == 204
    except Exception as e:
        logger.error("Execution alert Discord failed: %s", e)
        return False


def _record_alert(alert_type: str, details: Dict):
    history = _load_alert_history()
    history.append({
        "type": alert_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    })
    _save_alert_history(history)


def alert_order_filled(
    symbol: str,
    side: str,
    filled_qty: float,
    filled_price: float,
    region_id: str = "",
    coid: str = "",
):
    embed = {
        "title": "Order Filled",
        "color": 0x00FF00,
        "fields": [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Side", "value": side.upper(), "inline": True},
            {"name": "Qty", "value": f"{filled_qty:.2f}", "inline": True},
            {"name": "Price", "value": f"${filled_price:.2f}", "inline": True},
            {"name": "Notional", "value": f"${filled_qty * filled_price:,.0f}", "inline": True},
            {"name": "Region", "value": region_id or "N/A", "inline": True},
        ],
        "footer": {"text": f"coid: {coid}"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _send_discord(embed)
    _record_alert("ORDER_FILLED", {
        "symbol": symbol, "side": side, "qty": filled_qty,
        "price": filled_price, "coid": coid,
    })


def alert_order_rejected(
    symbol: str,
    reason: str,
    region_id: str = "",
    coid: str = "",
):
    embed = {
        "title": "Order Rejected",
        "color": 0xFF0000,
        "fields": [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Reason", "value": reason, "inline": True},
            {"name": "Region", "value": region_id or "N/A", "inline": True},
        ],
        "footer": {"text": f"coid: {coid}"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _send_discord(embed)
    _record_alert("ORDER_REJECTED", {"symbol": symbol, "reason": reason, "coid": coid})


def alert_reconciler(
    stranded_found: int,
    stranded_cancelled: int,
    stranded_resubmitted: int,
    drift_found: int,
    errors: List[str],
):
    if not errors and stranded_found == 0 and drift_found == 0:
        return

    color = 0xFFAA00 if not errors else 0xFF0000
    fields = [
        {"name": "Stranded", "value": str(stranded_found), "inline": True},
        {"name": "Cancelled", "value": str(stranded_cancelled), "inline": True},
        {"name": "Resubmitted", "value": str(stranded_resubmitted), "inline": True},
        {"name": "Drift", "value": str(drift_found), "inline": True},
    ]
    if errors:
        fields.append({"name": "Errors", "value": "; ".join(errors[:3]), "inline": False})

    embed = {
        "title": "Reconciler Alert",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _send_discord(embed)
    _record_alert("RECONCILER_ALERT", {
        "stranded": stranded_found, "drift": drift_found, "errors": errors,
    })


def alert_daily_summary(
    filled_count: int,
    rejected_count: int,
    total_notional: float,
    daily_cap: float,
    open_positions: int,
):
    embed = {
        "title": f"Daily Execution Summary — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "color": 0x3498DB,
        "fields": [
            {"name": "Filled", "value": str(filled_count), "inline": True},
            {"name": "Rejected", "value": str(rejected_count), "inline": True},
            {"name": "Notional Used", "value": f"${total_notional:,.0f}", "inline": True},
            {"name": "Daily Cap", "value": f"${daily_cap:,.0f}", "inline": True},
            {"name": "Open Positions", "value": str(open_positions), "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _send_discord(embed)
    _record_alert("DAILY_SUMMARY", {
        "filled": filled_count, "rejected": rejected_count,
        "notional": total_notional, "positions": open_positions,
    })
