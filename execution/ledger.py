"""
SQLite order ledger for persistent execution state.

Stores orders, fills, risk decisions, and reconciliation runs.
Required for idempotency (preventing duplicate orders on cron restarts)
and audit trail (every order attempt is recorded).
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from execution.models import (
    FillEvent,
    OrderIntent,
    OrderResult,
    RiskDecision,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    position_intent TEXT NOT NULL DEFAULT 'open_position',
    order_class TEXT NOT NULL DEFAULT 'simple',
    quantity REAL,
    notional REAL,
    order_type TEXT NOT NULL,
    time_in_force TEXT NOT NULL,
    limit_price REAL,
    stop_price REAL,
    take_profit_limit REAL,
    stop_loss_stop REAL,
    stop_loss_limit REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    broker TEXT,
    broker_order_id TEXT,
    parent_broker_order_id TEXT,
    rationale TEXT,
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    filled_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_coid ON orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_parent ON orders(parent_broker_order_id);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id TEXT NOT NULL,
    fill_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fills_boid ON fills(broker_order_id);

CREATE TABLE IF NOT EXISTS risk_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT,
    details TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_risk_coid ON risk_decisions(client_order_id);

CREATE TABLE IF NOT EXISTS reconciliation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    status TEXT NOT NULL,
    orders_drift INTEGER DEFAULT 0,
    positions_drift INTEGER DEFAULT 0,
    fills_missing INTEGER DEFAULT 0,
    alert TEXT,
    details TEXT
);
"""


class OrderLedger:
    def __init__(self, db_path: str = "outputs/execution/orders.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)

    def insert_order(self, intent: OrderIntent, broker: str = "shadow") -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT OR IGNORE INTO orders
            (client_order_id, symbol, side, position_intent, order_class,
             quantity, notional, order_type, time_in_force, limit_price,
             stop_price, take_profit_limit, stop_loss_stop, stop_loss_limit,
             status, broker, rationale, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
            (
                intent.client_order_id,
                intent.symbol,
                intent.side.value,
                intent.position_intent.value,
                intent.order_class.value,
                intent.quantity,
                intent.notional,
                intent.order_type.value,
                intent.time_in_force.value,
                intent.limit_price,
                intent.stop_price,
                intent.take_profit_limit,
                intent.stop_loss_stop,
                intent.stop_loss_limit,
                broker,
                intent.rationale[:500],
                intent.created_at.isoformat(),
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_order_status(
        self,
        client_order_id: str,
        status: str,
        broker_order_id: Optional[str] = None,
        filled_qty: Optional[float] = None,
        filled_avg_price: Optional[float] = None,
        filled_at: Optional[str] = None,
    ):
        parts = ["status = ?", "updated_at = ?"]
        values = [status, datetime.now(timezone.utc).isoformat()]

        if broker_order_id is not None:
            parts.append("broker_order_id = ?")
            values.append(broker_order_id)
        if filled_qty is not None:
            parts.append("quantity = ?")
            values.append(filled_qty)
        if filled_avg_price is not None:
            parts.append("limit_price = ?")
            values.append(filled_avg_price)
        if filled_at is not None:
            parts.append("filled_at = ?")
            values.append(filled_at)

        values.append(client_order_id)
        self._conn.execute(
            f"UPDATE orders SET {', '.join(parts)} WHERE client_order_id = ?",
            values,
        )
        self._conn.commit()

    def update_submitted_at(self, client_order_id: str):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE orders SET submitted_at = ?, updated_at = ? WHERE client_order_id = ?",
            (now, now, client_order_id),
        )
        self._conn.commit()

    def get_order(self, client_order_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_pending_orders(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE status IN ('pending', 'accepted')"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_notional(self, date: str) -> float:
        row = self._conn.execute(
            """SELECT COALESCE(SUM(
                CASE WHEN notional IS NOT NULL THEN notional
                     WHEN quantity IS NOT NULL AND limit_price IS NOT NULL THEN quantity * limit_price
                     ELSE 0 END
            ), 0) as total FROM orders
            WHERE created_at >= ? AND created_at < ? AND status != 'rejected'""",
            (f"{date}T00:00:00", f"{date}T23:59:59"),
        ).fetchone()
        return row["total"] if row else 0.0

    def has_client_order_id(self, client_order_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return row is not None

    def insert_fill(self, fill: FillEvent):
        self._conn.execute(
            """INSERT OR IGNORE INTO fills
            (broker_order_id, fill_id, symbol, side, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                fill.broker_order_id,
                fill.fill_id,
                fill.symbol,
                fill.side.value,
                fill.quantity,
                fill.price,
                fill.timestamp.isoformat(),
            ),
        )
        self._conn.commit()

    def insert_risk_decision(
        self,
        client_order_id: str,
        approved: bool,
        reason: str,
        details: Optional[dict] = None,
    ):
        self._conn.execute(
            """INSERT INTO risk_decisions
            (client_order_id, approved, reason, details, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                client_order_id,
                1 if approved else 0,
                reason,
                json.dumps(details or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def insert_reconciliation(
        self,
        status: str,
        orders_drift: int = 0,
        positions_drift: int = 0,
        fills_missing: int = 0,
        alert: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        self._conn.execute(
            """INSERT INTO reconciliation_runs
            (run_at, status, orders_drift, positions_drift, fills_missing, alert, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                status,
                orders_drift,
                positions_drift,
                fills_missing,
                alert,
                json.dumps(details or {}),
            ),
        )
        self._conn.commit()

    def get_recent_fills(self, limit: int = 100) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM fills ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if hasattr(self, "_conn") and self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
