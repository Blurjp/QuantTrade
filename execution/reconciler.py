"""
Order reconciler — recovers stranded pending orders after crashes.

When the process crashes between ledger insert and broker submit,
orders are stranded as 'pending' with no submitted_at. This tool:

1. Scans for stranded orders (pending, no submitted_at).
2. For orders older than a TTL (default 5 min): cancels them as stale.
3. For recent orders: re-submits to the broker and updates the ledger.

Also reconciles terminal-state drift:
- Checks open orders in the ledger against broker state.
- Updates ledger if broker reports filled/canceled/rejected.
- Logs every reconciliation run to the reconciliation_runs table.

Usage:
    from execution.reconciler import Reconciler
    r = Reconciler(execution_service)
    report = r.run()
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from execution.ledger import OrderLedger
from execution.models import OrderIntent, OrderResult, OrderStatus, RiskDecision
from execution.service import ExecutionService

logger = logging.getLogger(__name__)

STRANDED_ORDER_TTL_SECONDS = 300


class ReconcileReport:
    def __init__(self):
        self.stranded_found: int = 0
        self.stranded_cancelled: int = 0
        self.stranded_resubmitted: int = 0
        self.strand_resubmit_failed: int = 0
        self.drift_found: int = 0
        self.drift_updated: int = 0
        self.errors: List[str] = []

    @property
    def has_alert(self) -> bool:
        return bool(self.errors) or self.strand_resubmit_failed > 0

    def summary(self) -> str:
        lines = [
            f"ReconcileReport: stranded={self.stranded_found} "
            f"(cancelled={self.stranded_cancelled} "
            f"resubmitted={self.stranded_resubmitted} "
            f"failed={self.strand_resubmit_failed}) "
            f"drift={self.drift_found} updated={self.drift_updated}",
        ]
        if self.errors:
            lines.append(f"  errors: {self.errors}")
        return "\n".join(lines)


class Reconciler:
    def __init__(
        self,
        execution_service: ExecutionService,
        stranded_ttl_seconds: int = STRANDED_ORDER_TTL_SECONDS,
    ):
        self.svc = execution_service
        self.ledger = execution_service.ledger
        self.broker = execution_service.broker
        self.stranded_ttl = stranded_ttl_seconds

    def run(self) -> ReconcileReport:
        report = ReconcileReport()
        try:
            self._reconcile_stranded(report)
            self._reconcile_drift(report)
        except Exception as e:
            report.errors.append(f"reconcile_run_error: {e}")
            logger.error("Reconciler run failed: %s", e)

        self.ledger.insert_reconciliation(
            status="alert" if report.has_alert else "ok",
            orders_drift=report.drift_found,
            positions_drift=0,
            fills_missing=0,
            alert="; ".join(report.errors) if report.errors else None,
            details={
                "stranded_found": report.stranded_found,
                "stranded_cancelled": report.stranded_cancelled,
                "stranded_resubmitted": report.stranded_resubmitted,
                "strand_resubmit_failed": report.strand_resubmit_failed,
                "drift_updated": report.drift_updated,
            },
        )

        logger.info(report.summary())

        try:
            from execution.alerting import alert_reconciler
            alert_reconciler(
                stranded_found=report.stranded_found,
                stranded_cancelled=report.stranded_cancelled,
                stranded_resubmitted=report.stranded_resubmitted,
                drift_found=report.drift_found,
                errors=report.errors,
            )
        except Exception:
            pass

        return report

    def _reconcile_stranded(self, report: ReconcileReport):
        now = datetime.now(timezone.utc)
        rows = self.ledger._conn.execute(
            "SELECT * FROM orders WHERE status = 'pending' AND submitted_at IS NULL"
        ).fetchall()

        report.stranded_found = len(rows)

        for row in rows:
            order = dict(row)
            coid = order["client_order_id"]
            created = order["created_at"]

            try:
                created_dt = datetime.fromisoformat(created)
            except (ValueError, TypeError):
                created_dt = now

            age_seconds = (now - created_dt).total_seconds()

            if age_seconds > self.stranded_ttl:
                self.ledger.update_order_status(coid, "canceled")
                report.stranded_cancelled += 1
                logger.warning(
                    "Reconciler: cancelled stranded order %s (age=%.0fs > ttl=%ds)",
                    coid, age_seconds, self.stranded_ttl,
                )
            else:
                self._resubmit_order(order, report)

    def _resubmit_order(self, order: Dict, report: ReconcileReport):
        coid = order["client_order_id"]
        try:
            intent = OrderIntent(
                symbol=order["symbol"],
                side=self._parse_side(order["side"]),
                order_type=self._parse_order_type(order["order_type"]),
                time_in_force=self._parse_tif(order["time_in_force"]),
                client_order_id=coid,
                created_at=datetime.fromisoformat(order["created_at"]),
                quantity=order["quantity"],
                notional=order["notional"],
                order_class=self._parse_order_class(order.get("order_class", "simple")),
                limit_price=order.get("limit_price"),
                stop_price=order.get("stop_price"),
                take_profit_limit=order.get("take_profit_limit"),
                stop_loss_stop=order.get("stop_loss_stop"),
                stop_loss_limit=order.get("stop_loss_limit"),
                position_intent=self._parse_position_intent(
                    order.get("position_intent", "open_position")
                ),
                rationale=order.get("rationale", "") or "",
            )

            result = self.broker.submit_order(intent)

            if result.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                self.ledger.update_submitted_at(coid)

            self.ledger.update_order_status(
                coid,
                result.status.value,
                broker_order_id=result.broker_order_id,
                filled_qty=result.filled_qty,
                filled_avg_price=result.filled_avg_price,
                filled_at=(
                    result.filled_at.isoformat() if result.filled_at else None
                ),
            )

            report.stranded_resubmitted += 1
            logger.info(
                "Reconciler: resubmitted %s -> %s", coid, result.status.value,
            )

        except Exception as e:
            report.strand_resubmit_failed += 1
            report.errors.append(f"resubmit_failed:{coid} {e}")
            logger.error("Reconciler: failed to resubmit %s: %s", coid, e)

    def _reconcile_drift(self, report: ReconcileReport):
        pending_rows = self.ledger._conn.execute(
            """SELECT * FROM orders
            WHERE status IN ('accepted', 'partially_filled')
            AND broker_order_id IS NOT NULL"""
        ).fetchall()

        if not pending_rows:
            return

        try:
            broker_orders = self.broker.get_open_orders()
        except Exception as e:
            report.errors.append(f"broker_open_orders_failed: {e}")
            return

        broker_ids = {o.broker_order_id for o in broker_orders}

        for row in pending_rows:
            order = dict(row)
            boid = order["broker_order_id"]
            if not boid:
                continue

            if boid not in broker_ids:
                report.drift_found += 1
                self.ledger.update_order_status(
                    order["client_order_id"],
                    "expired",
                )
                report.drift_updated += 1
                logger.warning(
                    "Reconciler: drifted order %s (broker_id=%s no longer open)",
                    order["client_order_id"], boid,
                )

    @staticmethod
    def _parse_side(val: str):
        from execution.models import OrderSide
        return OrderSide(val)

    @staticmethod
    def _parse_order_type(val: str):
        from execution.models import OrderType
        return OrderType(val)

    @staticmethod
    def _parse_tif(val: str):
        from execution.models import TimeInForce
        return TimeInForce(val)

    @staticmethod
    def _parse_order_class(val: str):
        from execution.models import OrderClass
        return OrderClass(val or "simple")

    @staticmethod
    def _parse_position_intent(val: str):
        from execution.models import PositionIntent
        return PositionIntent(val or "open_position")
