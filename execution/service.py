"""
Execution service — the only allowed order entry point.

All auto-trade code must call ExecutionService.submit(), never
MultiAssetPortfolio.open_position() directly. This service:
1. Runs the risk gate (Phase 1 will expand this).
2. Persists the intent to the SQLite ledger.
3. Submits to the configured broker client.
4. Updates the ledger with the result.
5. Returns the OrderResult to the caller.

CRITICAL routing rule: when EXECUTION_MODE is 'shadow' or 'paper',
the BROKER env var is ignored and ShadowBrokerClient is always used.
This prevents accidental live order submission.

Live mode requires TWO switches: EXECUTION_MODE=live AND
LIVE_TRADING_ENABLED=true. A misconfigured live mode (missing broker,
missing enable flag) fails closed — it never silently falls back to shadow.

Crash safety: risk check runs before ledger insert. If the process crashes
after insert but before broker submit, the intent is stranded as 'pending'
in the ledger. Retry with the same client_order_id will be rejected as
duplicate. This is intentional — it prevents double-submission. A future
reconciliation tool can scan for 'pending' orders with no submitted_at
and either resume or cancel them.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
    RiskDecision,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class ExecutionService:
    _account_cache_ttl_seconds = 60

    def __init__(
        self,
        ledger_path: str = "outputs/execution/orders.sqlite",
        execution_mode: Optional[str] = None,
        halt_trading_path: Optional[str] = None,
    ):
        self.execution_mode = execution_mode or os.getenv("EXECUTION_MODE", "shadow")
        self.halt_trading_path = Path(
            halt_trading_path or os.getenv("HALT_TRADING_PATH", "HALT_TRADING")
        )
        self.ledger = OrderLedger(db_path=ledger_path)
        self._account_cache: Optional[tuple] = None

        if self.execution_mode == "live":
            broker_name = os.getenv("BROKER", "")
            if broker_name == "alpaca":
                from execution.brokers.alpaca import AlpacaBrokerClient
                self.broker = AlpacaBrokerClient()
            else:
                raise RuntimeError(
                    f"EXECUTION_MODE=live but BROKER='{broker_name}'. "
                    f"A live broker adapter is required. Refusing to fall back "
                    f"to shadow — that would create phantom positions in "
                    f"portfolio.open_position(). Set BROKER=alpaca or switch "
                    f"to EXECUTION_MODE=shadow."
                )
        else:
            self.broker = ShadowBrokerClient()

        logger.info(
            "ExecutionService initialized: mode=%s broker=%s",
            self.execution_mode,
            type(self.broker).__name__,
        )

    def submit(self, intent: OrderIntent) -> OrderResult:
        risk = self._check_risk(intent)

        self.ledger.insert_order(intent, broker=type(self.broker).__name__)
        self.ledger.insert_risk_decision(
            intent.client_order_id,
            approved=risk.approved,
            reason=risk.reason,
            details=risk.details,
        )

        if not risk.approved:
            self.ledger.update_order_status(
                intent.client_order_id, "rejected",
                filled_at=datetime.now(timezone.utc).isoformat(),
            )
            return OrderResult(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=risk.reason,
            )

        result = self.broker.submit_order(intent)

        self.ledger.update_submitted_at(intent.client_order_id)

        self.ledger.update_order_status(
            intent.client_order_id,
            result.status.value,
            broker_order_id=result.broker_order_id,
            filled_qty=result.filled_qty,
            filled_avg_price=result.filled_avg_price,
            filled_at=(
                result.filled_at.isoformat() if result.filled_at else None
            ),
        )

        if result.status == OrderStatus.FILLED and result.filled_avg_price:
            from execution.models import FillEvent
            fill = FillEvent(
                broker_order_id=result.broker_order_id or "",
                fill_id=f"ledger-{intent.client_order_id}",
                symbol=intent.symbol,
                side=intent.side,
                quantity=result.filled_qty,
                price=result.filled_avg_price,
                timestamp=result.filled_at or datetime.now(timezone.utc),
            )
            self.ledger.insert_fill(fill)

        return result

    def _check_risk(self, intent: OrderIntent) -> RiskDecision:
        if self.halt_trading_path.exists():
            return RiskDecision(
                approved=False,
                reason="halt_trading",
                details={"path": str(self.halt_trading_path)},
            )

        if self.execution_mode == "live":
            if os.getenv("LIVE_TRADING_ENABLED") != "true":
                return RiskDecision(
                    approved=False,
                    reason="live_trading_not_enabled",
                    details={
                        "hint": "Set LIVE_TRADING_ENABLED=true to allow live orders"
                    },
                )

        if self.execution_mode not in ("shadow", "paper", "live"):
            return RiskDecision(
                approved=False,
                reason="invalid_execution_mode",
                details={"mode": self.execution_mode},
            )

        if self.ledger.has_client_order_id(intent.client_order_id):
            existing = self.ledger.get_order(intent.client_order_id)
            existing_status = existing["status"] if existing else "unknown"
            return RiskDecision(
                approved=False,
                reason="duplicate_client_order_id",
                details={"existing_status": existing_status},
            )

        max_order_notional = float(os.getenv("MAX_ORDER_NOTIONAL", "5000"))
        notional = intent.notional_value()
        if notional > max_order_notional:
            return RiskDecision(
                approved=False,
                reason="max_notional_exceeded",
                details={"limit": max_order_notional, "requested": notional},
            )

        ttl_minutes = int(os.getenv("ORDER_TTL_MINUTES", "120"))
        age = (datetime.now(timezone.utc) - intent.created_at).total_seconds() / 60
        if age > ttl_minutes:
            return RiskDecision(
                approved=False,
                reason="order_expired",
                details={"ttl_minutes": ttl_minutes, "age_minutes": round(age, 1)},
            )

        daily_cap = float(os.getenv("MAX_DAILY_NOTIONAL", "15000"))
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_used = self.ledger.get_daily_notional(today)
        if daily_used + notional > daily_cap:
            return RiskDecision(
                approved=False,
                reason="daily_notional_cap_exceeded",
                details={
                    "cap": daily_cap,
                    "used": daily_used,
                    "requested": notional,
                },
            )

        if intent.position_intent == PositionIntent.OPEN_POSITION and intent.side == OrderSide.SELL:
            if os.getenv("ALLOW_SHORT_SELLING") != "true":
                return RiskDecision(
                    approved=False,
                    reason="short_selling_not_allowed",
                    details={
                        "hint": "Set ALLOW_SHORT_SELLING=true to permit opening short positions"
                    },
                )

        if self.execution_mode == "live":
            try:
                if not self.broker.is_market_open():
                    return RiskDecision(
                        approved=False,
                        reason="market_closed",
                        details={"hint": "Live orders only accepted during market hours"},
                    )
            except Exception as e:
                logger.warning("Market hours check failed, rejecting: %s", e)
                return RiskDecision(
                    approved=False,
                    reason="market_status_unknown",
                    details={"error": str(e)[:200]},
                )

        max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.10"))
        account = self._get_cached_account()
        if account and account.equity > 0:
            try:
                positions = self.broker.get_positions()
            except Exception as e:
                logger.warning("Failed to fetch positions for concentration check: %s", e)
                positions = []
            for pos in positions:
                if pos.symbol == intent.symbol:
                    concentration = pos.market_value / account.equity
                    if concentration >= max_position_pct:
                        return RiskDecision(
                            approved=False,
                            reason="position_concentration_exceeded",
                            details={
                                "symbol": intent.symbol,
                                "current_pct": round(concentration, 4),
                                "max_pct": max_position_pct,
                            },
                        )

        return RiskDecision(approved=True, reason="all_checks_passed")

    def _get_cached_account(self):
        if not hasattr(self, "_account_cache"):
            self._account_cache = None
        now = datetime.now(timezone.utc)
        if self._account_cache is not None:
            cached_account, cached_at = self._account_cache
            age = (now - cached_at).total_seconds()
            if age < self._account_cache_ttl_seconds:
                return cached_account
        try:
            account = self.broker.get_account()
            if account is not None:
                self._account_cache = (account, now)
            return account
        except Exception as e:
            logger.warning("Failed to fetch account: %s", e)
            if self._account_cache is not None:
                return self._account_cache[0]
            return None

    @staticmethod
    def make_client_order_id(
        prefix: str, symbol: str, direction: str, date: str
    ) -> str:
        raw = f"{prefix}-{symbol}-{direction}-{date}"
        short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
        return f"qt-{prefix}-{symbol}-{direction}-{short_hash}"


def create_execution_service(**kwargs) -> ExecutionService:
    return ExecutionService(**kwargs)
