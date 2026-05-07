# Live Trading Migration Plan

This plan upgrades the current paper-trading flow to real order submission while keeping signal generation, risk control, execution, and reconciliation separate. The migration must be shadow-first: no broker live order path should exist until all auto-trade entry points are routed through one execution service and pre-submit risk gate.

## Current State

The current system has useful building blocks, but execution is not centralized:

- `execution/trade_mapper.py` maps `ResearchSignal` objects to `TradeCandidate` objects.
- `execution/portfolio_rules.py` applies candidate-level portfolio constraints.
- `paper_trading/multi_asset_portfolio.py` is the paper ledger and assumes immediate ideal fills.
- `scheduler_service.py` still reads signal JSON files directly and calls `MultiAssetPortfolio.open_position(...)` in both Bayesian and fallback paths.

Current production bypasses (the critical migration targets):

- **Bayesian path**: `scheduler_service.py:905` — auto-trade loop calls `portfolio.open_position(...)` after Bayesian fusion signal.
- **Fallback path**: `scheduler_service.py:951` — simple signal loop calls `portfolio.open_position(...)` when Bayesian is unavailable.

The most important migration constraint is therefore not the broker adapter itself. It is first closing every bypass around the future risk gate.

## Target Architecture

```
signals
  -> TradeMapper / signal decision logic
  -> PortfolioRules
  -> OrderIntent builder
  -> RiskGate
  -> ExecutionService
  -> BrokerClient
  -> Order/Fills ledger
  -> Reconciler
```

### 1. Execution Models

Add `execution/models.py` with broker-neutral runtime objects:

- `OrderIntent`: desired action before broker submission. Required fields: symbol, side, quantity, order type, time-in-force, price constraints, account context, stable `client_order_id`, and `created_at` (UTC timestamp used by the TTL check in `RiskGate`).
- `OrderResult`: broker response or shadow result.
- `BrokerPosition`: normalized broker position.
- `BrokerAccount`: normalized account/cash/equity/margin snapshot.
- `FillEvent`: normalized partial or complete fill.

`TradeCandidate` should not be submitted directly to a broker. It is a research/execution candidate with `size_pct`; live trading needs quantity, side, order type, time-in-force, price constraints, account context, and a stable `client_order_id`.

### 2. Execution Service

Add `execution/service.py` as the only allowed order entry point.

Responsibilities:

- Convert approved candidates or existing Bayesian decisions into `OrderIntent`.
- Attach deterministic `client_order_id` values for idempotency.
- Run `RiskGate` before any broker client call.
- Submit to the configured broker client.
- Persist order attempts, broker responses, and shadow results.
- Return explicit accepted/rejected/skipped outcomes to callers.

All existing auto-trade code should call this service. Direct calls to `MultiAssetPortfolio.open_position(...)` should remain only inside paper/shadow broker implementations or tests.

### 3. Broker Adapter Layer

Create `execution/brokers/`:

- `base.py`: `BrokerClient` protocol.
- `shadow.py`: logs and simulates accepted orders without live broker calls.
- `alpaca.py`: stocks/ETFs only in phase 2.
- `ibkr.py`: stocks/ETFs/futures in phase 3.

Base interface:

```python
class BrokerClient(Protocol):
    def submit_order(self, intent: OrderIntent) -> OrderResult: ...
    def cancel_order(self, broker_order_id: str) -> OrderResult: ...
    def get_open_orders(self) -> list[OrderResult]: ...
    def get_positions(self) -> list[BrokerPosition]: ...
    def get_account(self) -> BrokerAccount: ...
```

Broker routing rules:

- Alpaca is suitable for the first live phase for stocks and ETFs because the API is simple and supports paper/live environments.
- Alpaca should not be used for commodity futures in this plan. Its public Trading API documentation focuses on stocks and crypto, with futures/FX described separately as roadmap-style coverage.
- IBKR is the preferred broker for the full instrument universe because its API supports stocks, ETFs, options, futures, currencies, bonds, and funds, subject to account permissions and market-data subscriptions.

References:

- Alpaca Trading API: https://docs.alpaca.markets/v1.3/docs/trading-api
- Alpaca docs overview: https://docs.alpaca.markets/docs
- IBKR API solutions: https://brokerage.ibkr.com/en/trading/ib-api.php

### 4. Risk Gate

Add `execution/risk_gate.py` as a hard pre-submit control. This is separate from both `PortfolioRules` and the existing risk modules.

Relationship to existing risk code:

- `risk/risk_manager.py` — portfolio-level exposure, correlation, sector, and daily-open gating. Coexists; `RiskGate` calls into it for position concentration checks.
- `execution/risk.py` — stop-loss calculation, position sizing helpers, risk metrics, and signal-level risk utilities. Coexists; `RiskGate` does not replace it.
- `execution/risk_gate.py` — new. Pre-submit gate only. Wraps the above as needed but adds broker-facing hard stops (kill switch, mode checks, daily loss limits).

`PortfolioRules` answers: "Should this candidate exist, and what target allocation is reasonable?"

`RiskGate` answers: "Can this exact order be sent to a broker right now?"

Minimum hard checks:

- `EXECUTION_MODE` must be one of `paper`, `shadow`, `live`.
- Live mode must require an explicit `LIVE_TRADING_ENABLED=true`.
- Reject all orders when a `HALT_TRADING` file exists.
- Reject unsupported asset classes per broker.
- Reject missing or stale prices.
- Reject missing `client_order_id`.
- Enforce max notional per order.
- Enforce max daily notional.
- Enforce daily loss limit using broker account plus local fill ledger.
- Enforce max position concentration after hypothetical fill.
- Enforce max open orders per symbol.
- Enforce shortability or broker permission checks where available.
- Reject duplicate `client_order_id` submissions unless the stored outcome is known and safe.
- Reject expired order intents where `now - intent.created_at > ORDER_TTL_MINUTES`.

Risk gate outputs should be structured:

```json
{
  "approved": false,
  "reason": "max_notional_exceeded",
  "details": {"limit": 5000, "requested": 8200}
}
```

### 5. Order Ledger

Add a small persistent order ledger before live trading. SQLite is preferable to JSON for idempotency and queryability.

Minimum tables:

- `orders`: internal ID, `client_order_id`, symbol, side, quantity, notional, status, broker, broker order ID, timestamps.
- `fills`: broker order ID, fill ID, symbol, side, quantity, price, timestamp.
- `risk_decisions`: `client_order_id`, approved/rejected, reason, details.
- `reconciliation_runs`: timestamp, drift summary, alert status.

This ledger is required before live order submission because hourly cron or service restarts can otherwise duplicate orders after network ambiguity.

### 6. Reconciler

Add `execution/reconciler.py`.

It should compare:

- Broker open orders vs local `orders`.
- Broker fills vs local `fills`.
- Broker positions vs local shadow/reporting positions.
- Broker account equity/cash/margin vs local assumptions.

Reconciliation states:

- `ok`: within tolerance.
- `warning`: small drift, no trading halt.
- `critical`: unknown broker position, missing fill, oversized position, or account drawdown breach.

On `critical`, the reconciler should create or respect `HALT_TRADING` and block new submissions until manually cleared.

## Migration Phases

### Phase 0: Centralize Execution

Goal: remove bypasses before adding any live broker.

Current production bypasses to replace:

- Bayesian path: `scheduler_service.py` auto-trade loop calls `portfolio.open_position(...)`.
- Fallback path: `scheduler_service.py` simple signal loop calls `portfolio.open_position(...)`.

Tasks:

- Add `OrderIntent` and execution models.
- Add `ExecutionService` with a `ShadowBrokerClient`.
- Replace direct auto-trade calls in `scheduler_service.py` with `ExecutionService`.
- Keep `MultiAssetPortfolio` as paper/reporting state only.
- Add tests proving scheduler auto-trade paths cannot call `open_position` directly.

Acceptance criteria:

- All automated order attempts flow through `ExecutionService`.
- `rg "open_position\\("` shows no production auto-trade bypass.
- Shadow order attempts are persisted with stable `client_order_id`.

### Phase 1: Risk Gate and Shadow Mode

Goal: run production signals through the live-shaped path without placing live orders.

Tasks:

- Implement `RiskGate`.
- Add `EXECUTION_MODE=shadow`.
- Add `HALT_TRADING` kill switch support.
- Add SQLite order/fill/risk ledger.
- Add structured logs and daily summary of approved/rejected intents.

Acceptance criteria:

- Shadow mode runs for at least two weeks.
- No duplicate client order IDs for repeated scheduler runs.
- Every rejected order includes a machine-readable reason.
- Drift between shadow positions and paper/reporting positions is below the configured tolerance.

### Phase 2: Alpaca for Stocks and ETFs

Goal: enable limited live trading for the simplest instrument set.

Scope:

- Stocks and ETFs only.
- No commodity futures.
- Small notional caps.
- Live mode requires both `EXECUTION_MODE=live` and `LIVE_TRADING_ENABLED=true`.

Tasks:

- Implement `AlpacaBrokerClient`.
- Support paper and live Alpaca endpoints through configuration.
- Normalize Alpaca account, order, position, and fill responses.
- Enforce broker asset routing: futures rejected before adapter submission.

Acceptance criteria:

- Alpaca paper endpoint passes end-to-end submit/cancel/reconcile tests.
- Live account read-only reconciliation passes before first live order.
- First live order is manually approved and below a small notional cap.

### Phase 3: IBKR for Full Universe

Goal: support futures and broader instrument coverage.

Tasks:

- Implement `IBKRBrokerClient`.
- Add contract resolution for futures and ETFs.
- Add account permission checks where possible.
- Add market-data subscription checks or stale-price blocking.
- Add stricter trading-session checks for futures.

Acceptance criteria:

- IBKR paper/simulated account passes submit/cancel/reconcile tests.
- Futures symbols require explicit config mapping, not free-form ticker strings.
- Reconciler can detect partial fills and position drift.

## Configuration

Recommended environment variables:

```bash
EXECUTION_MODE=shadow
LIVE_TRADING_ENABLED=false
BROKER=shadow
ORDER_LEDGER_PATH=outputs/execution/orders.sqlite
HALT_TRADING_PATH=HALT_TRADING
MAX_ORDER_NOTIONAL=1000
MAX_DAILY_NOTIONAL=3000
MAX_DAILY_LOSS=500
MAX_POSITION_PCT=0.05   # Conservative live-launch cap; existing RiskManager defaults to 10%
ORDER_TTL_MINUTES=120
```

Live mode should fail closed if any required setting is missing.

## Testing Requirements

Minimum test coverage before live trading:

- `RiskGate` approves and rejects expected order intents.
- Kill switch blocks every order.
- Duplicate `client_order_id` is idempotent.
- Unsupported asset class is rejected for Alpaca.
- Scheduler auto-trade path calls `ExecutionService`.
- Reconciler detects broker/local position drift.
- Shadow broker writes deterministic order records.
- Expired order intent (beyond `ORDER_TTL_MINUTES`) is rejected by `RiskGate`.

## Operational Runbook

Before enabling live:

1. Confirm `EXECUTION_MODE=shadow` has run cleanly for the observation window.
2. Review daily rejected-order reasons.
3. Confirm no direct production calls to `MultiAssetPortfolio.open_position(...)`.
4. Run broker read-only reconciliation.
5. Set notional caps to intentionally small values.
6. Enable live with `LIVE_TRADING_ENABLED=true`.
7. Submit one manually approved order.
8. Confirm broker order, fill, ledger, and reporting state all reconcile.

Emergency stop:

```bash
touch HALT_TRADING
```

Resume only after:

- Open orders are reviewed.
- Broker positions match intended positions.
- `HALT_TRADING` is manually removed.
- Reconciler returns `ok`.

## Non-Goals for the First Live Release

- Fully automated futures trading.
- Intraday high-frequency execution.
- Smart order routing.
- Portfolio optimization beyond existing candidate sizing.
- Replacing the existing paper-reporting UI.
- Automatic recovery from critical reconciliation drift.
