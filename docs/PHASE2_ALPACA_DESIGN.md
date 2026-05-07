# Phase 2: Alpaca Live Trading — Detailed Design

This document specifies the implementation of Phase 2 from the [Live Trading Migration Plan](LIVE_TRADING_MIGRATION_PLAN.md). Phase 0 (centralize execution) and Phase 1 (risk gate + shadow mode) are prerequisites.

## Scope

- Stocks and ETFs only. No commodity futures.
- Alpaca is the sole live broker for this phase.
- Small notional caps. Conservative position sizes.
- Live mode requires both `EXECUTION_MODE=live` and `LIVE_TRADING_ENABLED=true`.

## Prerequisites

| Prerequisite | Status | Verification |
|---|---|---|
| Phase 0: ExecutionService with ShadowBroker | Must be complete | `rg "open_position\("` shows no production bypass |
| Phase 1: RiskGate + SQLite ledger | Must be complete | Shadow mode runs 2+ weeks clean |
| Alpaca account (paper + live) | Must be provisioned | Paper API key tested |
| Market data subscription | Required for real-time prices | Alpaca free tier covers IEX |

## Architecture

```
scheduler_service.py (auto-trade)
  -> ExecutionService.submit(intent)
       -> RiskGate.check(intent)
            -> risk/risk_manager.py (exposure, sector, daily limits)
            -> execution/risk.py (stop-loss, position sizing)
            -> Staleness check (ORDER_TTL_MINUTES)
            -> Mode check (EXECUTION_MODE, LIVE_TRADING_ENABLED, HALT_TRADING)
       -> AlpacaBrokerClient.submit_order(intent)
            -> Alpaca REST API
       -> SQLite ledger (orders, fills, risk_decisions)
```

## File Structure

```
execution/
├── __init__.py
├── models.py              # NEW: OrderIntent, OrderResult, BrokerPosition, BrokerAccount, FillEvent
├── service.py             # NEW: ExecutionService — single order entry point
├── risk_gate.py           # NEW: Pre-submit hard stops (Phase 1)
├── reconciler.py          # NEW: Broker vs local state comparison
├── trade_mapper.py        # EXISTING: ResearchSignal → TradeCandidate
├── portfolio_rules.py     # EXISTING: Portfolio-level constraints
├── risk.py                # EXISTING: Stop-loss, position sizing helpers
├── ledger.py              # NEW: SQLite order/fill/risk ledger
└── brokers/
    ├── __init__.py
    ├── base.py            # NEW: BrokerClient protocol
    ├── shadow.py          # NEW: Logs orders, simulates fills (Phase 1)
    └── alpaca.py          # NEW: Alpaca REST adapter (this phase)
```

## 1. Execution Models (`execution/models.py`)

All models are broker-neutral. Alpaca-specific mapping happens inside `AlpacaBrokerClient`.

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderClass(str, Enum):
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"


class PositionIntent(str, Enum):
    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    OPG = "opg"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce
    client_order_id: str
    created_at: datetime
    position_intent: PositionIntent = PositionIntent.OPEN_POSITION
    order_class: OrderClass = OrderClass.SIMPLE
    quantity: Optional[float] = None
    notional: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit_limit: Optional[float] = None
    stop_loss_stop: Optional[float] = None
    stop_loss_limit: Optional[float] = None
    asset_class: str = "us_equity"
    rationale: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        has_qty = self.quantity is not None and self.quantity > 0
        has_notional = self.notional is not None and self.notional > 0
        if not has_qty and not has_notional:
            raise ValueError("OrderIntent must have exactly one of quantity or notional > 0")
        if has_qty and has_notional:
            raise ValueError("OrderIntent quantity and notional are mutually exclusive")


@dataclass
class OrderResult:
    client_order_id: str
    status: OrderStatus
    broker_order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    legs: Optional[List['OrderResult']] = None
    raw_response: Optional[dict] = None


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    side: str  # "long" or "short"
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class BrokerAccount:
    equity: float
    cash: float
    buying_power: float
    initial_margin: float
    maintenance_margin: float
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool


@dataclass
class FillEvent:
    broker_order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
```

## 2. Broker Client Protocol (`execution/brokers/base.py`)

```python
from typing import Protocol, List
from execution.models import OrderIntent, OrderResult, BrokerPosition, BrokerAccount


class BrokerClient(Protocol):
    def submit_order(self, intent: OrderIntent) -> OrderResult: ...
    def cancel_order(self, broker_order_id: str) -> OrderResult: ...
    def get_open_orders(self) -> List[OrderResult]: ...
    def get_positions(self) -> List[BrokerPosition]: ...
    def get_account(self) -> BrokerAccount: ...
```

## 3. Alpaca Adapter (`execution/brokers/alpaca.py`)

### 3.1 Configuration

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper
# ALPACA_BASE_URL=https://api.alpaca.markets        # live
```

The adapter reads these from environment variables. Paper vs live is controlled by `ALPACA_BASE_URL`, not by `EXECUTION_MODE`. The `RiskGate` enforces `EXECUTION_MODE=live` + `LIVE_TRADING_ENABLED=true` before the adapter is ever called.

**Critical routing rule**: `ExecutionService` ignores the `BROKER` env var whenever `EXECUTION_MODE` is `shadow` or `paper`. In those modes, `ShadowBrokerClient` is always used regardless of `BROKER` setting. The `BROKER` env var only takes effect when `EXECUTION_MODE=live`. This prevents accidental live order submission during shadow testing.

### 3.2 Symbol Mapping

Current QuantTrade tickers that need mapping:

| QuantTrade Ticker | Alpaca Symbol | Notes |
|---|---|---|
| XLE | XLE | Direct |
| OIH | OIH | Direct |
| FXI | FXI | Direct |
| MCHI | MCHI | Direct |
| ASHR | ASHR | Direct |
| EWG | EWG | Direct |
| WMT, COST, etc. | Same | Direct |
| CORN, SOYB, WEAT | Same | ETF proxies (Teucrium/Teucrium/Teucrium). Subject to Alpaca asset lookup. |
| WTI, Brent | — | **Rejected**. Not tradeable on Alpaca (commodity indexes, not securities). |
| LE=F, ZC=F, ZS=F, CL=F | — | **Rejected**. Futures contracts not supported on Alpaca. |

The adapter maintains a `SUPPORTED_ASSETS` set. Any symbol not in the set is rejected with `reason: "unsupported_symbol"`. The set is populated from a config file or the Alpaca assets API at startup.

### 3.3 Order Flow

```
OrderIntent
  -> Validate symbol is tradeable on Alpaca
  -> Validate quantity vs notional (exactly one must be set)
  -> If notional and asset is not fractionable, convert to qty (floor)
  -> Map OrderSide to Alpaca "side" (buy/sell)
  -> Map OrderType to Alpaca "type" (market/limit/stop/stop_limit)
  -> Map OrderClass to Alpaca "order_class" (simple/bracket/oco/oto)
  -> If bracket: attach take_profit and stop_loss legs from intent fields
  -> POST /v2/orders with client_order_id for idempotency
  -> Parse response (including legs) into OrderResult
```

### 3.4 Idempotency

Alpaca supports `client_order_id` natively. If a duplicate `client_order_id` is submitted, Alpaca returns `409 Conflict` with the original order. The adapter should:

1. Check the SQLite ledger first (fast local check).
2. If not found locally, submit to Alpaca.
3. On `409`, fetch the existing order and return it as `OrderResult`.

### 3.5 Rate Limits

Alpaca rate limits: 200 requests/minute for non-pro accounts. The adapter should:

- Track request timestamps.
- Sleep if approaching the limit.
- Return a structured error on `429 Too Many Requests`.

### 3.6 Error Handling

| Alpaca Response | Adapter Behavior |
|---|---|
| 200 OK | Parse and return `OrderResult(status=ACCEPTED)` |
| 409 Conflict | Fetch existing order, return as `OrderResult` |
| 422 Unprocessable | Return `OrderResult(status=REJECTED, rejection_reason=...)` |
| 429 Too Many Requests | Sleep + retry once, then reject |
| 5xx / timeout | Return `OrderResult(status=PENDING)`, log error, reconciler catches drift |

### 3.7 Market Hours Check

Before submitting, the adapter should verify the market is open using Alpaca's `GET /v2/clock` endpoint. If the market is closed:

- Market orders are rejected (would execute at next open with gap risk).
- Limit orders within 1% of last close are allowed (GTC).
- All other orders are rejected with `reason: "market_closed"`.

## 4. Order Ledger (`execution/ledger.py`)

SQLite database at `ORDER_LEDGER_PATH` (default: `outputs/execution/orders.sqlite`).

### 4.1 Schema

```sql
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

CREATE INDEX IF NOT EXISTS idx_orders_parent ON orders(parent_broker_order_id);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id TEXT NOT NULL,
    fill_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (broker_order_id) REFERENCES orders(broker_order_id)
);

CREATE TABLE IF NOT EXISTS risk_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT,
    details TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (client_order_id) REFERENCES orders(client_order_id)
);

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
```

### 4.2 Ledger Operations

```python
class OrderLedger:
    def insert_order(self, intent: OrderIntent) -> int
    def update_order_status(self, client_order_id: str, status: str, **kwargs)
    def get_order(self, client_order_id: str) -> Optional[dict]
    def get_pending_orders(self) -> List[dict]
    def get_daily_notional(self, date: str) -> float
    def insert_fill(self, fill: FillEvent)
    def insert_risk_decision(self, client_order_id: str, approved: bool, reason: str, details: dict)
    def has_client_order_id(self, client_order_id: str) -> bool
```

## 5. Risk Gate (`execution/risk_gate.py`)

This builds on Phase 1's RiskGate with Alpaca-specific checks.

### Additional checks for Phase 2:

```python
class RiskGate:
    def check(self, intent: OrderIntent, ledger: OrderLedger, broker_positions: List[BrokerPosition]) -> RiskDecision:
        # --- Phase 1 checks (unchanged) ---
        # mode check, halt check, staleness, max notional, daily loss, etc.

        # --- Phase 2 additions ---
        self._check_alpaca_supported_symbol(intent)
        self._check_market_hours(intent)
        self._check_buying_power(intent, account)
        self._check_position_concentration(intent, broker_positions)
        self._check_position_intent(intent, broker_positions)
```

| Check | Source | Reject Reason |
|---|---|---|
| Symbol supported on Alpaca | `SUPPORTED_ASSETS` set | `unsupported_symbol` |
| Market is open | Alpaca clock API | `market_closed` |
| Sufficient buying power | BrokerAccount.buying_power | `insufficient_buying_power` |
| No position duplication | broker_positions + ledger | `duplicate_position` |
| Notional per order | `MAX_ORDER_NOTIONAL` env var | `max_notional_exceeded` |
| Daily notional cap | Ledger `get_daily_notional()` | `max_daily_notional_exceeded` |
| Daily loss limit | Ledger fills + `MAX_DAILY_LOSS` | `daily_loss_limit_exceeded` |
| Order staleness | `ORDER_TTL_MINUTES` | `order_expired` |
| Kill switch active | `HALT_TRADING_PATH` file exists | `halt_trading` |
| Mode mismatch | `EXECUTION_MODE` != `live` or `LIVE_TRADING_ENABLED` != `true` | `execution_mode_not_live` |
| Opening short without permission | `ALLOW_SHORT_SELLING` env var + Alpaca asset metadata | `short_selling_not_enabled` |
| Closing nonexistent position | broker_positions + ledger | `no_position_to_close` |
| Fractional not allowed | Alpaca asset `fractionable` flag | `fractional_not_supported` |

## 6. Reconciler (`execution/reconciler.py`)

Runs after each scheduler cycle (hourly) and on demand.

### 6.1 Checks

1. **Orders**: Compare broker open orders vs ledger pending orders. Flag any broker order not in ledger (unexpected) or ledger order not at broker (stale).

2. **Fills**: Fetch broker fills since last reconciliation. Insert missing fills into ledger. Flag any fill with no matching ledger order.

3. **Positions**: Compare broker positions vs local portfolio positions. Drift tolerance: 1% of notional.

4. **Account**: Compare broker equity/cash vs local assumptions. Flag if equity drawdown exceeds 5% since last check.

### 6.2 Drift Handling

| Drift Type | Tolerance | Action |
|---|---|---|
| Position qty differs by <1% | OK | Log, update local |
| Position qty differs by 1-5% | Warning | Log, update local, alert |
| Position qty differs by >5% | Critical | Create `HALT_TRADING`, alert |
| Unknown broker position | Critical | Create `HALT_TRADING`, alert |
| Missing fill in ledger | Warning | Insert fill, log |
| Account drawdown >5% | Critical | Create `HALT_TRADING`, alert |

## 7. Scheduler Integration

### 7.1 Current State (to be replaced)

Two bypasses in `scheduler_service.py`:

- **Line 905** (Bayesian path): `portfolio.open_position(ticker, direction, price, value, rationale, asset_class)`
- **Line 951** (Fallback path): `portfolio.open_position(ticker, direction, price, value, rationale, asset_class)`

Both call `RiskManager.check_risk()` before opening, but there is no execution service, no order ledger, and no kill switch.

### 7.2 Target State

```
for each trade decision (Bayesian or fallback):
    # Determine side and intent
    if direction == "long":
        side = OrderSide.BUY
        intent_kind = PositionIntent.OPEN_POSITION
    else:
        side = OrderSide.SELL
        # Check if we already hold a long position to close
        if ticker in broker_positions and broker_positions[ticker].side == "long":
            intent_kind = PositionIntent.CLOSE_POSITION
        elif os.getenv("ALLOW_SHORT_SELLING") == "true":
            intent_kind = PositionIntent.OPEN_POSITION
        else:
            logger.warning(f"Short selling not enabled, skipping {ticker}")
            continue

    # Use notional for fractional-friendly submission, or qty for whole shares
    intent = OrderIntent(
        symbol=ticker,
        side=side,
        notional=position_value,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        client_order_id=f"qt-{date}-{ticker}-{direction}-{hash}",
        created_at=datetime.now(datetime.timezone.utc),
        position_intent=intent_kind,
        order_class=OrderClass.BRACKET,
        stop_loss_stop=price * (1 - stop_loss_pct),
        take_profit_limit=price * (1 + take_profit_pct),
        rationale=...,
    )
    result = execution_service.submit(intent)
    if result.status in (ACCEPTED, FILLED):
        trades_made += 1
        logger.info(f"AUTO-TRADE: {ticker} {result.status}")
    else:
        logger.warning(f"Trade rejected: {result.rejection_reason}")
```

### 7.3 Stop-Loss and Take-Profit

Current system manages stop-loss/take-profit in `MultiAssetPortfolio.update_position_prices()`. For live trading:

1. Submit bracket orders via Alpaca using `order_class=OrderClass.BRACKET` with `stop_loss_stop` and `take_profit_limit` fields on `OrderIntent`. Alpaca creates the parent order plus two child legs (stop-loss and take-profit) atomically.
2. The adapter stores the parent `broker_order_id` in the ledger `orders` table and each leg's `broker_order_id` with `parent_broker_order_id` referencing the parent.
3. The reconciler verifies stop/take-profit legs are still open at broker. If a leg fills, the reconciler detects the fill and updates the local portfolio. If the parent fills but legs are missing, the reconciler logs a warning and re-submits standalone stop-loss.

### 7.4 Close Position Flow

```
intent = OrderIntent(
    symbol=ticker,
    side=OrderSide.SELL if position.direction == "long" else OrderSide.BUY,
    quantity=position.quantity,
    order_type=OrderType.MARKET,
    time_in_force=TimeInForce.DAY,
    client_order_id=f"qt-close-{date}-{ticker}",
    created_at=datetime.now(datetime.timezone.utc),
    position_intent=PositionIntent.CLOSE_POSITION,
    order_class=OrderClass.SIMPLE,
    rationale="Stop loss triggered / take profit / manual close",
)
result = execution_service.submit(intent)
```

## 8. Configuration

```bash
# --- Execution mode ---
EXECUTION_MODE=live                         # paper | shadow | live
LIVE_TRADING_ENABLED=true                   # Must be explicitly true for live
BROKER=alpaca                               # shadow | alpaca | ibkr (future)

# --- Alpaca credentials ---
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # Paper first!

# --- Risk limits ---
MAX_ORDER_NOTIONAL=1000                     # Max $ per order
MAX_DAILY_NOTIONAL=3000                     # Max $ per day
MAX_DAILY_LOSS=500                          # Max daily loss before halt
MAX_POSITION_PCT=0.05                       # Conservative live-launch cap
MAX_DAILY_OPENS=4                           # Max new positions per day
ORDER_TTL_MINUTES=120                       # Reject stale order intents
MAX_SECTOR_PCT=0.40                         # Max sector concentration
ALLOW_SHORT_SELLING=false                   # Must be explicitly true to open short positions

# --- Paths ---
ORDER_LEDGER_PATH=outputs/execution/orders.sqlite
HALT_TRADING_PATH=HALT_TRADING

# --- Reconciliation ---
RECON_POSITION_TOLERANCE=0.01               # 1% position drift tolerance
RECON_ACCOUNT_DRAWDOWN_ALERT=0.05           # 5% equity drawdown alert
```

Live mode fails closed if any required setting is missing or empty.

## 9. Testing Plan

### 9.1 Unit Tests

| Test | Module | What it verifies |
|---|---|---|
| `test_alpaca_adapter_submit` | `execution/brokers/alpaca.py` | Order submission with mocked HTTP |
| `test_alpaca_adapter_idempotency` | `execution/brokers/alpaca.py` | 409 returns existing order |
| `test_alpaca_adapter_market_closed` | `execution/brokers/alpaca.py` | Rejects market orders when closed |
| `test_alpaca_adapter_unsupported_symbol` | `execution/brokers/alpaca.py` | Rejects WTI, LE=F, etc. |
| `test_risk_gate_alpaca_checks` | `execution/risk_gate.py` | Buying power, supported symbol, market hours |
| `test_ledger_idempotency` | `execution/ledger.py` | Duplicate client_order_id is handled |
| `test_ledger_daily_notional` | `execution/ledger.py` | Daily notional accumulation correct |
| `test_reconciler_position_drift` | `execution/reconciler.py` | Detects and classifies drift |
| `test_reconciler_halt_on_critical` | `execution/reconciler.py` | Creates HALT_TRADING on critical drift |
| `test_execution_service_full_flow` | `execution/service.py` | Intent → RiskGate → Broker → Ledger |
| `test_execution_service_rejection` | `execution/service.py` | Rejected intent still persisted in ledger |
| `test_execution_service_shadow_overrides_broker` | `execution/service.py` | Shadow mode uses ShadowBrokerClient even when BROKER=alpaca |
| `test_risk_gate_short_selling_blocked` | `execution/risk_gate.py` | Opening short rejected when ALLOW_SHORT_SELLING=false |
| `test_risk_gate_close_without_position` | `execution/risk_gate.py` | Close intent rejected when no position exists |
| `test_risk_gate_fractional_check` | `execution/risk_gate.py` | Notional order rejected for non-fractionable assets |
| `test_alpaca_adapter_bracket_order` | `execution/brokers/alpaca.py` | Bracket order submits with legs, ledger stores parent+children |
| `test_alpaca_adapter_notional_order` | `execution/brokers/alpaca.py` | Notional-based order converts to qty when asset not fractionable |

### 9.2 Integration Tests

Using Alpaca paper endpoint:

1. Submit a market buy order for a liquid ETF (e.g., SPY, $100 notional).
2. Verify order appears in `get_open_orders()`.
3. Cancel the order.
4. Verify cancellation.
5. Reconcile ledger vs broker — expect `ok`.

### 9.3 End-to-End Test

1. Run `scheduler_service.py` with `EXECUTION_MODE=shadow` and `BROKER=shadow`.
2. Verify no live orders submitted (shadow broker intercepts).
3. Verify shadow orders in SQLite ledger.
4. Run with `BROKER=alpaca` but `EXECUTION_MODE=shadow`. Verify `ExecutionService` still uses `ShadowBrokerClient` (ignores `BROKER` when mode is not `live`).
5. Switch to `EXECUTION_MODE=live`, `LIVE_TRADING_ENABLED=true`, `BROKER=alpaca`.
6. Submit one manually approved small order.
6. Verify fill appears in broker, ledger, and local portfolio.
7. Run reconciler — expect `ok`.

## 10. Migration Steps

### Step 1: Create models and ledger (no behavioral change)

- Add `execution/models.py`.
- Add `execution/ledger.py`.
- Add `execution/brokers/base.py`.
- Run existing tests — should all pass (no integration yet).

### Step 2: Implement ShadowBrokerClient (Phase 1)

- Add `execution/brokers/shadow.py`.
- Add `execution/service.py` with shadow mode.
- Add `execution/risk_gate.py` with Phase 1 checks.
- Run shadow mode for 2+ weeks.

### Step 3: Implement Alpaca adapter

- Add `execution/brokers/alpaca.py`.
- Add Alpaca-specific risk gate checks.
- Run all unit and integration tests against Alpaca paper endpoint.
- Run reconciler in read-only mode against paper account.

### Step 4: Wire scheduler to ExecutionService

- Replace `scheduler_service.py:905` (Bayesian) with `ExecutionService.submit()`.
- Replace `scheduler_service.py:951` (Fallback) with `ExecutionService.submit()`.
- Remove direct `portfolio.open_position()` calls from production paths.
- Keep `MultiAssetPortfolio` for paper/shadow reporting only.

### Step 5: Go-live checklist

1. Confirm shadow mode ran clean for 2+ weeks.
2. Confirm Alpaca paper tests pass.
3. Run reconciler against paper account — expect `ok`.
4. Set `ALPACA_BASE_URL` to live endpoint.
5. Set `EXECUTION_MODE=live`, `LIVE_TRADING_ENABLED=true`.
6. Submit one manually approved order ($100-500 notional).
7. Verify broker fill, ledger entry, portfolio update, reconciler `ok`.
8. Monitor hourly for 24 hours.
9. Gradually increase notional caps.

## 11. Monitoring and Alerts

### Hourly (via scheduler)

- Reconcile broker vs local.
- Log daily P&L summary.
- Log risk decisions (approved/rejected counts).

### Alerts

| Condition | Alert |
|---|---|
| Critical reconciliation drift | Email + create HALT_TRADING |
| Daily loss exceeds 50% of limit | Email warning |
| Order rejected by broker | Log + email (first occurrence per day) |
| `HALT_TRADING` file exists | Email every hour until resolved |
| No successful reconciliation in 3 hours | Email warning |

## 12. Rollback Plan

If live trading causes issues:

1. `touch HALT_TRADING` — immediate stop.
2. Set `EXECUTION_MODE=shadow` — falls back to shadow without code deploy.
3. Set `BROKER=shadow` — bypasses Alpaca entirely.
4. Manually close broker positions via Alpaca dashboard if needed.
5. Review ledger `risk_decisions` table for root cause.

## 13. Out of Scope

- Futures trading (Phase 3 / IBKR).
- Options trading.
- Crypto trading.
- Smart order routing or algorithmic execution.
- Intraday high-frequency trading.
- Multi-broker setup.
- Tax lot tracking.
