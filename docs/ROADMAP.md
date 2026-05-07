# QuantTrade Roadmap — Path to Revenue

## Current State

Production-ready SAR-based maritime throughput pipeline:
- Ship detection (CFAR) at 4 chokepoints (Hormuz, Bab el-Mandeb, Suez, Malacca)
- Gate crossing counting with tracklet linking
- AIS calibration (bias-corrected throughput index)
- Basic trading signals (Long/Short disruption risk)
- Streamlit dashboard + FastAPI backend

**Gap to revenue**: signals exist but are unvalidated, manual, and not mapped to tradeable instruments.

---

## Feature 1: Backtesting Engine

**Priority**: Critical — validates edge before risking capital
**Timeline**: Week 1–2
**Module**: `backtesting/`

### Scope

- Integrate `vectorbt` for strategy backtesting
- Ingest `corrected_metrics.parquet` as signal source
- Pull historical price data (Yahoo Finance / CCXT) for target instruments
- Generate PnL curves, trade logs, and performance reports

### Key Metrics

- Sharpe ratio, Sortino ratio
- Max drawdown, recovery time
- Win rate, profit factor
- Signal-to-trade latency analysis

### Signal-to-Instrument Mapping (for backtests)

| Chokepoint | Primary Instruments | Signal Logic |
|---|---|---|
| Hormuz | Brent/WTI crude futures (CL, BZ), tanker stocks (FRO, STNG, EURN) | Low throughput → long crude / tankers |
| Bab el-Mandeb | Container shipping (ZIM, HLAG), freight rate futures (BDI) | Low throughput → long freight |
| Suez South | Same as Bab el-Mandeb + Suez-sensitive commodity spreads | Disruption → long alternatives |
| Malacca | Asian refinery margins, LNG futures, Singapore fuel oil | Low throughput → long Asian energy |

### Deliverables

- `backtesting/engine.py` — Core backtest runner
- `backtesting/signals.py` — Transform pipeline output into entry/exit signals
- `backtesting/report.py` — HTML/JSON performance report generation
- `configs/instruments.json` — Chokepoint-to-ticker mapping config
- Backtest results in `outputs/backtests/`

### Acceptance Criteria

- [ ] Can run a full backtest on Hormuz throughput vs CL futures
- [ ] Generates Sharpe, drawdown, win rate, PnL curve
- [ ] Supports configurable lookback, holding period, and position sizing
- [ ] Results viewable in UI (new "Backtest" tab)

---

## Feature 2: Daily Automation + Alerting

**Priority**: Critical — captures time-sensitive opportunities
**Timeline**: Week 2–3
**Module**: `scheduler/`, alert integration in `pipeline/`

### Scope

- Automated daily pipeline execution after Sentinel-1 data publishes (~6–12h post-acquisition)
- Push notifications when trading signals fire
- Pipeline health monitoring and failure alerts

### Scheduler Options

| Option | Pros | Cons |
|---|---|---|
| Cron + shell script | Simple, no dependencies | No retry, no DAG |
| Prefect | Retry, monitoring UI, DAG support | Extra infra |
| GitHub Actions | Free, no server needed | Limited compute, latency |

**Recommended**: Start with cron, migrate to Prefect when complexity warrants it.

### Alert Channels

- **Telegram bot** (primary) — instant mobile push, free, easy API
- **Email** (backup) — for daily summary digests
- **WeChat** (optional) — if preferred for Chinese market hours

### Alert Content

```
🔴 SIGNAL: Hormuz — Long Disruption Risk
Confidence: High (coverage 89%)
Corrected Index: 12.3 (7d mean: 18.7, -34%)
Day-over-day: -22%
Suggested: Long CL (crude), Long FRO/STNG (tankers)
```

### Deliverables

- `scheduler/daily_run.sh` — Cron-compatible wrapper script
- `pipeline/alerts.py` — Telegram/email notification module
- `configs/alerts.json` — Channel config, recipient list, thresholds
- Monitoring: alert on pipeline failure, low coverage days, stale data

### Acceptance Criteria

- [ ] Pipeline runs automatically once daily without intervention
- [ ] Telegram alert fires within 5 min of signal generation
- [ ] Failed runs trigger error alerts with diagnostics
- [ ] Daily summary email with all-region status

---

## Feature 3: Improved Signal Logic

**Priority**: High — reduces false signals, increases conviction
**Timeline**: Week 3–4
**Module**: updates to `pipeline/metrics.py`, `ui/app.py`

### Current Logic (Baseline)

```python
if below_trend and falling_fast:
    signal = "Long disruption risk"
elif above_trend and rising_fast:
    signal = "Short disruption risk"
```

### Improvements

#### 3a. Multi-Day Confirmation
- Require 2+ consecutive days of deviation before firing a signal
- Reduces noise from single-day coverage gaps or detection anomalies

#### 3b. Seasonality Adjustment
- Compute day-of-week and monthly seasonal baselines
- Normalize throughput index against seasonal expectation
- Avoid false signals from predictable weekly shipping patterns (e.g., Friday lulls)

#### 3c. Cross-Chokepoint Correlation
- If Hormuz drops but Bab el-Mandeb rises → reroute, not disruption
- If both drop → systemic event, higher confidence signal
- Build a correlation matrix across chokepoints

#### 3d. Confidence Scoring v2
- Weight by: coverage quality, calibration R², detection count, consecutive signal days
- Output a numeric confidence score (0–100) instead of just High/Medium/Low

### Deliverables

- `pipeline/signals.py` — New signal engine (replaces inline logic in UI)
- `configs/signal_rules.json` — Configurable thresholds and rules
- Updated UI Trading tab with confidence score and reasoning

### Acceptance Criteria

- [ ] Signal engine is separate from UI (testable, reusable)
- [ ] Multi-day confirmation reduces false signal rate by >30% in backtest
- [ ] Cross-chokepoint correlation detected and surfaced in alerts

---

## Feature 4: Historical Data Accumulation

**Priority**: Medium — deepens edge over time
**Timeline**: Ongoing (background)

### Scope

- Bulk-run pipeline for 12–24 months of historical Sentinel-1 data
- Build a continuous daily throughput time series per chokepoint
- Store efficiently for backtesting and model training

### Execution Plan

1. Run date ranges in batches (1 month at a time) to manage API rate limits
2. Target: 2023-01-01 to present for all 4 chokepoints
3. Store in `outputs/regions/{region}/metrics/daily.parquet` (append mode)
4. Track completion in `outputs/regions/{region}/bulk_run_status.json`

### Deliverables

- `scripts/bulk_run.py` — Batch historical processing with resume support
- Completed historical dataset for backtesting
- Data quality report (coverage gaps, detection anomalies)

### Acceptance Criteria

- [ ] 12+ months of continuous daily data for Hormuz
- [ ] 6+ months for other 3 chokepoints
- [ ] Coverage gaps documented and flagged

---

## Feature 5: Portfolio-Level Risk Management (Future)

**Priority**: Lower — after validating single-signal profitability
**Timeline**: Month 2+

### Scope

- Position sizing based on signal confidence and portfolio risk budget
- Maximum exposure limits per chokepoint and total
- Correlation-aware hedging (e.g., long crude + short tankers)
- Drawdown-based circuit breakers

### Deliverables

- `backtesting/portfolio.py` — Multi-asset portfolio simulator
- Risk dashboard tab in UI

---

## Execution Summary

```
Week 1–2:  Backtest engine         → Prove signals have edge
Week 2–3:  Daily automation        → Capture live signals
Week 3–4:  Improved signal logic   → Trade with higher conviction
Ongoing:   Historical accumulation → Build data moat
Month 2+:  Portfolio management    → Scale positions safely
```

**Principle**: Don't trade until backtests confirm an edge. Don't run manually once you start trading. Everything else amplifies these two foundations.
