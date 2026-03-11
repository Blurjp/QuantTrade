# QuantTrade Paper Trading System

Simulated trading account based on QuantTrade satellite signals.

## Initial Setup

- **Starting Capital:** $100,000
- **Max Position Size:** 5% of capital ($5,000)
- **Stop Loss:** 4%
- **Take Profit:** 15%

## Trading Strategy

| Signal | Confidence | Actionability | Action |
|--------|------------|---------------|--------|
| Long disruption risk | High/Medium | Actionable | SHORT oil |
| Short disruption risk | High/Medium | Actionable | Close short / stay flat |
| Any | Low | Ignore | No action |

## Current Position (2026-03-09)

- **Position:** SHORT 41.67 contracts
- **Entry:** $120.00
- **Target:** $100-105 (risk premium collapse)
- **Stop Loss:** $125

## Rationale

- Satellite data shows Hormuz throughput normalized (3/4-7: throughput=1)
- Market prices $120 assuming severe disruption
- Expecting $20-30 risk premium to collapse
- Bet: Market overreacting, actual flow is normal

## Daily Reports

Reports are generated automatically and saved to:
```
outputs/paper_trading/report_YYYY-MM-DD.json
```

Account state is saved to:
```
outputs/paper_trading/account_state.json
```

## Manual Commands

Check current position:
```bash
cd /path/to/QuantTrade
source .venv/bin/activate
python -c "
from paper_trading.portfolio import PaperTradingAccount
account = PaperTradingAccount(100000, 'outputs')
summary = account.get_summary(120.0)  # current price
print(f\"Total: \${summary['total_value']:,.2f}\")
print(f\"Return: {summary['total_return_pct']:+.2f}%\")
print(f\"Position: {summary['position']}\")
"
```

Update with new price:
```bash
OIL_PRICE=115.0 ./scripts/daily_report.sh
```

## Risk Management

- Position size limited to 5% of capital
- Automatic stop loss at -4%
- Automatic take profit at +15%
- Maximum one open position at a time

## Performance Tracking

All trades are recorded in `account_state.json`:
- Open/close timestamps
- Entry/exit prices
- Realized P&L
- Rationale for each trade

---

**Disclaimer:** This is a paper trading simulation for educational purposes. Not financial advice.
