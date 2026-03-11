# QuantTrade Automation Setup

## Quick Start

### 1. Install Crontab (Daily Automation)

```bash
crontab /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade/scripts/quanttrade.cron
```

Verify installation:
```bash
crontab -l
```

### 2. Manual Daily Run

If you prefer to run manually instead of cron:

```bash
cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade
./scripts/daily_run.sh
```

### 3. Backfill Historical Data

```bash
cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade
source .venv/bin/activate

# Backfill last 30 days
python -m pipeline.run --region hormuz --start 2026-02-01 --end 2026-03-09 --output outputs

# Or use the backfill automation
python -m automation.backfill --days 30 --output outputs
```

## Schedule

| Task | Schedule | Script |
|------|----------|--------|
| Daily pipeline | 6:00 AM EST | `scripts/daily_run.sh` |
| Weekly backfill | Sunday 5:00 AM EST | `automation.backfill` |

## Logs

- Daily runs: `logs/daily_YYYY-MM-DD.log`
- Backfill: `logs/backfill.log`

## Monitoring

Check latest signal for all regions:
```bash
source .venv/bin/activate
python -c "
from pipeline.signals import latest_region_signal
from pipeline.regions import list_regions

for region in list_regions():
    signal = latest_region_signal(region['id'], output_base='outputs')
    if signal:
        print(f\"{region['name']}: {signal['signal']} ({signal['confidence']})\")
"
```

## Troubleshooting

### No Sentinel-1 data for date
- Normal - Sentinel-1 has ~6-day revisit time
- Check coverage in `outputs/YYYY-MM-DD/qa/run_report.json`

### Low confidence signals
- Need more historical data for baseline
- Run backfill to build up history

### Crontab not running
- Check cron service: `sudo launchctl list | grep cron`
- Check logs: `logs/daily_*.log`
