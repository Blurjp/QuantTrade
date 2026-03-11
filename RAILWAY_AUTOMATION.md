# Railway Daily Automation

## Purpose

Run the daily pipeline on Railway and generate a Chinese brief with:

- 今日可交易
- 今日不可交易
- 今日观察名单
- HTML dashboard

## Recommended Railway setup

Create a Railway service for this repo and use a scheduled job with this command:

```bash
bash scripts/railway_daily_brief.sh
```

## Output files

Each daily run writes to `outputs/YYYY-MM-DD/`:

- `daily_summary.json`
- `daily_brief_zh.md`
- `daily_brief_zh.txt`
- `signals_dashboard.html`

## Email delivery

If you want Railway to email the Chinese brief after each run, set these env vars:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `SMTP_TO`
- optional: `SMTP_STARTTLS=true`

When `SMTP_TO` is present, `scripts/railway_daily_brief.sh` will automatically send `daily_brief_zh.txt` by email.

## Notes

- The command uses `python3`, which fits Railway better than a local `.venv` path.
- If you want a fixed timezone, set Railway env var `TZ=America/New_York` or your preferred timezone.
- If you want to rerun a specific date manually, set:

```bash
RUN_DATE=2026-03-11 bash scripts/railway_daily_brief.sh
```
