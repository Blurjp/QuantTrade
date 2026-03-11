#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATE="${RUN_DATE:-$(date +%F)}"

cd "$PROJECT_DIR"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m pipeline.run_daily --output outputs
"$PYTHON_BIN" scripts/china_daily_brief.py --date "$DATE" --output outputs
"$PYTHON_BIN" scripts/active_signals_report.py --date "$DATE" --output outputs
"$PYTHON_BIN" scripts/signals_dashboard.py --date "$DATE" --output outputs

if [ -n "${SMTP_TO:-}" ]; then
  "$PYTHON_BIN" scripts/send_email_report.py "QuantTrade 每日简报 ${DATE}" < "outputs/${DATE}/daily_brief_zh.txt"
fi
