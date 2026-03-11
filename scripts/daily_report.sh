#!/bin/bash
# QuantTrade Paper Trading Daily Report
# Sends daily P&L update to Discord

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)

cd "$PROJECT_DIR"
source .venv/bin/activate

# Get current oil price (would integrate with API in production)
# For now, use environment variable or default
OIL_PRICE=${OIL_PRICE:-120.0}

# Generate report
python -c "
from paper_trading.daily_report import generate_daily_report, format_report_message
import os

price = float(os.environ.get('OIL_PRICE', 120.0))

report = generate_daily_report(
    region='hormuz',
    output_base='outputs',
    initial_capital=100000,
    current_price=price,
)

print(format_report_message(report))
" 2>&1

echo
echo "Brazil soy snapshot"
echo "-------------------"
python scripts/brazil_soy_report.py --date "$DATE" --output outputs 2>&1 || true

echo
python scripts/active_signals_report.py --date "$DATE" --output outputs 2>&1 || true

echo
python scripts/signals_dashboard.py --date "$DATE" --output outputs 2>&1 || true
