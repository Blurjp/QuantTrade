#!/bin/bash
# QuantTrade Unified Daily Pipeline
# Processes all active monitoring targets and generates trading signals

set -e

PROJECT_DIR="/Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/daily_${DATE}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "QuantTrade Daily Pipeline Started"
log "========================================"
log ""

cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Step 1: Run detection for all active regions
log "Step 1: Running multi-region detection..."
python -m pipeline.run_daily --date "$DATE" --output outputs 2>&1 | tee -a "$LOG_FILE"
log ""

# Step 2: Generate trading signals
log "Step 2: Generating trading signals..."
python -c "
from paper_trading.daily_multi_report import generate_multi_asset_report, format_discord_report
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
import json
from pathlib import Path

# Load portfolio
portfolio = MultiAssetPortfolio(100000, 'outputs')

# Load today's signals if available
signals = {}
signal_file = Path('outputs') / '$DATE' / 'daily_summary.json'
if signal_file.exists():
    data = json.loads(signal_file.read_text())
    signals = data.get('signals', {})

# Placeholder prices (would fetch from API in production)
prices = {
    'WTI': 120.0,
    'Brent': 118.0,
    'WMT': 165.0,
    'COST': 720.0,
    'F': 12.50,
    'GM': 42.0,
    'XLI': 110.0,
}

# Update positions with prices
closed = portfolio.update_position_prices(prices)
for trade in closed:
    print(f'  Trade closed: {trade.ticker} P&L: \${trade.pnl:+,.2f}')

# Generate report
report = generate_multi_asset_report(portfolio, prices, signals)
print(report)
" 2>&1 | tee -a "$LOG_FILE"
log ""

# Step 3: Check for actionable signals
log "Step 3: Checking actionable signals..."
python -c "
from paper_trading.multi_asset_portfolio import MultiAssetPortfolio
from paper_trading.daily_multi_report import generate_multi_asset_report
import json
from pathlib import Path

# Load today's summary
summary_file = Path('outputs') / '$DATE' / 'daily_summary.json'
if not summary_file.exists():
    print('No summary file found')
    exit(0)

summary = json.loads(summary_file.read_text())
signals = summary.get('signals', {})

actionable = []
for region, sig in signals.items():
    if sig.get('actionability') == 'Actionable':
        actionable.append(f'  {region}: {sig[\"signal\"]} ({sig[\"confidence\"]})')

if actionable:
    print('Actionable signals found:')
    for s in actionable:
        print(s)
else:
    print('No actionable signals today')
" 2>&1 | tee -a "$LOG_FILE"
log ""

log "========================================"
log "Daily Pipeline Complete"
log "Log: $LOG_FILE"
log "========================================"
