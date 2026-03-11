#!/bin/bash
# QuantTrade Daily Pipeline Runner
# Runs detection for all configured regions and generates signals

set -e

PROJECT_DIR="/Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/daily_${DATE}.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate virtual environment
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "========================================" | tee -a "$LOG_FILE"
echo "QuantTrade Daily Run - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run daily automation for all regions
echo "Running daily pipeline for all regions..." | tee -a "$LOG_FILE"
python -m automation.daily --date "$DATE" --output outputs 2>&1 | tee -a "$LOG_FILE"

# Generate signal summary
echo "" | tee -a "$LOG_FILE"
echo "Signal Summary:" | tee -a "$LOG_FILE"
python -c "
from pipeline.signals import latest_region_signal
from pipeline.regions import list_regions

for region in list_regions():
    signal = latest_region_signal(region['id'], output_base='outputs', version='v2')
    if signal:
        print(f\"  {region['name']}: {signal['signal']} ({signal['confidence']}) - {signal['actionability']}\")
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Daily run complete at $(date)" | tee -a "$LOG_FILE"
