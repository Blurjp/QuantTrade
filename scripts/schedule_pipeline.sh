#!/bin/bash
# Auto-run QuantTrade pipeline every hour
# Install: crontab -e
# Add line: 0 * * * * /Users/jianping/projects/QuantTrade/scripts/schedule_pipeline.sh

set -e

PROJECT_DIR="/Users/jianping/projects/QuantTrade"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y-%m-%d)
DATETIME=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="$LOG_DIR/pipeline_$DATE.log"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

echo "[$DATETIME] Starting pipeline run..." >> "$LOG_FILE"

PYTHONPATH=. .venv/bin/python -m pipeline.run_daily --date "$DATE" >> "$LOG_FILE" 2>&1

echo "[$DATETIME] Pipeline complete. Rebuilding asset history..." >> "$LOG_FILE"

PYTHONPATH=. .venv/bin/python scripts/rebuild_asset_history.py --output outputs --initial-capital 100000 >> "$LOG_FILE" 2>&1

echo "[$DATETIME] Done." >> "$LOG_FILE"
