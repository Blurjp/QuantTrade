#!/bin/bash
###############################################################################
# QuantTrade Daily Automated Pipeline Runner
#
# This script runs the daily trading signal pipeline automatically.
# It's designed to be called by cron or systemd timer.
#
# Usage:
#   ./scripts/run_daily_automated.sh
#
# Environment setup:
#   - Make sure PYTHONPATH includes the project root
#   - Optional: Set up a virtual environment
###############################################################################

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Project configuration
PROJECT_DIR="/Users/jianping/projects/QuantTrade"
PYTHONPATH="${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs/logs"
LOG_FILE="${LOG_DIR}/daily_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "Starting QuantTrade Daily Pipeline"
log "===================================="

# Change to project directory
cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    log "Activating virtual environment..."
    source "${PROJECT_DIR}/.venv/bin/activate"
elif [ -f "${PROJECT_DIR}/.venv313/bin/activate" ]; then
    log "Activating virtual environment (Python 3.13)..."
    source "${PROJECT_DIR}/.venv313/bin/activate"
fi

# Run the daily pipeline
log "Running daily signal generation..."
PYTHONPATH="${PYTHONPATH}" python scripts/run_daily.py >> "${LOG_FILE}" 2>&1

PIPELINE_EXIT_CODE=$?

if [ ${PIPELINE_EXIT_CODE} -eq 0 ]; then
    log "✅ Pipeline completed successfully"

    # Count actionable signals
    if [ -f "outputs/$(date +%Y-%m-%d)/daily_summary.json" ]; then
        ACTIONABLE=$(python -c "
import json
from pathlib import Path
summary = json.loads(Path('outputs/$(date +%Y-%m-%d)/daily_summary.json').read_text())
actionable = sum(1 for s in summary.get('signals', {}).values() if s.get('actionability') == 'Actionable')
print(actionable)
" 2>/dev/null || echo "?")
        log "📊 Actionable signals today: ${ACTIONABLE}"

        # Send email notification if actionable signals exist
        if [ "${ACTIONABLE}" != "0" ] && [ "${ACTIONABLE}" != "?" ]; then
            log "📧 Sending email notification for ${ACTIONABLE} actionable signals..."
            PYTHONPATH="${PYTHONPATH}" python -c "from pipeline.notifications import send_notification_on_signals; send_notification_on_signals()" >> "${LOG_FILE}" 2>&1
            if [ $? -eq 0 ]; then
                log "✅ Email notification sent"
            else
                log "⚠️ Email notification failed (check configuration)"
            fi
        fi
    fi
else
    log "❌ Pipeline failed with exit code ${PIPELINE_EXIT_CODE}"
    # Send failure notification if email is configured
    log "📧 Attempting to send failure notification..."
    PYTHONPATH="${PYTHONPATH}" python -c "
import os
from pipeline.notifications import SignalNotifier
notifier = SignalNotifier()
if notifier.config.is_configured():
    notifier.send_email('❌ QuantTrade Pipeline Failed', f'The pipeline failed on $(date +%Y-%m-%d) with exit code ${PIPELINE_EXIT_CODE}. Check logs for details.')
" >> "${LOG_FILE}" 2>&1 || true
fi

log "Pipeline run completed"
log "Log file: ${LOG_FILE}"

# Optional: Clean up old logs (keep last 30 days)
find "${LOG_DIR}" -name "daily_pipeline_*.log" -mtime +30 -delete 2>/dev/null || true

exit 0
