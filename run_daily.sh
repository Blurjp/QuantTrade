#!/bin/bash
# Run daily pipeline with correct Python environment
# Usage: ./run_daily.sh

# Activate virtual environment and set PYTHONPATH
source .venv/bin/activate
export PYTHONPATH=/Users/jianping/projects/QuantTrade

# Run the daily pipeline
python scripts/run_daily.py
