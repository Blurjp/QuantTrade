#!/bin/bash
# Start script for Railway deployment
# Usage: Set SERVICE_TYPE env var to "web" or "scheduler"

if [ "$SERVICE_TYPE" = "scheduler" ]; then
    echo "Starting scheduler service..."
    exec python scheduler_service.py
else
    echo "Starting web UI..."
    exec streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0
fi
