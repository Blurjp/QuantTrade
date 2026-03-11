#!/bin/bash
# Start QuantTrade Dashboard

cd /Users/jianpinghuang/.openclaw/workspace/projects/QuantTrade

echo "🚀 Starting QuantTrade Dashboard..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Check if port 8501 is available
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8501 is already in use"
    echo "   Dashboard may already be running"
    echo ""
    echo "   To stop: pkill -f 'streamlit run'"
    echo "   Then run this script again"
    echo ""
    echo "   Or access existing dashboard at: http://localhost:8501"
    exit 1
fi

# Start dashboard
echo "📊 Starting Streamlit server..."
echo "   URL: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

streamlit run dashboard/app.py --server.headline=true
