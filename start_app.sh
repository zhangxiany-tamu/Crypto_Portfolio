#!/bin/bash

echo "🚀 Starting Crypto Portfolio Optimizer..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the crypto_portfolio directory"
    exit 1
fi

# Kill any existing Streamlit processes to free up ports
echo "🔄 Stopping any existing Streamlit processes..."
pkill -f streamlit 2>/dev/null || true
sleep 2

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip list | grep -q streamlit || pip install streamlit pandas numpy plotly

# Start the application
echo "🌐 Starting Streamlit application..."
echo "   -> Open your browser and go to: http://localhost:8503"
echo "   -> Press Ctrl+C to stop the application"
echo ""

cd frontend
streamlit run app_with_optimization.py --server.port 8503 --server.headless false

echo "👋 Application stopped."