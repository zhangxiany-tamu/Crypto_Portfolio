#!/bin/bash

# Crypto Portfolio Optimizer - Launch Script

echo "🚀 Starting Crypto Portfolio Optimizer..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Launch options
echo ""
echo "Choose launch option:"
echo "1) Streamlit Web Interface (Recommended)"
echo "2) FastAPI Backend Only"
echo "3) Both (Streamlit + FastAPI)"

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🌐 Starting Streamlit interface..."
        cd frontend
        streamlit run app.py
        ;;
    2)
        echo "🔧 Starting FastAPI backend..."
        cd backend
        python main.py
        ;;
    3)
        echo "🚀 Starting both services..."
        echo "Starting FastAPI backend in background..."
        cd backend
        python main.py &
        BACKEND_PID=$!
        cd ../frontend
        echo "Starting Streamlit interface..."
        streamlit run app.py
        # Kill backend when streamlit exits
        kill $BACKEND_PID
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac