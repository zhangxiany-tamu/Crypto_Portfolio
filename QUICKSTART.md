# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+ installed
- Internet connection for data fetching

## 1. Setup (One-time)

```bash
# Navigate to project directory
cd "/Users/xianyangzhang/My Drive/Crypto_Portfolio/crypto_portfolio"

# Install dependencies (if not already done)
pip install streamlit pandas numpy plotly yfinance scikit-learn cvxpy scipy

# Make launch script executable
chmod +x run.sh
```

## 2. Launch Application

### Option A: Simple Launch
```bash
streamlit run frontend/app_simple.py
```

### Option B: Full Launch Script
```bash
./run.sh
```

### Option C: Demo Script
```bash
python demo_sample.py
```

## 3. Access the Application

Open your browser and go to: **http://localhost:8501**

## 4. Using the Interface

### Portfolio Analysis Mode
1. Select cryptocurrencies from the sidebar
2. Set time period for analysis
3. View price charts, correlations, and statistics

### Sample Backtest Mode
1. Set portfolio weights using sliders
2. Configure initial capital and rebalancing
3. Compare performance vs BTC hold strategy

### Market Insights Mode
1. Analyze risk-return profiles
2. View rolling performance metrics
3. Examine market correlations

## 5. Key Features

✅ **Portfolio Optimization**: Mean-variance, risk parity, Black-Litterman
✅ **Backtesting**: Compare against BTC and equal-weight benchmarks  
✅ **Risk Analysis**: Sharpe ratio, volatility, max drawdown
✅ **Market Analysis**: Correlations, technical indicators
✅ **Interactive Charts**: Zoom, hover, export capabilities

## 6. Sample Workflows

### Optimize a Portfolio
1. Go to "Portfolio Analysis" mode
2. Select 3-5 cryptocurrencies
3. Review correlation matrix and risk metrics
4. Run optimization (backend) for optimal weights

### Backtest a Strategy  
1. Switch to "Sample Backtest" mode
2. Set portfolio weights manually
3. Configure initial capital ($100,000 default)
4. Compare performance vs benchmarks

### Analyze Market Conditions
1. Use "Market Insights" mode
2. View risk-return scatter plot
3. Analyze rolling Sharpe ratios
4. Check volatility levels

## 7. Next Steps

- **Real Data**: Replace sample data with live Yahoo Finance feeds
- **Advanced Optimization**: Enable mean-variance with QCP solver
- **Export Results**: Save portfolios and backtests to CSV
- **Custom Strategies**: Add momentum, mean reversion factors

## Troubleshooting

**Port Already in Use?**
```bash
streamlit run frontend/app_simple.py --server.port 8502
```

**Dependencies Missing?**
```bash
pip install -r requirements.txt
```

**Performance Issues?**
- Use shorter time periods (6 months vs 1 year)
- Reduce number of assets (3-4 vs 5+)

---

## Support

- Check the main README.md for detailed documentation
- Run `python demo_sample.py` for a command-line demonstration
- All code is documented with inline comments