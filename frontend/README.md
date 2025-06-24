# Crypto Portfolio Optimizer - Frontend

## Quick Start

```bash
streamlit run app.py
```

## Features

- **Advanced Portfolio Optimization**: 6 different algorithms (Mean-Variance, Risk Parity, etc.)
- **Complete Backtesting**: Portfolio performance analysis vs benchmarks
- **Market Insights**: Risk-return analysis, correlation matrices, rolling metrics
- **Real Market Data**: Multi-source data fetching with smart caching
- **Professional UI**: Coinbase-inspired light mode design
- **Smart Data Caching**: Only fetches data when symbols/dates change

## Analysis Modes

1. **Portfolio Analysis**: Basic portfolio metrics and price performance
2. **Portfolio Optimization**: Advanced optimization with customizable parameters
3. **Sample Backtest**: Complete backtesting with benchmarks and drawdown analysis
4. **Market Insights**: Detailed market analysis and statistical insights

## Files

- `app.py` - Main application (full-featured version)
- `app_backup.py` - Backup copy

## Dependencies

Requires parent directory files:
- `portfolio_optimizer.py` - Advanced optimization algorithms
- `robust_data_fetcher.py` - Multi-source data fetching

## Usage

1. Choose analysis mode in sidebar
2. Configure time period and cryptocurrencies  
3. Set optimization parameters (if applicable)
4. Run analysis/optimization

## Data Caching

- Data is automatically cached for 1 hour
- Only re-fetches when symbols or date range changes
- Use "Refresh Data" button to force fresh data

Access at: http://localhost:8503