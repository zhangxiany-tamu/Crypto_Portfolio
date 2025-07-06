# Crypto Portfolio Optimizer

A web application for cryptocurrency portfolio optimization and analysis with machine learning predictions and AI-powered investment advice.

## Features

- **Market Insights**: Price performance analysis, technical indicators, and coin analytics
- **Portfolio Analysis**: Backtesting with customizable weights and rebalancing strategies
- **Portfolio Optimization**: Maximum Sharpe Ratio, Minimum Variance, and Risk Parity algorithms
- **Machine Learning Predictions**: Multiple ML models including Random Forest, Linear Regression, Lasso, and Gradient Boosting with ensemble predictions
- **AI Investment Advisor**: Integration with OpenAI GPT-4, Anthropic Claude, and Google Gemini
- **Real-time Data**: Data loading with local database and API fallback

## Quick Start

### Live Demo

**[Live application](https://zhangxianyang-crypto-portfolio.share.connect.posit.cloud/)**

### Local Installation

```bash
git clone https://github.com/zhangxiany-tamu/Crypto_Portfolio.git
cd Crypto_Portfolio
python -m pip install -r requirements.txt
```

### Run Locally

```bash
cd frontend
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## Usage

### 1. Market Insights
- View price performance charts with normalized and raw price options
- Analyze individual cryptocurrencies with detailed metrics
- Explore technical indicators including RSI and moving averages
- Review price distributions and returns statistics

### 2. Portfolio Analysis & Backtest
- Define portfolio weights (automatically uses optimized weights if available)
- Run backtesting with various rebalancing strategies
- Analyze performance metrics, drawdowns, and risk-adjusted returns
- Compare against benchmark portfolios

### 3. Portfolio Optimization
- Choose from Maximum Sharpe Ratio, Minimum Variance, or Risk Parity strategies
- Set maximum weight constraints for diversification
- View optimized allocations with immediate weight application
- Seamlessly transfer optimized weights to other analysis modes

### 4. Machine Learning Predictions
- Select prediction horizon (1-30 days) for price forecasting
- Compare multiple ML models: Random Forest, Linear Regression, Lasso, XGBoost/Gradient Boosting
- Ensemble predictions combining all models for improved accuracy
- View portfolio-level predictions using current or optimized weights
- Analyze model confidence and prediction accuracy

### 5. AI Investment Advisor
- Input your current portfolio allocation
- Get investment recommendations from multiple AI models
- Receive analysis of market conditions and portfolio positioning
- Configure API keys for OpenAI GPT-4, Anthropic Claude, or Google Gemini

## Supported Assets

60+ cryptocurrencies with 5+ years of historical data, including:

**Major Assets:** BTC, ETH, XRP, BNB, SOL, DOGE, ADA, MATIC, DOT, UNI

**DeFi & Smart Contracts:** AAVE, LINK, SUSHI, COMP, MKR, YFI, CRV, SNX

**Layer 1/2 & Infrastructure:** AVAX, NEAR, ATOM, ICP, APT, SUI, FTM, ALGO

**Additional Coverage:** Database includes emerging and established cryptocurrencies

## AI Integration

To use the AI Investment Advisor, you'll need an API key from:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://ai.google.dev/

## Project Structure

```
Crypto_Portfolio/
├── frontend/app.py              # Main Streamlit application
├── portfolio_optimizer.py       # Optimization algorithms  
├── robust_data_fetcher.py       # Data fetching utilities
├── enhanced_crypto_loader.py    # Data loading with local database
├── data/crypto_extended.db      # Cryptocurrency database
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## Disclaimer

**For educational and research purposes only.**

- Not financial advice
- Cryptocurrency investments are highly risky
- Past performance does not guarantee future results
- Always do your own research before investing

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Always conduct your own research and consider consulting with financial professionals before making investment decisions.*