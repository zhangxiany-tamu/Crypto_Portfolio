# Crypto Portfolio Optimizer

A web application for cryptocurrency portfolio optimization and analysis with machine learning predictions and AI-powered investment advice.

## Features

- **Portfolio Optimization**: Mean-Variance, Risk Parity, Black-Litterman, and Maximum Sharpe algorithms
- **Machine Learning Predictions**: Random Forest and Linear Regression models with technical indicators
- **AI Investment Advisor**: Integration with OpenAI GPT-4, Anthropic Claude, and Google Gemini
- **Market Analysis**: Correlation matrices, risk-return analysis, and rolling performance metrics
- **Real-time Data**: Multi-source data fetching from Yahoo Finance, CoinGecko, and Binance
- **Interactive Visualizations**: Professional charts and analytics

## Quick Start

### Installation

```bash
git clone https://github.com/zhangxiany-tamu/Crypto_Portfolio.git
cd Crypto_Portfolio
pip install -r requirements.txt
```

### Run the Application

```bash
cd frontend
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## Usage

### 1. Portfolio Optimization
- Select cryptocurrencies from the sidebar
- Choose optimization method and risk tolerance
- Get optimal portfolio weights and allocations

### 2. Portfolio Analysis
- Define custom portfolio weights
- View performance metrics and risk analysis
- Analyze correlations and diversification

### 3. Machine Learning Predictions
- Select prediction horizon (1-30 days)
- Compare Random Forest and Linear Regression models
- View individual asset and portfolio-level forecasts

### 4. AI Investment Advisor
- Configure your current portfolio allocation
- Get AI-powered investment recommendations
- Receive precise rebalancing plans with expected returns

### 5. Market Insights
- Explore asset correlations and market trends
- Analyze risk-return profiles
- View rolling performance metrics

## Supported Assets

BTC, ETH, BNB, XRP, ADA, SOL, MATIC, DOT, AVAX, LINK, UNI, LTC, BCH, ATOM, ALGO

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