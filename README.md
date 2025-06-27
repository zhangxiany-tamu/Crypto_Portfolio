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

### Live Demo

🌐 **[Try the live application](https://zhangxianyang-crypto-portfolio.share.connect.posit.cloud/)**

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

**30 cryptocurrencies across major categories:**

**Top Market Cap:** BTC, ETH, XRP, BNB, SOL, DOGE, ADA, TRX, SHIB, AVAX

**DeFi & Layer 1/2:** LINK, DOT, UNI, AAVE, MATIC, NEAR, ICP, APT, SUI, ATOM

**Established Coins:** LTC, BCH, XLM, XMR, ETC, HBAR, TON, ALGO, VET, FTM

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