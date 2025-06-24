# 🚀 Crypto Portfolio Optimizer

A sophisticated web application for cryptocurrency portfolio optimization and backtesting, featuring real-time data fetching, advanced optimization algorithms, and professional Coinbase-inspired UI.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

### 📊 **Portfolio Analysis & Optimization**
- **4 Optimization Methods**: Mean-Variance, Risk Parity, Black-Litterman, and Maximum Sharpe
- **Risk Management**: Conservative, Moderate, and Aggressive risk profiles
- **Real-time Data**: Multi-source data fetching with fallback mechanisms (Yahoo Finance, CoinGecko, Binance)
- **Professional Visualizations**: Enhanced correlation matrices with insights, beautiful pie charts with crypto-specific colors

### 🔍 **Market Insights**
- **Enhanced Correlation Analysis**: Interactive heatmaps with diversification insights
- **Risk vs Return Profiles**: Scatter plots showing asset positioning
- **Rolling Performance Metrics**: 30-day rolling analysis with Sharpe ratios
- **Statistical Summaries**: Comprehensive performance statistics

### 📈 **Sample Backtesting**
- **Portfolio Performance Tracking**: Monitor your portfolio against benchmarks
- **Benchmark Comparisons**: Compare against BTC-only and equal-weight strategies
- **Performance Metrics**: Total return, volatility, Sharpe ratio, and maximum drawdown

### 🎨 **Professional UI/UX**
- **Coinbase-Inspired Design**: Clean, modern interface with professional styling
- **Discrete Loading Indicators**: Non-intrusive data loading feedback
- **Responsive Layout**: Optimized for desktop and mobile viewing
- **Interactive Charts**: Plotly-powered visualizations with hover details

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/zhangxiany-tamu/Crypto_Portfolio.git
cd Crypto_Portfolio
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the application**
```bash
cd frontend
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🔧 Usage Guide

### Portfolio Optimization
1. **Select Assets**: Choose from 15 major cryptocurrencies in the sidebar
2. **Set Parameters**: Configure time period and risk tolerance
3. **Choose Method**: Select optimization algorithm (Mean-Variance, Risk Parity, etc.)
4. **Optimize**: Click "Optimize Portfolio" to get optimal weights
5. **Analyze Results**: View allocation charts, risk contributions, and metrics

### Market Analysis
1. **Navigate**: Select "Market Insights" mode
2. **Explore Correlations**: View enhanced correlation matrix with insights
3. **Risk Analysis**: Examine risk vs return scatter plots
4. **Deep Dive**: Select specific assets for detailed rolling analysis

### Portfolio Analysis
1. **Set Weights**: Use "Portfolio Analysis" mode to input custom weights
2. **View Metrics**: See comprehensive performance statistics
3. **Correlation Check**: Review asset correlations for diversification

## 📁 Project Structure

```
Crypto_Portfolio/
├── frontend/
│   ├── app.py                  # Main Streamlit application
│   └── README.md              # Frontend documentation
├── backend/                   # FastAPI backend (optional)
├── portfolio_optimizer.py     # Core optimization algorithms
├── robust_data_fetcher.py     # Multi-source data fetching
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── docs/                     # Additional documentation
```

## 🛠 Technical Details

### Data Sources & Reliability
- **Primary**: Yahoo Finance (yfinance) - Most reliable for historical data
- **Secondary**: CoinGecko API - Real-time price data backup
- **Tertiary**: Binance API - High-frequency data source
- **Fallback**: Cached data and error handling for robust operation

### Optimization Algorithms

#### 1. **Mean-Variance Optimization**
- Maximizes expected return for given risk level
- Uses CVXPY for convex optimization
- Constraints: long-only positions, maximum asset weights

#### 2. **Risk Parity**
- Equal risk contribution from all assets
- Minimizes portfolio concentration risk
- Ideal for diversification-focused strategies

#### 3. **Black-Litterman**
- Bayesian approach with market equilibrium baseline
- Incorporates investor views and confidence levels
- More stable than pure mean-variance optimization

#### 4. **Maximum Sharpe Ratio**
- Optimizes risk-adjusted returns
- Maximizes excess return per unit of risk
- Popular for risk-conscious investors

### Enhanced Visualizations
- **Correlation Matrices**: RdYlBu color scheme with correlation insights panel
- **Pie Charts**: Crypto-specific color palette (Bitcoin Orange, Ethereum Blue, etc.)
- **Interactive Elements**: Hover details, zoom functionality, responsive design

## 🎯 Risk Profiles

| Profile | Target Return | Max Volatility | Max Asset Weight |
|---------|---------------|----------------|------------------|
| **Conservative** | 8% | 15% | 30% |
| **Moderate** | 15% | 25% | 40% |
| **Aggressive** | 25% | 40% | 60% |

## 📊 Supported Cryptocurrencies

- **Bitcoin (BTC)** - Digital gold standard
- **Ethereum (ETH)** - Smart contract platform
- **Binance Coin (BNB)** - Exchange token
- **XRP** - Cross-border payments
- **Cardano (ADA)** - Research-driven blockchain
- **Solana (SOL)** - High-performance blockchain
- **Polygon (MATIC)** - Ethereum scaling solution
- **Polkadot (DOT)** - Interoperable blockchain
- **Avalanche (AVAX)** - Consensus platform
- **Chainlink (LINK)** - Oracle network
- **Uniswap (UNI)** - Decentralized exchange
- **Litecoin (LTC)** - Digital silver
- **Bitcoin Cash (BCH)** - Bitcoin fork
- **Cosmos (ATOM)** - Internet of blockchains
- **Algorand (ALGO)** - Pure proof-of-stake

## ⚠️ Important Disclaimers

**This application is for educational and research purposes only.**

- 📊 **Not Financial Advice**: All results are based on historical data and simulations
- 💹 **Market Risk**: Cryptocurrency markets are highly volatile and unpredictable
- 🔮 **Model Limitations**: Past performance does not guarantee future results
- 💸 **Trading Risks**: Real trading involves additional costs, slippage, and market impact
- 🏛️ **Regulatory Risk**: Cryptocurrency regulations may vary by jurisdiction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
- 📧 Create an issue on GitHub
- 📖 Check the documentation in `/docs`
- 💬 Review existing issues for solutions

---

**Made with ❤️ for the crypto community**

*Always conduct your own research and consider consulting with financial professionals before making investment decisions.*