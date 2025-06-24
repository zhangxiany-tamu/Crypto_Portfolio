#!/usr/bin/env python3
"""
Crypto Portfolio Optimizer - Sample Data Demonstration
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from core.portfolio_optimizer import PortfolioOptimizer
from core.backtester import Backtester

class MockDataManager:
    """Mock data manager with sample data for demonstration"""
    
    def __init__(self):
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
        
    async def get_price_data(self, symbols, start_date, end_date):
        """Generate sample price data"""
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic crypto price movements
        np.random.seed(42)  # For reproducible results
        
        price_data = pd.DataFrame(index=dates)
        
        # Starting prices (approximate recent values)
        starting_prices = {
            'BTC-USD': 45000,
            'ETH-USD': 3000,
            'BNB-USD': 300,
            'XRP-USD': 0.6,
            'ADA-USD': 0.5
        }
        
        # Generate price series for each symbol
        for symbol in symbols:
            if symbol in starting_prices:
                # Parameters for realistic crypto volatility
                drift = 0.0005  # Small positive drift
                volatility = {'BTC-USD': 0.04, 'ETH-USD': 0.05, 'BNB-USD': 0.06, 
                             'XRP-USD': 0.08, 'ADA-USD': 0.09}[symbol]
                
                # Generate random walk
                returns = np.random.normal(drift, volatility, len(dates))
                
                # Calculate prices
                prices = [starting_prices[symbol]]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                price_data[symbol] = prices[:len(dates)]
        
        return price_data
    
    def calculate_returns(self, price_data):
        """Calculate returns from price data"""
        return price_data.pct_change().dropna()
    
    async def get_available_symbols(self):
        return self.crypto_symbols

async def demonstrate_functionality():
    print("🚀 Crypto Portfolio Optimizer - Sample Data Demonstration")
    print("=" * 70)
    
    # Initialize components with mock data
    print("\n📊 Initializing components with sample data...")
    mock_data_manager = MockDataManager()
    portfolio_optimizer = PortfolioOptimizer(mock_data_manager)
    backtester = Backtester(mock_data_manager)
    
    # Demo parameters
    symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=365)    # 1 year ago
    
    print(f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"💰 Selected Cryptocurrencies: {', '.join(symbols)}")
    print("📊 Using simulated market data for demonstration")
    
    try:
        # 1. Data Fetching Demo
        print("\n" + "="*70)
        print("1️⃣  DATA FETCHING DEMONSTRATION")
        print("="*70)
        
        print("🔄 Generating sample historical price data...")
        price_data = await mock_data_manager.get_price_data(
            symbols, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"✅ Successfully generated {len(price_data)} days of data")
        print(f"📊 Data shape: {price_data.shape}")
        print("\n📈 Sample prices (last 5 days):")
        print(price_data.tail().round(2))
        
        # Calculate basic statistics
        returns = mock_data_manager.calculate_returns(price_data)
        print(f"\n📊 Average Daily Returns (Annualized):")
        for symbol in symbols:
            if symbol in returns.columns:
                avg_return = returns[symbol].mean() * 252
                volatility = returns[symbol].std() * np.sqrt(252)
                print(f"   {symbol:10}: {avg_return:+6.2%} (σ: {volatility:5.2%})")
        
        # 2. Portfolio Optimization Demo
        print("\n" + "="*70)
        print("2️⃣  PORTFOLIO OPTIMIZATION DEMONSTRATION")
        print("="*70)
        
        optimization_methods = ['mean_variance', 'risk_parity', 'black_litterman']
        risk_tolerances = ['conservative', 'moderate', 'aggressive']
        
        optimization_results = {}
        
        for risk_tolerance in risk_tolerances:
            print(f"\n🎯 Optimizing for {risk_tolerance.upper()} risk tolerance:")
            
            for method in optimization_methods:
                print(f"   🔧 Running {method.replace('_', ' ').title()} optimization...")
                
                try:
                    result = await portfolio_optimizer.optimize(
                        symbols=symbols,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        risk_tolerance=risk_tolerance,
                        method=method
                    )
                    
                    optimization_results[f"{risk_tolerance}_{method}"] = result
                    
                    print(f"      📊 Expected Return: {result['expected_return']:.2%}")
                    print(f"      📉 Volatility: {result['volatility']:.2%}")
                    print(f"      ⚡ Sharpe Ratio: {result['sharpe_ratio']:.3f}")
                    
                    print("      💼 Optimal Weights:")
                    for i, symbol in enumerate(result['symbols']):
                        weight = result['weights'][i]
                        print(f"         {symbol:10}: {weight:6.1%}")
                    
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
        
        # 3. Backtesting Demo
        print("\n" + "="*70)
        print("3️⃣  BACKTESTING DEMONSTRATION")
        print("="*70)
        
        # Use moderate mean-variance portfolio for backtesting
        if 'moderate_mean_variance' in optimization_results:
            test_portfolio = optimization_results['moderate_mean_variance']
            
            print("🔄 Running comprehensive backtest...")
            print(f"   Portfolio: Moderate Mean-Variance Optimization")
            weights_str = [f'{w:.1%}' for w in test_portfolio['weights']]
            print(f"   Weights: {dict(zip(symbols, weights_str))}")
            
            backtest_result = await backtester.run_backtest(
                symbols=test_portfolio['symbols'],
                weights=test_portfolio['weights'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=100000,
                rebalance_freq='monthly'
            )
            
            portfolio_metrics = backtest_result['portfolio']['metrics']
            btc_metrics = backtest_result['benchmarks']['btc_hold']['metrics']
            
            print("\n📊 BACKTEST RESULTS:")
            print(f"   💰 Initial Capital: $100,000")
            print(f"   📅 Rebalancing: Monthly")
            print(f"   💸 Trading Fees: 0.1% per trade")
            
            print(f"\n🎯 PORTFOLIO PERFORMANCE:")
            print(f"   Final Value:        ${backtest_result['portfolio']['returns'][-1]:,.0f}")
            print(f"   Total Return:       {portfolio_metrics['total_return']:+7.2%}")
            print(f"   Annualized Return:  {portfolio_metrics['annualized_return']:+7.2%}")
            print(f"   Volatility:         {portfolio_metrics['volatility']:7.2%}")
            print(f"   Sharpe Ratio:       {portfolio_metrics['sharpe_ratio']:7.3f}")
            print(f"   Max Drawdown:       {portfolio_metrics['max_drawdown']:7.2%}")
            print(f"   Calmar Ratio:       {portfolio_metrics['calmar_ratio']:7.3f}")
            print(f"   Win Rate:           {portfolio_metrics['win_rate']:7.2%}")
            
            print(f"\n₿ BTC BUY & HOLD BENCHMARK:")
            print(f"   Final Value:        ${backtest_result['benchmarks']['btc_hold']['returns'][-1]:,.0f}")
            print(f"   Total Return:       {btc_metrics['total_return']:+7.2%}")
            print(f"   Annualized Return:  {btc_metrics['annualized_return']:+7.2%}")
            print(f"   Volatility:         {btc_metrics['volatility']:7.2%}")
            print(f"   Sharpe Ratio:       {btc_metrics['sharpe_ratio']:7.3f}")
            print(f"   Max Drawdown:       {btc_metrics['max_drawdown']:7.2%}")
            
            print(f"\n⚖️  COMPARISON (Portfolio vs BTC Hold):")
            comparison = backtest_result['comparison']['vs_btc']
            print(f"   Return Difference:  {comparison['return_diff']:+7.2%}")
            print(f"   Sharpe Difference:  {comparison['sharpe_diff']:+7.3f}")
            print(f"   Volatility Diff:    {comparison['volatility_diff']:+7.2%}")
            print(f"   Max DD Difference:  {comparison['max_dd_diff']:+7.2%}")
            
            # Performance ranking
            rankings = backtest_result['comparison']['ranking']
            print(f"\n🏆 STRATEGY RANKINGS:")
            for metric, ranking in rankings.items():
                print(f"   {metric.replace('_', ' ').title():15}: {' > '.join(ranking)}")
        
        # 4. Risk Analysis
        print("\n" + "="*70)
        print("4️⃣  PORTFOLIO RISK ANALYSIS")
        print("="*70)
        
        # Compare all optimized portfolios
        print("📊 OPTIMIZATION METHOD COMPARISON:")
        print(f"{'Method':<20} {'Risk Level':<12} {'Return':<8} {'Vol':<6} {'Sharpe':<7}")
        print("-" * 55)
        
        for key, result in optimization_results.items():
            risk_level, method = key.split('_', 1)
            method_name = method.replace('_', ' ').title()
            print(f"{method_name:<20} {risk_level.title():<12} "
                  f"{result['expected_return']:>6.1%} {result['volatility']:>5.1%} "
                  f"{result['sharpe_ratio']:>6.2f}")
        
        # 5. Market Analysis Summary
        print("\n" + "="*70)
        print("5️⃣  MARKET ANALYSIS SUMMARY")
        print("="*70)
        
        # Correlation analysis
        correlations = returns.corr()
        print("🔗 Asset Correlation Matrix:")
        print(correlations.round(3))
        
        # Volatility analysis
        print("\n📊 Volatility Ranking (Annualized):")
        volatilities = returns.std() * np.sqrt(252)
        vol_sorted = volatilities.sort_values(ascending=False)
        
        for i, (symbol, vol) in enumerate(vol_sorted.items(), 1):
            risk_level = "🔴 High" if vol > 0.6 else "🟡 Medium" if vol > 0.3 else "🟢 Low"
            print(f"   {i}. {symbol:10}: {vol:6.2%} {risk_level}")
        
        print("\n✅ Demonstration completed successfully!")
        print("\n" + "="*70)
        print("🌐 NEXT STEPS:")
        print("   1. Run the Streamlit web interface:")
        print("      cd frontend && streamlit run app.py")
        print("   2. Access the FastAPI backend:")
        print("      cd backend && python main.py")
        print("   3. Run tests:")
        print("      pytest tests/")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_functionality())