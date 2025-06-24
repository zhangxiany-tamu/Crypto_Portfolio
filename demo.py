#!/usr/bin/env python3
"""
Crypto Portfolio Optimizer - Demonstration Script
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from core.data_manager import DataManager
from core.portfolio_optimizer import PortfolioOptimizer
from core.backtester import Backtester

async def demonstrate_functionality():
    print("🚀 Crypto Portfolio Optimizer - Live Demonstration")
    print("=" * 60)
    
    # Initialize components
    print("\n📊 Initializing components...")
    data_manager = DataManager()
    portfolio_optimizer = PortfolioOptimizer(data_manager)
    backtester = Backtester(data_manager)
    
    # Demo parameters
    symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"💰 Selected Cryptocurrencies: {', '.join(symbols)}")
    
    try:
        # 1. Data Fetching Demo
        print("\n" + "="*60)
        print("1️⃣  DATA FETCHING DEMONSTRATION")
        print("="*60)
        
        print("🔄 Fetching historical price data...")
        price_data = await data_manager.get_price_data(
            symbols, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"✅ Successfully fetched {len(price_data)} days of data")
        print(f"📊 Data shape: {price_data.shape}")
        print("\n📈 Latest prices:")
        for symbol in symbols:
            if symbol in price_data.columns:
                latest_price = price_data[symbol].iloc[-1]
                print(f"   {symbol:10}: ${latest_price:,.2f}")
        
        # Calculate basic statistics
        returns = data_manager.calculate_returns(price_data)
        print(f"\n📊 Average Daily Returns (Annualized):")
        for symbol in symbols:
            if symbol in returns.columns:
                avg_return = returns[symbol].mean() * 252
                volatility = returns[symbol].std() * np.sqrt(252)
                print(f"   {symbol:10}: {avg_return:+6.2%} (σ: {volatility:5.2%})")
        
        # 2. Portfolio Optimization Demo
        print("\n" + "="*60)
        print("2️⃣  PORTFOLIO OPTIMIZATION DEMONSTRATION")
        print("="*60)
        
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
        print("\n" + "="*60)
        print("3️⃣  BACKTESTING DEMONSTRATION")
        print("="*60)
        
        # Use moderate mean-variance portfolio for backtesting
        if 'moderate_mean_variance' in optimization_results:
            test_portfolio = optimization_results['moderate_mean_variance']
            
            print("🔄 Running comprehensive backtest...")
            print(f"   Portfolio: Moderate Mean-Variance Optimization")
            print(f"   Weights: {[f'{w:.1%}' for w in test_portfolio['weights']]}")
            
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
            print(f"   Total Return:       {portfolio_metrics['total_return']:+7.2%}")
            print(f"   Annualized Return:  {portfolio_metrics['annualized_return']:+7.2%}")
            print(f"   Volatility:         {portfolio_metrics['volatility']:7.2%}")
            print(f"   Sharpe Ratio:       {portfolio_metrics['sharpe_ratio']:7.3f}")
            print(f"   Max Drawdown:       {portfolio_metrics['max_drawdown']:7.2%}")
            print(f"   Calmar Ratio:       {portfolio_metrics['calmar_ratio']:7.3f}")
            print(f"   Win Rate:           {portfolio_metrics['win_rate']:7.2%}")
            
            print(f"\n₿ BTC BUY & HOLD BENCHMARK:")
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
        
        # 4. Market Analysis Summary
        print("\n" + "="*60)
        print("4️⃣  MARKET ANALYSIS SUMMARY")
        print("="*60)
        
        # Correlation analysis
        correlations = returns.corr()
        print("🔗 Correlation Matrix (Top correlations):")
        
        # Find highest correlations
        corr_pairs = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr_pairs.append((
                    correlations.columns[i], 
                    correlations.columns[j], 
                    correlations.iloc[i, j]
                ))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for pair in corr_pairs[:5]:
            print(f"   {pair[0]:10} - {pair[1]:10}: {pair[2]:+6.3f}")
        
        # Volatility analysis
        print("\n📊 Volatility Analysis (Annualized):")
        volatilities = returns.std() * np.sqrt(252)
        for symbol in volatilities.index:
            vol = volatilities[symbol]
            risk_level = "High" if vol > 0.6 else "Medium" if vol > 0.3 else "Low"
            print(f"   {symbol:10}: {vol:6.2%} ({risk_level} Risk)")
        
        print("\n✅ Demonstration completed successfully!")
        print("\n🌐 To access the full web interface, run:")
        print("   cd frontend && streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_functionality())