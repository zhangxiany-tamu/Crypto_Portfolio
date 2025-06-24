import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from .data_manager import DataManager

class Backtester:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
        # Trading costs
        self.trading_fee = 0.001  # 0.1% trading fee
        self.tax_rate = 0.0  # Capital gains tax (can be customized)
        
    async def run_backtest(self, symbols: List[str], weights: List[float], 
                          start_date: str, end_date: str, 
                          initial_capital: float = 100000,
                          rebalance_freq: str = 'monthly') -> Dict:
        """Run comprehensive backtest with benchmark comparison"""
        
        try:
            # Get price data
            price_data = await self.data_manager.get_price_data(symbols, start_date, end_date)
            
            # Run portfolio backtest
            portfolio_results = self._backtest_portfolio(
                price_data, weights, initial_capital, rebalance_freq
            )
            
            # Run benchmark backtests
            btc_results = await self._backtest_btc_hold(start_date, end_date, initial_capital)
            equal_weight_results = self._backtest_equal_weight(price_data, initial_capital)
            
            # Calculate performance metrics
            portfolio_metrics = self._calculate_performance_metrics(portfolio_results)
            btc_metrics = self._calculate_performance_metrics(btc_results)
            equal_weight_metrics = self._calculate_performance_metrics(equal_weight_results)
            
            # Compare with benchmarks
            comparison = self._compare_strategies(
                portfolio_metrics, btc_metrics, equal_weight_metrics
            )
            
            return {
                'portfolio': {
                    'returns': portfolio_results.tolist(),
                    'metrics': portfolio_metrics,
                    'symbols': symbols,
                    'weights': weights
                },
                'benchmarks': {
                    'btc_hold': {
                        'returns': btc_results.tolist(),
                        'metrics': btc_metrics
                    },
                    'equal_weight': {
                        'returns': equal_weight_results.tolist(),
                        'metrics': equal_weight_metrics
                    }
                },
                'comparison': comparison,
                'dates': price_data.index.strftime('%Y-%m-%d').tolist()
            }
            
        except Exception as e:
            raise Exception(f"Backtest failed: {str(e)}")
    
    def _backtest_portfolio(self, price_data: pd.DataFrame, weights: List[float], 
                           initial_capital: float, rebalance_freq: str) -> pd.Series:
        """Backtest a portfolio with rebalancing"""
        
        returns = price_data.pct_change().fillna(0)
        portfolio_value = pd.Series(index=price_data.index, dtype=float)
        portfolio_value.iloc[0] = initial_capital
        
        # Convert weights to numpy array
        weights = np.array(weights)
        
        # Track position values and rebalancing
        current_value = initial_capital
        rebalance_dates = self._get_rebalance_dates(price_data.index, rebalance_freq)
        
        for i in range(1, len(price_data)):
            date = price_data.index[i]
            daily_returns = returns.iloc[i].values
            
            # Calculate portfolio return
            portfolio_return = np.dot(weights, daily_returns)
            
            # Apply trading costs on rebalancing days
            if date in rebalance_dates and i > 1:
                # Estimate turnover (simplified)
                turnover = 0.5  # Assume 50% turnover on rebalance
                trading_cost = current_value * turnover * self.trading_fee
                current_value -= trading_cost
            
            # Update portfolio value
            current_value *= (1 + portfolio_return)
            portfolio_value.iloc[i] = current_value
        
        return portfolio_value
    
    async def _backtest_btc_hold(self, start_date: str, end_date: str, 
                                initial_capital: float) -> pd.Series:
        """Backtest BTC buy and hold strategy"""
        
        btc_data = await self.data_manager.get_price_data(['BTC-USD'], start_date, end_date)
        btc_prices = btc_data['BTC-USD']
        
        # Initial purchase with trading fee
        initial_shares = (initial_capital * (1 - self.trading_fee)) / btc_prices.iloc[0]
        
        # Calculate portfolio value over time
        portfolio_value = initial_shares * btc_prices
        
        return portfolio_value
    
    def _backtest_equal_weight(self, price_data: pd.DataFrame, 
                              initial_capital: float) -> pd.Series:
        """Backtest equal weight strategy"""
        
        n_assets = len(price_data.columns)
        equal_weights = [1.0 / n_assets] * n_assets
        
        return self._backtest_portfolio(price_data, equal_weights, initial_capital, 'monthly')
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, freq: str) -> List:
        """Get rebalancing dates based on frequency"""
        
        if freq == 'daily':
            return dates.tolist()
        elif freq == 'weekly':
            return [d for d in dates if d.weekday() == 0]  # Mondays
        elif freq == 'monthly':
            return [d for d in dates if d.day <= 7 and d == dates[dates.month == d.month][0]]
        elif freq == 'quarterly':
            return [d for d in dates if d.month in [1, 4, 7, 10] and d.day <= 7]
        else:
            return []
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Convert to returns if values are given
        if returns.iloc[0] > 100:  # Assume these are portfolio values, not returns
            pct_returns = returns.pct_change().fillna(0)
        else:
            pct_returns = returns
        
        # Basic metrics
        total_return = (returns.iloc[-1] / returns.iloc[0] - 1) if returns.iloc[0] != 0 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = pct_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        var_95 = np.percentile(pct_returns, 5)
        cvar_95 = pct_returns[pct_returns <= var_95].mean()
        
        # Other metrics
        win_rate = (pct_returns > 0).sum() / len(pct_returns)
        best_day = pct_returns.max()
        worst_day = pct_returns.min()
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'win_rate': float(win_rate),
            'best_day': float(best_day),
            'worst_day': float(worst_day)
        }
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    def _compare_strategies(self, portfolio: Dict, btc: Dict, equal_weight: Dict) -> Dict:
        """Compare portfolio performance against benchmarks"""
        
        return {
            'vs_btc': {
                'return_diff': portfolio['total_return'] - btc['total_return'],
                'sharpe_diff': portfolio['sharpe_ratio'] - btc['sharpe_ratio'],
                'volatility_diff': portfolio['volatility'] - btc['volatility'],
                'max_dd_diff': portfolio['max_drawdown'] - btc['max_drawdown']
            },
            'vs_equal_weight': {
                'return_diff': portfolio['total_return'] - equal_weight['total_return'],
                'sharpe_diff': portfolio['sharpe_ratio'] - equal_weight['sharpe_ratio'],
                'volatility_diff': portfolio['volatility'] - equal_weight['volatility'],
                'max_dd_diff': portfolio['max_drawdown'] - equal_weight['max_drawdown']
            },
            'ranking': self._rank_strategies(portfolio, btc, equal_weight)
        }
    
    def _rank_strategies(self, portfolio: Dict, btc: Dict, equal_weight: Dict) -> Dict:
        """Rank strategies by different metrics"""
        
        strategies = {
            'Portfolio': portfolio,
            'BTC Hold': btc,
            'Equal Weight': equal_weight
        }
        
        rankings = {}
        metrics = ['total_return', 'sharpe_ratio', 'calmar_ratio']
        
        for metric in metrics:
            sorted_strategies = sorted(strategies.items(), 
                                     key=lambda x: x[1][metric], reverse=True)
            rankings[metric] = [name for name, _ in sorted_strategies]
        
        return rankings