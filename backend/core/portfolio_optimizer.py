import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from .data_manager import DataManager

class PortfolioOptimizer:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
        # Risk tolerance mappings
        self.risk_mappings = {
            'conservative': {'target_return': 0.08, 'max_volatility': 0.15, 'max_weight': 0.3},
            'moderate': {'target_return': 0.15, 'max_volatility': 0.25, 'max_weight': 0.4},
            'aggressive': {'target_return': 0.25, 'max_volatility': 0.4, 'max_weight': 0.6}
        }
    
    async def optimize(self, symbols: List[str], start_date: str, end_date: str, 
                      risk_tolerance: str, method: str = 'mean_variance') -> Dict:
        """Main optimization function"""
        try:
            # Get price data and calculate returns
            price_data = await self.data_manager.get_price_data(symbols, start_date, end_date)
            returns = self.data_manager.calculate_returns(price_data)
            
            if method == 'mean_variance':
                weights = self._mean_variance_optimization(returns, risk_tolerance)
            elif method == 'risk_parity':
                weights = self._risk_parity_optimization(returns)
            elif method == 'black_litterman':
                weights = self._black_litterman_optimization(returns, risk_tolerance)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'symbols': symbols,
                'weights': weights.tolist(),
                'expected_return': metrics['expected_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'method': method,
                'risk_tolerance': risk_tolerance
            }
            
        except Exception as e:
            raise Exception(f"Portfolio optimization failed: {str(e)}")
    
    def _mean_variance_optimization(self, returns: pd.DataFrame, risk_tolerance: str) -> np.ndarray:
        """Mean-variance optimization using CVXPY"""
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean() * 252  # Annualized returns
        Sigma = returns.cov() * 252  # Annualized covariance
        
        # Get risk parameters
        risk_params = self.risk_mappings[risk_tolerance]
        
        # Define optimization variables
        w = cp.Variable(n_assets)
        
        # Define objective (maximize return for given risk or minimize risk for given return)
        portfolio_return = mu.values @ w
        portfolio_risk = cp.quad_form(w, Sigma.values)
        
        # Objective: maximize Sharpe ratio approximation
        risk_free = 0.02  # 2% risk-free rate
        objective = cp.Maximize(portfolio_return - 0.5 * risk_params['target_return'] * portfolio_risk)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,  # Long-only
            w <= risk_params['max_weight'],  # Maximum weight per asset
            cp.sqrt(portfolio_risk) <= risk_params['max_volatility']  # Volatility constraint
        ]
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if w.value is None:
            # Fallback to equal weights if optimization fails
            return np.ones(n_assets) / n_assets
        
        return np.array(w.value)
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """Risk parity optimization - equal risk contribution"""
        n_assets = len(returns.columns)
        Sigma = returns.cov() * 252  # Annualized covariance
        
        def risk_budget_objective(weights, cov_matrix):
            """Objective function for risk parity"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = np.multiply(marginal_contrib, weights)
            
            # Risk parity: equal risk contribution
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.01, 0.4) for _ in range(n_assets))  # Min 1%, max 40%
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_budget_objective,
            x0,
            args=(Sigma.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
    
    def _black_litterman_optimization(self, returns: pd.DataFrame, risk_tolerance: str) -> np.ndarray:
        """Black-Litterman optimization with market views"""
        n_assets = len(returns.columns)
        
        # Market cap weights (simplified - equal weights as proxy)
        w_market = np.ones(n_assets) / n_assets
        
        # Historical parameters
        Sigma = returns.cov() * 252
        
        # Risk aversion parameter based on risk tolerance
        risk_aversion = {'conservative': 5, 'moderate': 3, 'aggressive': 1}[risk_tolerance]
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(Sigma.values, w_market)
        
        # Black-Litterman parameters
        tau = 0.05  # Uncertainty of prior
        
        # Views (simplified - no specific views, use equilibrium)
        P = np.eye(n_assets)  # Identity matrix (views on all assets)
        Q = pi  # View returns equal to equilibrium
        Omega = np.diag(np.diag(tau * Sigma.values))  # View uncertainty
        
        # Black-Litterman formula
        M1 = np.linalg.inv(tau * Sigma.values)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
        M3 = np.dot(np.linalg.inv(tau * Sigma.values), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        Sigma_bl = np.linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        w = cp.Variable(n_assets)
        portfolio_return = mu_bl @ w
        portfolio_risk = cp.quad_form(w, Sigma_bl)
        
        risk_params = self.risk_mappings[risk_tolerance]
        objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= risk_params['max_weight']
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return np.array(w.value) if w.value is not None else w_market
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict:
        """Calculate portfolio performance metrics"""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Annualized metrics
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (expected_return - 0.02) / volatility if volatility > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)  # 5% VaR
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        return {
            'expected_return': float(expected_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'var_95': float(var_95),
            'max_drawdown': float(max_drawdown)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()