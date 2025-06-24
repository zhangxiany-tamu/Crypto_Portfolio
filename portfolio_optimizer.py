import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple strategies and risk preferences
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer
        
        Args:
            returns: DataFrame of asset returns (daily)
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)
        
        # Calculate key statistics
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        self.std_devs = np.sqrt(np.diag(self.cov_matrix))
        self.sharpe_ratios = (self.mean_returns - risk_free_rate) / self.std_devs
    
    def mean_variance_optimization(self, 
                                 risk_tolerance: float = 0.5,
                                 target_return: Optional[float] = None,
                                 max_weight: float = 0.4) -> Dict:
        """
        Mean-Variance Optimization (Markowitz)
        
        Args:
            risk_tolerance: 0 (risk-averse) to 1 (risk-seeking)
            target_return: Target portfolio return (if None, optimize risk-return tradeoff)
            max_weight: Maximum weight per asset
        """
        try:
            # Define optimization variables
            w = cp.Variable(self.n_assets)
            
            # Portfolio return and risk
            portfolio_return = self.mean_returns.values @ w  
            portfolio_risk = cp.quad_form(w, self.cov_matrix.values)
            
            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,          # Long-only
                w <= max_weight  # Max weight constraint
            ]
            
            if target_return is not None:
                # Target return constraint
                constraints.append(portfolio_return >= target_return)
                objective = cp.Minimize(portfolio_risk)
            else:
                # Risk-return tradeoff based on risk tolerance
                # risk_tolerance: 0 = minimize risk, 1 = maximize return
                risk_aversion = 1 - risk_tolerance
                objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                weights = np.maximum(weights, 0)  # Ensure non-negative
                weights = weights / weights.sum()  # Normalize
                
                return self._create_portfolio_result(weights, "Mean-Variance Optimization")
            else:
                return self._fallback_portfolio("Mean-Variance failed")
                
        except Exception as e:
            return self._fallback_portfolio(f"Mean-Variance error: {str(e)[:50]}")
    
    def risk_parity_optimization(self, max_weight: float = 0.4) -> Dict:
        """
        Risk Parity Portfolio - Equal risk contribution from each asset
        """
        try:
            def risk_parity_objective(weights):
                # Portfolio variance
                portfolio_var = np.dot(weights, np.dot(self.cov_matrix.values, weights))
                
                # Risk contributions
                marginal_contrib = np.dot(self.cov_matrix.values, weights)
                risk_contrib = weights * marginal_contrib / portfolio_var
                
                # Minimize sum of squared deviations from equal risk
                target_risk = 1.0 / self.n_assets
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
            ]
            bounds = [(0.01, max_weight) for _ in range(self.n_assets)]  # User-defined max weight
            
            # Initial guess (equal weights)
            x0 = np.ones(self.n_assets) / self.n_assets
            
            # Optimize
            result = minimize(risk_parity_objective, x0, method='SLSQP', 
                            constraints=constraints, bounds=bounds)
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()  # Normalize
                return self._create_portfolio_result(weights, "Risk Parity")
            else:
                return self._fallback_portfolio("Risk Parity failed")
                
        except Exception as e:
            return self._fallback_portfolio(f"Risk Parity error: {str(e)[:50]}")
    
    def minimum_variance_optimization(self, max_weight: float = 0.4) -> Dict:
        """
        Minimum Variance Portfolio - Minimize portfolio volatility
        """
        try:
            # Define optimization variables
            w = cp.Variable(self.n_assets)
            
            # Minimize portfolio variance
            portfolio_risk = cp.quad_form(w, self.cov_matrix.values)
            
            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,          # Long-only
                w <= max_weight  # Max weight constraint
            ]
            
            # Solve optimization
            problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                
                return self._create_portfolio_result(weights, "Minimum Variance")
            else:
                return self._fallback_portfolio("Minimum Variance failed")
                
        except Exception as e:
            return self._fallback_portfolio(f"Minimum Variance error: {str(e)[:50]}")
    
    def maximum_sharpe_optimization(self, max_weight: float = 0.4) -> Dict:
        """
        Maximum Sharpe Ratio Portfolio
        """
        try:
            # Define optimization variables
            w = cp.Variable(self.n_assets)
            
            # Portfolio return and risk
            portfolio_return = self.mean_returns.values @ w
            portfolio_risk = cp.quad_form(w, self.cov_matrix.values)
            
            # Maximize Sharpe ratio (using a transformation to make it convex)
            # We minimize the reciprocal of Sharpe ratio
            excess_return = portfolio_return - self.risk_free_rate
            
            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,          # Long-only
                w <= max_weight, # Max weight constraint
                excess_return >= 0.01  # Ensure positive excess return
            ]
            
            # Objective: minimize risk for given return (equivalent to max Sharpe)
            objective = cp.Minimize(portfolio_risk)
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                
                return self._create_portfolio_result(weights, "Maximum Sharpe Ratio")
            else:
                return self._fallback_portfolio("Maximum Sharpe failed")
                
        except Exception as e:
            return self._fallback_portfolio(f"Maximum Sharpe error: {str(e)[:50]}")
    
    def momentum_portfolio(self, lookback_days: int = 60, max_weight: float = 0.4) -> Dict:
        """
        Momentum-based portfolio - Weight by recent performance
        """
        try:
            # Calculate momentum scores (recent returns)
            recent_returns = self.returns.tail(lookback_days).mean() * 252
            
            # Weight by positive momentum, with minimum allocation
            momentum_scores = np.maximum(recent_returns.values, 0.01)
            weights = momentum_scores / momentum_scores.sum()
            
            # Apply maximum weight constraint with proper redistribution
            weights = self._apply_weight_constraints(weights, max_weight)
            
            return self._create_portfolio_result(weights, "Momentum Portfolio")
            
        except Exception as e:
            return self._fallback_portfolio(f"Momentum error: {str(e)[:50]}")
    
    def equal_weight_portfolio(self, max_weight: float = 0.4) -> Dict:
        """
        Equal Weight Portfolio - Simple 1/N allocation (respecting max weight)
        """
        # Check if equal weight violates max weight constraint
        equal_weight = 1.0 / self.n_assets
        
        if equal_weight <= max_weight:
            # Standard equal weight is fine
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            # Equal weight violates constraint, use proper constraint handling
            weights = np.ones(self.n_assets) / self.n_assets
            weights = self._apply_weight_constraints(weights, max_weight)
        
        return self._create_portfolio_result(weights, "Equal Weight")
    
    def custom_risk_budget_portfolio(self, risk_budgets: List[float]) -> Dict:
        """
        Custom Risk Budget Portfolio
        
        Args:
            risk_budgets: List of risk budget allocations (must sum to 1)
        """
        try:
            if len(risk_budgets) != self.n_assets or abs(sum(risk_budgets) - 1.0) > 1e-6:
                return self._fallback_portfolio("Invalid risk budgets")
            
            def objective(weights):
                portfolio_var = np.dot(weights, np.dot(self.cov_matrix.values, weights))
                marginal_contrib = np.dot(self.cov_matrix.values, weights)
                risk_contrib = weights * marginal_contrib / portfolio_var
                
                # Minimize squared deviations from target risk budgets
                return np.sum((risk_contrib - np.array(risk_budgets)) ** 2)
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.01, 0.6) for _ in range(self.n_assets)]
            x0 = np.array(risk_budgets)  # Start with risk budgets as initial weights
            
            result = minimize(objective, x0, method='SLSQP', 
                            constraints=constraints, bounds=bounds)
            
            if result.success:
                weights = result.x
                weights = weights / weights.sum()
                return self._create_portfolio_result(weights, "Custom Risk Budget")
            else:
                return self._fallback_portfolio("Custom Risk Budget failed")
                
        except Exception as e:
            return self._fallback_portfolio(f"Custom Risk Budget error: {str(e)[:50]}")
    
    def _create_portfolio_result(self, weights: np.ndarray, strategy_name: str) -> Dict:
        """Create standardized portfolio result dictionary"""
        weights = np.array(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, self.mean_returns.values)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Calculate risk contributions
        marginal_contrib = np.dot(self.cov_matrix.values, weights)
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return {
            'strategy': strategy_name,
            'weights': dict(zip(self.asset_names, weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'risk_contributions': dict(zip(self.asset_names, risk_contrib)),
            'success': True
        }
    
    def _apply_weight_constraints(self, weights: np.ndarray, max_weight: float) -> np.ndarray:
        """
        Apply weight constraints with proper redistribution
        """
        weights = np.array(weights)
        
        # Check if constraint is mathematically possible
        if self.n_assets * max_weight < 1.0:
            # Impossible constraint - would need more than 100% to satisfy
            # Return equal weights at max_weight (will sum to less than 100%)
            return np.full(self.n_assets, max_weight) / (self.n_assets * max_weight)
        
        max_iterations = 10
        
        for _ in range(max_iterations):
            # Find weights that exceed the maximum (with small tolerance)
            exceeding = weights > max_weight + 1e-6
            
            if not np.any(exceeding):
                break  # All constraints satisfied
            
            # Cap the exceeding weights
            excess = np.sum(weights[exceeding] - max_weight)
            weights[exceeding] = max_weight
            
            # Redistribute excess to non-capped weights
            non_capped = ~exceeding
            if np.any(non_capped):
                # Distribute proportionally among non-capped weights
                current_non_capped_sum = np.sum(weights[non_capped])
                if current_non_capped_sum > 0:
                    scaling_factor = (current_non_capped_sum + excess) / current_non_capped_sum
                    weights[non_capped] *= scaling_factor
                else:
                    # All weights were capped, distribute equally
                    weights[non_capped] = excess / np.sum(non_capped)
            else:
                # All weights are at max - constraint is impossible
                # Just keep them all at max and normalize
                break
        
        # Final normalization and strict constraint enforcement
        weights = weights / weights.sum()
        
        # Force strict compliance - if any weight still exceeds, cap it
        exceeding_mask = weights > max_weight
        if np.any(exceeding_mask):
            weights[exceeding_mask] = max_weight
            weights = weights / weights.sum()
        
        return weights
    
    def _fallback_portfolio(self, error_msg: str) -> Dict:
        """Fallback to equal weight portfolio when optimization fails"""
        weights = np.ones(self.n_assets) / self.n_assets
        result = self._create_portfolio_result(weights, f"Equal Weight (Fallback)")
        result['success'] = False
        result['error'] = error_msg
        return result
    
    def get_efficient_frontier(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier points
        
        Returns:
            Tuple of (returns, volatilities) for efficient frontier
        """
        try:
            min_ret = self.mean_returns.min()
            max_ret = self.mean_returns.max()
            target_returns = np.linspace(min_ret, max_ret * 0.9, n_points)
            
            frontier_volatilities = []
            frontier_returns = []
            
            for target_return in target_returns:
                try:
                    result = self.mean_variance_optimization(target_return=target_return)
                    if result['success']:
                        frontier_returns.append(result['expected_return'])
                        frontier_volatilities.append(result['volatility'])
                except:
                    continue
            
            return np.array(frontier_returns), np.array(frontier_volatilities)
            
        except Exception:
            # Return empty arrays if calculation fails
            return np.array([]), np.array([])