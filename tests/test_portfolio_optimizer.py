import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.portfolio_optimizer import PortfolioOptimizer
from core.data_manager import DataManager

class TestPortfolioOptimizer:
    
    @pytest.fixture
    async def optimizer(self):
        # Mock data manager
        mock_data_manager = MagicMock(spec=DataManager)
        return PortfolioOptimizer(mock_data_manager)
    
    @pytest.fixture
    def sample_returns(self):
        # Create sample return data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        returns = pd.DataFrame({
            'BTC-USD': np.random.normal(0.001, 0.04, 252),
            'ETH-USD': np.random.normal(0.0008, 0.045, 252),
            'BNB-USD': np.random.normal(0.0005, 0.035, 252)
        }, index=dates)
        
        return returns
    
    @pytest.mark.asyncio
    async def test_mean_variance_optimization(self, optimizer, sample_returns):
        # Mock the data manager methods
        optimizer.data_manager.get_price_data = AsyncMock(
            return_value=sample_returns.cumsum()
        )
        optimizer.data_manager.calculate_returns = MagicMock(
            return_value=sample_returns
        )
        
        # Test optimization
        result = await optimizer.optimize(
            symbols=['BTC-USD', 'ETH-USD', 'BNB-USD'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            risk_tolerance='moderate',
            method='mean_variance'
        )
        
        # Assertions
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        
        # Check weights sum to 1
        assert abs(sum(result['weights']) - 1.0) < 1e-6
        
        # Check all weights are non-negative
        assert all(w >= 0 for w in result['weights'])
    
    @pytest.mark.asyncio
    async def test_risk_parity_optimization(self, optimizer, sample_returns):
        # Mock the data manager methods
        optimizer.data_manager.get_price_data = AsyncMock(
            return_value=sample_returns.cumsum()
        )
        optimizer.data_manager.calculate_returns = MagicMock(
            return_value=sample_returns
        )
        
        # Test risk parity optimization
        result = await optimizer.optimize(
            symbols=['BTC-USD', 'ETH-USD', 'BNB-USD'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            risk_tolerance='moderate',
            method='risk_parity'
        )
        
        # Assertions
        assert 'weights' in result
        assert len(result['weights']) == 3
        assert abs(sum(result['weights']) - 1.0) < 1e-6
    
    def test_calculate_portfolio_metrics(self, optimizer, sample_returns):
        weights = np.array([0.4, 0.4, 0.2])
        
        metrics = optimizer._calculate_portfolio_metrics(sample_returns, weights)
        
        # Check required metrics are present
        required_metrics = [
            'expected_return', 'volatility', 'sharpe_ratio', 
            'var_95', 'max_drawdown'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_risk_tolerance_mappings(self, optimizer):
        # Test that all risk tolerance levels are defined
        risk_levels = ['conservative', 'moderate', 'aggressive']
        
        for level in risk_levels:
            assert level in optimizer.risk_mappings
            mapping = optimizer.risk_mappings[level]
            
            assert 'target_return' in mapping
            assert 'max_volatility' in mapping
            assert 'max_weight' in mapping
            
            # Check reasonable ranges
            assert 0 < mapping['target_return'] < 1
            assert 0 < mapping['max_volatility'] < 1
            assert 0 < mapping['max_weight'] <= 1
    
    @pytest.mark.asyncio
    async def test_invalid_optimization_method(self, optimizer, sample_returns):
        optimizer.data_manager.get_price_data = AsyncMock(
            return_value=sample_returns.cumsum()
        )
        optimizer.data_manager.calculate_returns = MagicMock(
            return_value=sample_returns
        )
        
        # Test with invalid method
        with pytest.raises(Exception) as excinfo:
            await optimizer.optimize(
                symbols=['BTC-USD', 'ETH-USD'],
                start_date='2023-01-01',
                end_date='2023-12-31',
                risk_tolerance='moderate',
                method='invalid_method'
            )
        
        assert "Unknown optimization method" in str(excinfo.value)

if __name__ == "__main__":
    pytest.main([__file__])