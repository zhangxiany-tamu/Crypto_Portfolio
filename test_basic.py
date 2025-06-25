#!/usr/bin/env python3
"""
Basic tests for the Crypto Portfolio Optimizer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test that all core modules can be imported"""
    try:
        from portfolio_optimizer import PortfolioOptimizer
        from robust_data_fetcher import RobustDataFetcher
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_data_fetcher_initialization():
    """Test RobustDataFetcher initialization"""
    from robust_data_fetcher import RobustDataFetcher
    
    fetcher = RobustDataFetcher()
    
    # Check that crypto symbols are defined
    assert hasattr(fetcher, 'crypto_symbols')
    assert len(fetcher.crypto_symbols) > 0
    assert 'BTC-USD' in fetcher.crypto_symbols
    assert 'ETH-USD' in fetcher.crypto_symbols
    
    # Check that mappings exist
    assert hasattr(fetcher, 'coingecko_map')
    assert len(fetcher.coingecko_map) > 0

def test_portfolio_optimizer_initialization():
    """Test PortfolioOptimizer initialization"""
    from portfolio_optimizer import PortfolioOptimizer
    
    # Create sample returns data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns = pd.DataFrame(
        np.random.randn(100, 3) * 0.02,
        index=dates,
        columns=['BTC-USD', 'ETH-USD', 'SOL-USD']
    )
    
    optimizer = PortfolioOptimizer(returns)
    
    # Check basic attributes
    assert optimizer.n_assets == 3
    assert len(optimizer.asset_names) == 3
    assert hasattr(optimizer, 'mean_returns')
    assert hasattr(optimizer, 'cov_matrix')

def test_crypto_symbols_consistency():
    """Test that crypto symbols are consistent across files"""
    from robust_data_fetcher import RobustDataFetcher
    
    fetcher = RobustDataFetcher()
    
    # Check that all symbols have corresponding mappings
    for symbol in fetcher.crypto_symbols[:10]:  # Test first 10
        assert symbol in fetcher.coingecko_map, f"Missing CoinGecko mapping for {symbol}"

def test_fallback_data_generation():
    """Test fallback data generation"""
    from robust_data_fetcher import RobustDataFetcher
    
    fetcher = RobustDataFetcher()
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    
    data = fetcher.get_fallback_data(symbols, start_date, end_date)
    
    # Check data structure
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert all(symbol in data.columns for symbol in symbols)
    
    # Check data quality
    for symbol in symbols:
        prices = data[symbol]
        assert prices.min() > 0  # All prices should be positive
        assert not prices.isna().all()  # Should have valid data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])