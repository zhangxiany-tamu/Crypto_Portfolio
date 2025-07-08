#!/usr/bin/env python3
"""
Test different scenarios that could cause "Insufficient data for analysis"
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from technical_analysis import TechnicalAnalyzer

def test_insufficient_data_scenarios():
    """Test various scenarios that could cause insufficient data"""
    
    print("=== TESTING INSUFFICIENT DATA SCENARIOS ===\n")
    
    # Test 1: Very limited data (less than required for indicators)
    print("1. Testing with very limited data (10 days)...")
    limited_dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    limited_prices = pd.Series([100 + i for i in range(10)], index=limited_dates)
    
    ta_limited = TechnicalAnalyzer(limited_prices)
    signals_limited = ta_limited.analyze_signals()
    
    print(f"   Medium-term signal: {signals_limited['medium_term']['signal']}")
    print(f"   Reasons: {signals_limited['medium_term']['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in signals_limited['medium_term']['reasons'])}")
    print()
    
    # Test 2: Empty data
    print("2. Testing with empty data...")
    empty_prices = pd.Series([], dtype=float)
    empty_prices.index = pd.DatetimeIndex([])
    
    try:
        ta_empty = TechnicalAnalyzer(empty_prices)
        signals_empty = ta_empty.analyze_signals()
        print(f"   Medium-term signal: {signals_empty['medium_term']['signal']}")
        print(f"   Reasons: {signals_empty['medium_term']['reasons']}")
    except Exception as e:
        print(f"   Exception: {e}")
    print()
    
    # Test 3: Data with NaN values
    print("3. Testing with NaN values...")
    nan_dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    nan_prices = pd.Series([100 + i if i % 10 != 0 else np.nan for i in range(100)], index=nan_dates)
    
    ta_nan = TechnicalAnalyzer(nan_prices)
    signals_nan = ta_nan.analyze_signals()
    
    print(f"   Medium-term signal: {signals_nan['medium_term']['signal']}")
    print(f"   Reasons: {signals_nan['medium_term']['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in signals_nan['medium_term']['reasons'])}")
    print()
    
    # Test 4: Data with insufficient length for specific indicators
    print("4. Testing with insufficient data for SMA-200...")
    short_dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
    short_prices = pd.Series([100 + i*0.1 for i in range(150)], index=short_dates)
    
    ta_short = TechnicalAnalyzer(short_prices)
    indicators_short = ta_short.calculate_all_indicators()
    
    # Check what indicators are missing
    missing_indicators = []
    for key, indicator in indicators_short.items():
        if len(indicator.dropna()) == 0:
            missing_indicators.append(key)
    
    print(f"   Missing indicators: {missing_indicators}")
    
    signals_short = ta_short.analyze_signals()
    print(f"   Medium-term signal: {signals_short['medium_term']['signal']}")
    print(f"   Reasons: {signals_short['medium_term']['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in signals_short['medium_term']['reasons'])}")
    print()
    
    # Test 5: Force empty signals in medium-term analysis
    print("5. Testing forced empty signals scenario...")
    
    # Create a custom analyzer that forces empty signals
    class TestAnalyzer(TechnicalAnalyzer):
        def _analyze_medium_term_signals(self, current_price, latest_values):
            # Force empty signals by returning empty list
            return self._consolidate_signals([], 'medium_term')
    
    normal_dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    normal_prices = pd.Series([100 + i*0.1 for i in range(365)], index=normal_dates)
    
    ta_forced = TestAnalyzer(normal_prices)
    signals_forced = ta_forced.analyze_signals()
    
    print(f"   Medium-term signal: {signals_forced['medium_term']['signal']}")
    print(f"   Reasons: {signals_forced['medium_term']['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in signals_forced['medium_term']['reasons'])}")
    print()
    
    # Test 6: Test with indicators that have no valid values
    print("6. Testing with indicators that have no valid values...")
    
    class TestAnalyzer2(TechnicalAnalyzer):
        def calculate_all_indicators(self):
            indicators = super().calculate_all_indicators()
            # Force all indicators to have no valid values
            for key in indicators:
                indicators[key] = pd.Series([np.nan] * len(self.price_data), index=self.price_data.index)
            return indicators
    
    ta_no_indicators = TestAnalyzer2(normal_prices)
    signals_no_indicators = ta_no_indicators.analyze_signals()
    
    print(f"   Medium-term signal: {signals_no_indicators['medium_term']['signal']}")
    print(f"   Reasons: {signals_no_indicators['medium_term']['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in signals_no_indicators['medium_term']['reasons'])}")
    print()
    
    # Test 7: Test what happens with very recent data (might not have enough history)
    print("7. Testing with very recent data (last 30 days)...")
    recent_dates = pd.date_range(start='2024-12-01', periods=30, freq='D')
    recent_prices = pd.Series([100 + i*0.1 for i in range(30)], index=recent_dates)
    
    ta_recent = TechnicalAnalyzer(recent_prices)
    signals_recent = ta_recent.analyze_signals()
    
    print(f"   Medium-term signal: {signals_recent['medium_term']['signal']}")
    print(f"   Reasons: {signals_recent['medium_term']['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in signals_recent['medium_term']['reasons'])}")
    print()

if __name__ == "__main__":
    test_insufficient_data_scenarios()