#!/usr/bin/env python3
"""
Test the exact scenario that would cause empty signals
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from technical_analysis import TechnicalAnalyzer

def test_exact_failure():
    """Test the exact scenario that would cause empty signals"""
    
    print("=== TESTING EXACT FAILURE SCENARIO ===\n")
    
    # Test 1: Single data point (should cause failure)
    print("1. Testing with single data point...")
    single_dates = pd.date_range(start='2024-01-01', periods=1, freq='D')
    single_prices = pd.Series([100], index=single_dates)
    
    ta_single = TechnicalAnalyzer(single_prices)
    
    # Get latest values
    indicators = ta_single.calculate_all_indicators()
    latest_values = {}
    for key, indicator in indicators.items():
        if len(indicator.dropna()) > 0:
            latest_values[key] = indicator.iloc[-1]
    
    print(f"   Available indicators: {list(latest_values.keys())}")
    
    current_price = single_prices.iloc[-1]
    medium_term_result = ta_single._analyze_medium_term_signals(current_price, latest_values)
    
    print(f"   Medium-term signal: {medium_term_result['signal']}")
    print(f"   Reasons: {medium_term_result['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in medium_term_result['reasons'])}")
    print()
    
    # Test 2: No valid indicators and less than 2 data points
    print("2. Testing with no indicators and minimal data...")
    
    class TestAnalyzer(TechnicalAnalyzer):
        def calculate_all_indicators(self):
            # Force all indicators to be empty
            indicators = {}
            for key in ['sma_7', 'sma_21', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'ema_50', 
                       'rsi', 'macd_line', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 
                       'bb_position', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'atr', 
                       'momentum_10', 'momentum_20', 'roc_10', 'roc_20']:
                indicators[key] = pd.Series([np.nan] * len(self.price_data), index=self.price_data.index)
            return indicators
    
    ta_no_indicators = TestAnalyzer(single_prices)
    
    # Get latest values (should be empty)
    indicators_empty = ta_no_indicators.calculate_all_indicators()
    latest_values_empty = {}
    for key, indicator in indicators_empty.items():
        if len(indicator.dropna()) > 0:
            latest_values_empty[key] = indicator.iloc[-1]
    
    print(f"   Available indicators: {list(latest_values_empty.keys())}")
    
    current_price = single_prices.iloc[-1]
    medium_term_result = ta_no_indicators._analyze_medium_term_signals(current_price, latest_values_empty)
    
    print(f"   Medium-term signal: {medium_term_result['signal']}")
    print(f"   Reasons: {medium_term_result['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in medium_term_result['reasons'])}")
    print()
    
    # Test 3: Empty price data (should cause exception)
    print("3. Testing with empty price data...")
    empty_prices = pd.Series([], dtype=float)
    empty_prices.index = pd.DatetimeIndex([])
    
    try:
        ta_empty = TechnicalAnalyzer(empty_prices)
        indicators_empty = ta_empty.calculate_all_indicators()
        latest_values_empty = {}
        for key, indicator in indicators_empty.items():
            if len(indicator.dropna()) > 0:
                latest_values_empty[key] = indicator.iloc[-1]
        
        # This should fail because empty_prices.iloc[-1] will fail
        current_price = empty_prices.iloc[-1]
        medium_term_result = ta_empty._analyze_medium_term_signals(current_price, latest_values_empty)
        
        print(f"   Medium-term signal: {medium_term_result['signal']}")
        print(f"   Reasons: {medium_term_result['reasons']}")
        print(f"   Contains 'Limited'? {any('Limited' in r for r in medium_term_result['reasons'])}")
    except Exception as e:
        print(f"   Exception: {e}")
        print("   This would cause the app to show an error or default message")
    print()
    
    # Test 4: Force a specific condition where all checks fail
    print("4. Testing with forced condition where all checks fail...")
    
    class TestAnalyzer2(TechnicalAnalyzer):
        def _analyze_medium_term_signals(self, current_price, latest_values):
            # Simulate all checks failing by returning empty signals
            signals = []
            
            # Manually check each condition to see why they might fail
            print(f"     Checking SMA-21 and SMA-50: {'sma_21' in latest_values and 'sma_50' in latest_values}")
            print(f"     Checking Bollinger Bands: {all(k in latest_values for k in ['bb_upper', 'bb_lower', 'bb_position'])}")
            print(f"     Checking CCI: {'cci' in latest_values}")
            print(f"     Checking Momentum: {'momentum_20' in latest_values}")
            print(f"     Checking EMA: {all(k in latest_values for k in ['ema_12', 'ema_26'])}")
            print(f"     Checking SMA-21: {'sma_21' in latest_values}")
            print(f"     Price data length: {len(self.price_data)}")
            
            # If all checks fail, this would trigger the fallback
            if not signals:
                if len(self.price_data) >= 2:
                    recent_change = (self.price_data.iloc[-1] - self.price_data.iloc[-2]) / self.price_data.iloc[-2]
                    print(f"     Recent change: {recent_change}")
                    if recent_change > 0.01:
                        signals.append(('bullish', 'Recent price increase', 0.3))
                    elif recent_change < -0.01:
                        signals.append(('bearish', 'Recent price decrease', 0.3))
                    else:
                        signals.append(('neutral', 'Price relatively stable', 0.2))
                else:
                    signals.append(('neutral', 'Basic price analysis', 0.2))
            
            print(f"     Generated signals: {len(signals)}")
            return self._consolidate_signals(signals, 'medium_term')
    
    normal_dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    normal_prices = pd.Series([100 + i*0.1 for i in range(100)], index=normal_dates)
    
    ta_debug = TestAnalyzer2(normal_prices)
    
    # Get latest values
    indicators_debug = ta_debug.calculate_all_indicators()
    latest_values_debug = {}
    for key, indicator in indicators_debug.items():
        if len(indicator.dropna()) > 0:
            latest_values_debug[key] = indicator.iloc[-1]
    
    print(f"   Available indicators: {list(latest_values_debug.keys())}")
    
    current_price = normal_prices.iloc[-1]
    medium_term_result = ta_debug._analyze_medium_term_signals(current_price, latest_values_debug)
    
    print(f"   Medium-term signal: {medium_term_result['signal']}")
    print(f"   Reasons: {medium_term_result['reasons']}")
    print(f"   Contains 'Limited'? {any('Limited' in r for r in medium_term_result['reasons'])}")
    print()

if __name__ == "__main__":
    test_exact_failure()