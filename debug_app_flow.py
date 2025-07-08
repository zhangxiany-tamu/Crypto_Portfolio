#!/usr/bin/env python3
"""
Debug script to test the exact app flow for technical analysis
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_crypto_loader import EnhancedCryptoLoader
from technical_analysis import TechnicalAnalyzer

class AppDataManager:
    def __init__(self, crypto_loader):
        self.crypto_loader = crypto_loader
    
    def get_real_data(self, symbols, start_date, end_date):
        """Get real data same as app"""
        return self.crypto_loader.get_real_data(symbols, start_date, end_date)

def debug_app_flow():
    """Debug the exact app flow"""
    print("=== DEBUGGING APP FLOW ===")
    
    # Initialize the same way as the app (data_manager is directly EnhancedCryptoLoader)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    data_manager = EnhancedCryptoLoader(data_dir=data_dir)
    
    # Use the same date range as the app (1 year)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Test with BTC-USD (same as app)
    selected_coin = "BTC-USD"
    
    print(f"\nTesting with {selected_coin}...")
    
    try:
        # Get data the same way as the app
        coin_data = data_manager.get_real_data([selected_coin], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if coin_data is not None and not coin_data.empty:
            print(f"Got coin data: {coin_data.shape}")
            print(f"Columns: {coin_data.columns.tolist()}")
            
            coin_series = coin_data[selected_coin].dropna()
            print(f"Coin series shape after dropna: {coin_series.shape}")
            print(f"Coin series data type: {type(coin_series)}")
            print(f"Latest price: ${coin_series.iloc[-1]:,.2f}")
            
            # Initialize technical analyzer same as app
            ta = TechnicalAnalyzer(coin_series)
            
            # Calculate indicators and analyze signals same as app
            print("\nCalculating indicators...")
            indicators = ta.calculate_all_indicators()
            print(f"Calculated {len(indicators)} indicators")
            
            print("\nAnalyzing signals...")
            signals = ta.analyze_signals()
            print(f"Signal keys: {signals.keys()}")
            
            print("\nMedium-term signals:")
            medium_term = signals['medium_term']
            for key, value in medium_term.items():
                print(f"  {key}: {value}")
            
            print("\nGetting signal summary...")
            summary = ta.get_signal_summary()
            print(f"Summary keys: {summary.keys()}")
            
            # Check if the issue is in the reasons
            if 'reasons' in medium_term:
                print(f"\nMedium-term reasons:")
                for reason in medium_term['reasons']:
                    print(f"  - {reason}")
                    
                # Check if any reason contains "Limited" or "Insufficient"
                problem_reasons = [r for r in medium_term['reasons'] if 'Limited' in r or 'Insufficient' in r]
                if problem_reasons:
                    print(f"\nPROBLEM REASONS FOUND:")
                    for reason in problem_reasons:
                        print(f"  - {reason}")
                else:
                    print("\nNo problem reasons found.")
        else:
            print("ERROR: No coin data returned or empty data")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_app_flow()