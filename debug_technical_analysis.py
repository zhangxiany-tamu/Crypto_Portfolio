#!/usr/bin/env python3
"""
Debug script for technical analysis medium-term signals
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from technical_analysis import TechnicalAnalyzer
import traceback

def debug_technical_analysis():
    """Debug the technical analysis medium-term signals"""
    
    # Try both databases
    db_extended = '/Users/xianyangzhang/My Drive/Crypto_Portfolio/data/crypto_extended.db'
    db_main = '/Users/xianyangzhang/My Drive/Crypto_Portfolio/crypto_data.db'
    
    # First try extended database
    try:
        conn = sqlite3.connect(db_extended)
        query = """
        SELECT date, close
        FROM extended_prices 
        WHERE symbol = 'BTC-USD'
        AND date >= date('now', '-365 days')
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Using extended database: {db_extended}")
    except:
        # Fallback to main database
        conn = sqlite3.connect(db_main)
        query = """
        SELECT date, price as close
        FROM price_data 
        WHERE symbol = 'BTC-USD'
        AND date >= date('now', '-365 days')
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Using main database: {db_main}")
    
    print(f"=== DEBUG: Technical Analysis Medium-Term Signals ===")
    print(f"Total data points: {len(df)}")
    
    if len(df) == 0:
        print("ERROR: No data available!")
        return
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    if len(df) < 200:
        print(f"WARNING: Only {len(df)} days of data available")
    
    # Convert to pandas Series with datetime index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    price_data = df['close']
    
    print(f"\nPrice data info:")
    print(f"- Shape: {price_data.shape}")
    print(f"- Latest price: ${price_data.iloc[-1]:,.2f}")
    print(f"- Has NaN values: {price_data.isna().sum()}")
    
    # Initialize analyzer
    analyzer = TechnicalAnalyzer(price_data)
    
    # Calculate indicators
    print(f"\n=== Calculating Indicators ===")
    try:
        indicators = analyzer.calculate_all_indicators()
        print(f"Successfully calculated {len(indicators)} indicators")
        
        # Check which indicators have valid values
        print(f"\nIndicator availability:")
        for name, indicator in indicators.items():
            valid_count = len(indicator.dropna())
            latest_val = indicator.iloc[-1] if valid_count > 0 else "N/A"
            print(f"  {name}: {valid_count} valid values, latest: {latest_val}")
            
    except Exception as e:
        print(f"ERROR calculating indicators: {e}")
        traceback.print_exc()
        return
    
    # Get latest values for debugging
    print(f"\n=== Getting Latest Values ===")
    current_price = price_data.iloc[-1]
    latest_values = {}
    
    for key, indicator in indicators.items():
        if len(indicator.dropna()) > 0:
            latest_values[key] = indicator.iloc[-1]
    
    print(f"Current price: ${current_price:,.2f}")
    print(f"Latest values available: {len(latest_values)}")
    
    # Debug medium-term analysis step by step
    print(f"\n=== DEBUG: Medium-Term Analysis Step by Step ===")
    
    try:
        # Manually run the medium-term analysis with debug info
        signals = []
        
        # 1. Medium-term moving averages
        print(f"\n1. Checking medium-term moving averages...")
        if all(k in latest_values for k in ['sma_21', 'sma_50']):
            sma_21 = latest_values['sma_21']
            sma_50 = latest_values['sma_50']
            
            print(f"   SMA-21: ${sma_21:,.2f}")
            print(f"   SMA-50: ${sma_50:,.2f}")
            print(f"   Current price: ${current_price:,.2f}")
            
            if current_price > sma_21 > sma_50:
                signals.append(('bullish', 'Bullish medium-term trend', 0.7))
                print(f"   → BULLISH: Price > SMA-21 > SMA-50")
            elif current_price < sma_21 < sma_50:
                signals.append(('bearish', 'Bearish medium-term trend', 0.7))
                print(f"   → BEARISH: Price < SMA-21 < SMA-50")
            else:
                signals.append(('neutral', 'Mixed medium-term signals', 0.3))
                print(f"   → NEUTRAL: Mixed conditions")
        else:
            print(f"   Missing SMA data: SMA-21 available: {'sma_21' in latest_values}, SMA-50 available: {'sma_50' in latest_values}")
        
        # 2. Bollinger Bands
        print(f"\n2. Checking Bollinger Bands...")
        if all(k in latest_values for k in ['bb_upper', 'bb_lower', 'bb_position']):
            bb_pos = latest_values['bb_position']
            bb_upper = latest_values['bb_upper']
            bb_lower = latest_values['bb_lower']
            
            print(f"   BB Upper: ${bb_upper:,.2f}")
            print(f"   BB Lower: ${bb_lower:,.2f}")
            print(f"   BB Position: {bb_pos:.3f}")
            
            if bb_pos < 0.2:
                signals.append(('bullish', 'Near Bollinger lower band', 0.6))
                print(f"   → BULLISH: Near lower band")
            elif bb_pos > 0.8:
                signals.append(('bearish', 'Near Bollinger upper band', 0.6))
                print(f"   → BEARISH: Near upper band")
            else:
                signals.append(('neutral', 'Within Bollinger bands', 0.3))
                print(f"   → NEUTRAL: Within bands")
        else:
            missing = [k for k in ['bb_upper', 'bb_lower', 'bb_position'] if k not in latest_values]
            print(f"   Missing BB data: {missing}")
        
        # 3. CCI
        print(f"\n3. Checking CCI...")
        if 'cci' in latest_values:
            cci = latest_values['cci']
            print(f"   CCI: {cci:.2f}")
            
            if cci < -100:
                signals.append(('bullish', 'CCI oversold', 0.5))
                print(f"   → BULLISH: CCI oversold")
            elif cci > 100:
                signals.append(('bearish', 'CCI overbought', 0.5))
                print(f"   → BEARISH: CCI overbought")
            else:
                signals.append(('neutral', 'CCI neutral range', 0.2))
                print(f"   → NEUTRAL: CCI in neutral range")
        else:
            print(f"   CCI not available")
        
        # 4. Momentum
        print(f"\n4. Checking momentum...")
        if 'momentum_20' in latest_values:
            momentum = latest_values['momentum_20']
            print(f"   20-day momentum: {momentum:.4f} ({momentum*100:.2f}%)")
            
            if momentum > 0.05:
                signals.append(('bullish', 'Positive momentum', 0.5))
                print(f"   → BULLISH: Positive momentum")
            elif momentum < -0.05:
                signals.append(('bearish', 'Negative momentum', 0.5))
                print(f"   → BEARISH: Negative momentum")
            else:
                signals.append(('neutral', 'Neutral momentum', 0.2))
                print(f"   → NEUTRAL: Neutral momentum")
        else:
            print(f"   Momentum not available")
        
        # 5. EMA crossover
        print(f"\n5. Checking EMA crossover...")
        if all(k in latest_values for k in ['ema_12', 'ema_26']):
            ema_12 = latest_values['ema_12']
            ema_26 = latest_values['ema_26']
            
            print(f"   EMA-12: ${ema_12:,.2f}")
            print(f"   EMA-26: ${ema_26:,.2f}")
            
            if ema_12 > ema_26:
                signals.append(('bullish', 'EMA bullish crossover', 0.6))
                print(f"   → BULLISH: EMA-12 > EMA-26")
            elif ema_12 < ema_26:
                signals.append(('bearish', 'EMA bearish crossover', 0.6))
                print(f"   → BEARISH: EMA-12 < EMA-26")
        else:
            missing = [k for k in ['ema_12', 'ema_26'] if k not in latest_values]
            print(f"   Missing EMA data: {missing}")
        
        # 6. Price position relative to SMA-21
        print(f"\n6. Checking price position relative to SMA-21...")
        if 'sma_21' in latest_values:
            sma_21 = latest_values['sma_21']
            price_deviation = (current_price - sma_21) / sma_21
            
            print(f"   Price deviation from SMA-21: {price_deviation:.4f} ({price_deviation*100:.2f}%)")
            
            if price_deviation > 0.02:
                signals.append(('bullish', 'Price above SMA-21', 0.4))
                print(f"   → BULLISH: Price significantly above SMA-21")
            elif price_deviation < -0.02:
                signals.append(('bearish', 'Price below SMA-21', 0.4))
                print(f"   → BEARISH: Price significantly below SMA-21")
            else:
                signals.append(('neutral', 'Price near SMA-21', 0.2))
                print(f"   → NEUTRAL: Price near SMA-21")
        else:
            print(f"   SMA-21 not available")
        
        print(f"\n=== Signals Summary ===")
        print(f"Total signals generated: {len(signals)}")
        
        if signals:
            for i, (signal_type, reason, weight) in enumerate(signals):
                print(f"  {i+1}. {signal_type.upper()} - {reason} (weight: {weight})")
        else:
            print("  No signals generated - checking fallback...")
            
            # Fallback logic
            if len(price_data) >= 2:
                recent_change = (price_data.iloc[-1] - price_data.iloc[-2]) / price_data.iloc[-2]
                print(f"  Recent change: {recent_change:.4f} ({recent_change*100:.2f}%)")
                
                if recent_change > 0.01:
                    signals.append(('bullish', 'Recent price increase', 0.3))
                    print(f"  → FALLBACK BULLISH: Recent price increase")
                elif recent_change < -0.01:
                    signals.append(('bearish', 'Recent price decrease', 0.3))
                    print(f"  → FALLBACK BEARISH: Recent price decrease")
                else:
                    signals.append(('neutral', 'Price relatively stable', 0.2))
                    print(f"  → FALLBACK NEUTRAL: Price relatively stable")
            else:
                signals.append(('neutral', 'Basic price analysis', 0.2))
                print(f"  → FALLBACK NEUTRAL: Basic price analysis")
        
        # Test consolidation
        print(f"\n=== Testing Signal Consolidation ===")
        consolidated = analyzer._consolidate_signals(signals, 'medium_term')
        print(f"Consolidated result:")
        for key, value in consolidated.items():
            print(f"  {key}: {value}")
        
        # Compare with actual method call
        print(f"\n=== Comparing with Actual Method Call ===")
        actual_result = analyzer._analyze_medium_term_signals(current_price, latest_values)
        print(f"Actual method result:")
        for key, value in actual_result.items():
            print(f"  {key}: {value}")
        
        # Full analysis
        print(f"\n=== Full Analysis ===")
        full_analysis = analyzer.analyze_signals()
        print(f"Medium-term from full analysis:")
        for key, value in full_analysis['medium_term'].items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"ERROR in medium-term analysis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_technical_analysis()