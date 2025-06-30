#!/usr/bin/env python3
"""
Enhanced cryptocurrency data loader with access to 60+ coins and 5+ years of data
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedCryptoLoader:
    """
    High-performance cryptocurrency data loader with extended historical data
    - 60+ cryptocurrencies
    - 5+ years of daily data (2020-2025)
    - Instant access from local SQLite database
    - Hybrid approach: local data + real-time API fetching for missing dates
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.extended_db = os.path.join(data_dir, "crypto_extended.db")
        self.original_db = os.path.join(data_dir, "crypto_historical.db")
        
        # Use the extended database (only remaining database)
        if os.path.exists(self.extended_db):
            self.db_path = self.extended_db
            self.db_type = "extended"
            print(f"🚀 Using EXTENDED database with 60+ cryptocurrencies")
        else:
            raise FileNotFoundError(
                f"Extended cryptocurrency database not found at {self.extended_db}. Please run download_extended_data.py first."
            )
        
        self._load_database_info()
    
    def _load_database_info(self):
        """Load information about the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use extended database table names
        table_name = "extended_prices"
        metadata_table = "extended_metadata"
        
        self.table_name = table_name
        
        # Load available symbols
        cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol")
        self.crypto_symbols = [row[0] for row in cursor.fetchall()]
        
        # Get database statistics
        cursor.execute(f'''
            SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(date), MAX(date) 
            FROM {table_name}
        ''')
        symbols_count, total_records, earliest_date, latest_date = cursor.fetchone()
        
        self.stats = {
            'symbols_count': symbols_count,
            'total_records': total_records,
            'earliest_date': earliest_date,
            'latest_date': latest_date,
            'years_coverage': (datetime.strptime(latest_date, '%Y-%m-%d') - 
                             datetime.strptime(earliest_date, '%Y-%m-%d')).days / 365.25
        }
        
        conn.close()
        
        print(f"✅ Loaded {symbols_count} cryptocurrencies")
        print(f"📅 Coverage: {earliest_date} to {latest_date} ({self.stats['years_coverage']:.1f} years)")
        print(f"📊 Total records: {total_records:,}")
    
    def get_real_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Ultra-fast retrieval of historical cryptocurrency data
        
        Args:
            symbols: List of cryptocurrency symbols (e.g., ['BTC-USD', 'ETH-USD'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with dates as index and symbols as columns (closing prices)
        """
        # Validate symbols
        available_symbols = [s for s in symbols if s in self.crypto_symbols]
        missing_symbols = [s for s in symbols if s not in self.crypto_symbols]
        
        if missing_symbols:
            print(f"⚠️  Symbols not available: {missing_symbols}")
            print(f"💡 Available symbols: {len(self.crypto_symbols)} total")
            
        if not available_symbols:
            print("❌ No valid symbols provided")
            print(f"🔍 Try from: {self.crypto_symbols[:10]}...")
            return pd.DataFrame()
        
        print(f"⚡ Loading {len(available_symbols)} symbols from {self.db_type} database...")
        start_time = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Build optimized query
            placeholders = ','.join(['?' for _ in available_symbols])
            query = f'''
                SELECT symbol, date, close
                FROM {self.table_name}
                WHERE symbol IN ({placeholders})
                  AND date >= ? 
                  AND date <= ?
                ORDER BY date, symbol
            '''
            
            params = available_symbols + [start_date, end_date]
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                print(f"❌ No data found for date range {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Pivot to get symbols as columns
            price_data = df.pivot(index='date', columns='symbol', values='close')
            price_data.index = pd.to_datetime(price_data.index)
            
            # Ensure all requested symbols are present
            for symbol in available_symbols:
                if symbol not in price_data.columns:
                    price_data[symbol] = np.nan
            
            # Reorder columns to match input order and forward fill
            price_data = price_data[available_symbols].fillna(method='ffill')
            
            load_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Loaded {price_data.shape[0]} days × {price_data.shape[1]} symbols in {load_time:.3f}s")
            
            return price_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available cryptocurrency symbols"""
        return self.crypto_symbols.copy()
    
    def get_symbol_info(self, symbol: str = None) -> pd.DataFrame:
        """Get detailed information about available symbols"""
        conn = sqlite3.connect(self.db_path)
        
        if symbol:
            query = f'''
                SELECT 
                    symbol,
                    COUNT(*) as total_days,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    ROUND(MIN(close), 4) as lowest_price,
                    ROUND(MAX(close), 2) as highest_price,
                    ROUND(AVG(close), 4) as avg_price
                FROM {self.table_name}
                WHERE symbol = ?
                GROUP BY symbol
            '''
            params = [symbol]
        else:
            query = f'''
                SELECT 
                    symbol,
                    COUNT(*) as total_days,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    ROUND(MIN(close), 4) as lowest_price,
                    ROUND(MAX(close), 2) as highest_price,
                    ROUND(AVG(close), 4) as avg_price
                FROM {self.table_name}
                GROUP BY symbol
                ORDER BY total_days DESC, symbol
            '''
            params = []
        
        df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            # Add years coverage
            df['years_coverage'] = df.apply(
                lambda row: (datetime.strptime(row['latest_date'], '%Y-%m-%d') - 
                           datetime.strptime(row['earliest_date'], '%Y-%m-%d')).days / 365.25, 
                axis=1
            ).round(1)
        
        conn.close()
        return df
    
    def _check_date_coverage(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[bool, str, str]:
        """
        Check if requested date range is fully covered by the database
        
        Returns:
            (is_fully_covered, db_latest_date, missing_start_date)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get the latest date available for the requested symbols
            placeholders = ','.join(['?' for _ in symbols])
            cursor.execute(f'''
                SELECT MAX(date) 
                FROM {self.table_name} 
                WHERE symbol IN ({placeholders})
            ''', symbols)
            
            db_latest_date = cursor.fetchone()[0]
            
            if not db_latest_date:
                # No data at all for these symbols
                return False, None, start_date
            
            # Convert dates for comparison
            db_latest = datetime.strptime(db_latest_date, '%Y-%m-%d').date()
            request_end = datetime.strptime(end_date, '%Y-%m-%d').date()
            request_start = datetime.strptime(start_date, '%Y-%m-%d').date()
            
            # Check if database covers the entire requested range
            if db_latest >= request_end:
                return True, db_latest_date, None
            else:
                # Database doesn't cover the full range
                missing_start = (db_latest + timedelta(days=1)).strftime('%Y-%m-%d')
                return False, db_latest_date, missing_start
                
        finally:
            conn.close()
    
    def _fetch_missing_data_from_api(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch missing data from CryptoCompare API for date ranges not in database
        """
        print(f"📡 Fetching missing data from API: {start_date} to {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            crypto_symbol = symbol.replace('-USD', '')
            print(f"   🔄 Fetching {symbol}...")
            
            try:
                # Calculate days needed
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_needed = (end_dt - start_dt).days + 1
                
                # CryptoCompare API call
                url = "https://min-api.cryptocompare.com/data/v2/histoday"
                params = {
                    'fsym': crypto_symbol,
                    'tsym': 'USD',
                    'limit': min(days_needed, 2000),  # API limit
                    'toTs': int(end_dt.timestamp())
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('Response') == 'Success' and 'Data' in data:
                        historical_data = data['Data']['Data']
                        
                        for point in historical_data:
                            if point['close'] > 0:  # Valid data
                                date = datetime.fromtimestamp(point['time']).date()
                                
                                # Include data within requested range, or most recent data if requesting future dates
                                if start_dt.date() <= date <= end_dt.date() or (end_dt.date() > datetime.now().date() and date <= datetime.now().date()):
                                    all_data.append({
                                        'symbol': symbol,
                                        'date': date.strftime('%Y-%m-%d'),
                                        'close': point['close']
                                    })
                        
                        print(f"   ✅ {symbol}: API data fetched")
                    else:
                        print(f"   ❌ {symbol}: API error - {data.get('Message', 'Unknown')}")
                else:
                    print(f"   ❌ {symbol}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {str(e)}")
            
            # Rate limiting
            time.sleep(0.5)
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Pivot to match expected format
            price_data = df.pivot(index='date', columns='symbol', values='close')
            price_data.index = pd.to_datetime(price_data.index)
            
            print(f"📡 API fetch complete: {price_data.shape[0]} days × {price_data.shape[1]} symbols")
            return price_data
        else:
            print("❌ No API data retrieved")
            return pd.DataFrame()
    
    def get_hybrid_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Hybrid data retrieval: combines local database with real-time API fetching
        
        This method:
        1. Checks if the requested date range is fully covered by local database
        2. If yes: returns local data only (ultra-fast)
        3. If no: fetches missing dates from API and combines with local data
        
        Args:
            symbols: List of cryptocurrency symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Complete DataFrame with all requested data
        """
        print(f"🔍 Checking coverage for {len(symbols)} symbols: {start_date} to {end_date}")
        
        # Check what data we have locally
        is_fully_covered, db_latest_date, missing_start = self._check_date_coverage(symbols, start_date, end_date)
        
        if is_fully_covered:
            # All data available locally - use super-fast local retrieval
            print(f"✅ Full coverage in database - using local data only")
            return self.get_real_data(symbols, start_date, end_date)
        else:
            print(f"⚠️  Partial coverage: DB latest = {db_latest_date}, Need until = {end_date}")
            
            # Get local data for covered period
            local_data = pd.DataFrame()
            if db_latest_date:
                print(f"📊 Fetching local data: {start_date} to {db_latest_date}")
                local_data = self.get_real_data(symbols, start_date, db_latest_date)
            
            # Get missing data from API
            if missing_start:
                api_data = self._fetch_missing_data_from_api(symbols, missing_start, end_date)
            else:
                api_data = self._fetch_missing_data_from_api(symbols, start_date, end_date)
            
            # Combine data
            if not local_data.empty and not api_data.empty:
                print(f"🔗 Combining local data ({local_data.shape}) + API data ({api_data.shape})")
                
                # Ensure same columns
                all_symbols = list(set(symbols))
                for symbol in all_symbols:
                    if symbol not in local_data.columns:
                        local_data[symbol] = np.nan
                    if symbol not in api_data.columns:
                        api_data[symbol] = np.nan
                
                # Combine and sort
                combined_data = pd.concat([local_data, api_data]).sort_index()
                combined_data = combined_data[all_symbols].fillna(method='ffill')
                
                print(f"✅ Hybrid data ready: {combined_data.shape[0]} days × {combined_data.shape[1]} symbols")
                return combined_data
                
            elif not local_data.empty:
                print(f"📊 Using local data only: {local_data.shape}")
                return local_data
                
            elif not api_data.empty:
                print(f"📡 Using API data only: {api_data.shape}")
                return api_data
                
            else:
                print(f"❌ No data available for requested range")
                return pd.DataFrame()
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data"""
        return price_data.pct_change().dropna()
    
    def get_latest_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get the most recent prices for symbols"""
        if symbols is None:
            symbols = self.crypto_symbols
        
        available_symbols = [s for s in symbols if s in self.crypto_symbols]
        
        conn = sqlite3.connect(self.db_path)
        
        placeholders = ','.join(['?' for _ in available_symbols])
        query = f'''
            SELECT symbol, close, date
            FROM {self.table_name}
            WHERE symbol IN ({placeholders})
              AND date = (
                  SELECT MAX(date) 
                  FROM {self.table_name}
                  WHERE symbol = {self.table_name}.symbol
              )
        '''
        
        df = pd.read_sql_query(query, conn, params=available_symbols)
        conn.close()
        
        result = {}
        for _, row in df.iterrows():
            result[row['symbol']] = {
                'price': row['close'],
                'date': row['date']
            }
        
        return result
    
    def get_performance_summary(self, symbols: List[str], periods: List[str] = None) -> pd.DataFrame:
        """Get performance summary for multiple time periods"""
        if periods is None:
            periods = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']
        
        available_symbols = [s for s in symbols if s in self.crypto_symbols]
        
        # Calculate end date and start dates for each period
        end_date = datetime.now().date()
        
        period_days = {
            '1M': 30, '3M': 90, '6M': 180, '1Y': 365, 
            '2Y': 730, '3Y': 1095, '5Y': 1825
        }
        
        results = []
        
        for symbol in available_symbols:
            row_data = {'symbol': symbol}
            
            for period in periods:
                if period in period_days:
                    start_date = end_date - timedelta(days=period_days[period])
                    
                    try:
                        data = self.get_real_data([symbol], 
                                                start_date.strftime('%Y-%m-%d'),
                                                end_date.strftime('%Y-%m-%d'))
                        
                        if not data.empty and len(data) > 1:
                            start_price = data[symbol].iloc[0]
                            end_price = data[symbol].iloc[-1]
                            
                            if start_price > 0:
                                performance = (end_price / start_price - 1) * 100
                                row_data[period] = round(performance, 1)
                            else:
                                row_data[period] = None
                        else:
                            row_data[period] = None
                    except:
                        row_data[period] = None
            
            results.append(row_data)
        
        return pd.DataFrame(results)
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        return {
            'database_type': self.db_type,
            'database_path': self.db_path,
            'symbols_count': self.stats['symbols_count'],
            'total_records': self.stats['total_records'],
            'date_range': f"{self.stats['earliest_date']} to {self.stats['latest_date']}",
            'years_coverage': round(self.stats['years_coverage'], 1),
            'available_symbols': self.crypto_symbols
        }
    
    def search_symbols(self, pattern: str) -> List[str]:
        """Search for symbols matching a pattern"""
        pattern = pattern.upper()
        return [s for s in self.crypto_symbols if pattern in s.upper()]

def test_enhanced_loader():
    """Test the enhanced crypto loader including hybrid functionality"""
    print("🧪 Testing EnhancedCryptoLoader with Hybrid Data Fetching")
    print("=" * 70)
    
    try:
        loader = EnhancedCryptoLoader()
        
        # Test 1: Basic functionality (local data only)
        print("\n1️⃣ Testing basic data loading (local data)...")
        sample_symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD']
        end_date = '2025-06-26'  # Date covered in database
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
        
        data = loader.get_real_data(sample_symbols, start_date, end_date)
        
        if not data.empty:
            print(f"✅ Loaded data: {data.shape}")
            latest_prices = data.iloc[-1]
            print(f"💰 Latest prices:")
            for symbol, price in latest_prices.items():
                print(f"   {symbol}: ${price:,.2f}")
        
        # Test 2: Hybrid functionality (local + API)
        print("\n2️⃣ Testing hybrid data loading (local + API)...")
        today = datetime.now().date().strftime('%Y-%m-%d')
        hybrid_start = (datetime.now().date() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"Requesting data from {hybrid_start} to {today}")
        hybrid_data = loader.get_hybrid_data(['BTC-USD', 'ETH-USD'], hybrid_start, today)
        
        if not hybrid_data.empty:
            print(f"✅ Hybrid data: {hybrid_data.shape}")
            latest_hybrid = hybrid_data.iloc[-1]
            print(f"💰 Latest hybrid prices:")
            for symbol, price in latest_hybrid.items():
                if not pd.isna(price):
                    print(f"   {symbol}: ${price:,.2f}")
        
        # Test 3: Pure API test (future dates)
        print("\n3️⃣ Testing pure API mode (future/recent dates)...")
        future_start = (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
        future_end = (datetime.now().date() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"Requesting future data from {future_start} to {future_end}")
        future_data = loader.get_hybrid_data(['BTC-USD'], future_start, future_end)
        
        if future_data.empty:
            print("✅ Correctly handled future dates (no data available)")
        else:
            print(f"📊 Future data: {future_data.shape}")
        
        # Test 4: Long-term analysis (local data only)
        print("\n4️⃣ Testing long-term data (2020-2025, local only)...")
        long_term_data = loader.get_hybrid_data(['BTC-USD', 'ETH-USD'], '2020-01-01', '2025-06-26')
        
        if not long_term_data.empty:
            print(f"✅ Long-term data: {long_term_data.shape}")
            
            # Calculate 5-year performance
            returns = loader.calculate_returns(long_term_data)
            total_returns = (1 + returns).prod() - 1
            
            print(f"📈 5-year performance:")
            for symbol in total_returns.index:
                perf = total_returns[symbol]
                print(f"   {symbol}: {perf:+.0%}")
        
        # Test 5: Database statistics
        print("\n5️⃣ Database statistics...")
        stats = loader.get_database_stats()
        print(f"✅ Database: {stats['database_type']}")
        print(f"📊 Symbols: {stats['symbols_count']}")
        print(f"📅 Coverage: {stats['years_coverage']} years")
        print(f"🔢 Records: {stats['total_records']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_loader()
    
    if success:
        print("\n🎉 EnhancedCryptoLoader with Hybrid Data Fetching is ready!")
        print("\n💡 Usage examples:")
        print("   from enhanced_crypto_loader import EnhancedCryptoLoader")
        print("   loader = EnhancedCryptoLoader()")
        print("   ")
        print("   # HYBRID: Get recent data (local + API for missing dates)")
        print("   recent_data = loader.get_hybrid_data(['BTC-USD', 'ETH-USD'], '2025-06-01', '2025-06-28')")
        print("   ")
        print("   # LOCAL ONLY: Get historical data (ultra-fast)")
        print("   historical = loader.get_real_data(['BTC-USD'], '2020-01-01', '2025-06-26')")
        print("   ")
        print("   # SMART: Automatically chooses best method")
        print("   smart_data = loader.get_hybrid_data(['BTC-USD'], '2024-01-01', '2025-06-28')")
    else:
        print("\n❌ EnhancedCryptoLoader has issues")