import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import sqlite3
import os
from pathlib import Path

class DataManager:
    def __init__(self, db_path: str = "data/crypto_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Initialize exchanges
        self.binance = ccxt.binance()
        
        # Common crypto symbols
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD',
            'SHIB-USD', 'UNI-USD', 'ATOM-USD', 'LINK-USD', 'LTC-USD'
        ]
    
    def _init_database(self):
        """Initialize SQLite database for caching price data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def get_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for given symbols"""
        try:
            # Try to get from cache first
            cached_data = self._get_cached_data(symbols, start_date, end_date)
            
            # Fetch missing data
            missing_symbols = []
            for symbol in symbols:
                if symbol not in cached_data or len(cached_data[symbol]) == 0:
                    missing_symbols.append(symbol)
            
            if missing_symbols:
                new_data = await self._fetch_fresh_data(missing_symbols, start_date, end_date)
                self._cache_data(new_data)
                cached_data.update(new_data)
            
            # Combine data into single DataFrame
            combined_data = pd.DataFrame()
            for symbol in symbols:
                if symbol in cached_data:
                    combined_data[symbol] = cached_data[symbol]['close']
            
            return combined_data.fillna(method='ffill').dropna()
            
        except Exception as e:
            raise Exception(f"Error fetching price data: {str(e)}")
    
    def _get_cached_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get cached price data from database"""
        conn = sqlite3.connect(self.db_path)
        cached_data = {}
        
        for symbol in symbols:
            query = '''
                SELECT date, open, high, low, close, volume 
                FROM price_data 
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                cached_data[symbol] = df
        
        conn.close()
        return cached_data
    
    async def _fetch_fresh_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch fresh data from Yahoo Finance"""
        fresh_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Rename columns to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    fresh_data[symbol] = data
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return fresh_data
    
    def _cache_data(self, data: Dict[str, pd.DataFrame]):
        """Cache price data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol, df in data.items():
            for date, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 
                    date.strftime('%Y-%m-%d'),
                    row.get('open', 0),
                    row.get('high', 0), 
                    row.get('low', 0),
                    row.get('close', 0),
                    row.get('volume', 0)
                ))
        
        conn.commit()
        conn.close()
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of available crypto symbols"""
        return self.crypto_symbols
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'change_24h': info.get('regularMarketChangePercent', 0)
            }
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data"""
        return price_data.pct_change().dropna()
    
    def calculate_volatility(self, returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized