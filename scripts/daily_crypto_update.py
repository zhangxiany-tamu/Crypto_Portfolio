#!/usr/bin/env python3
"""
Daily Crypto Database Update Script

This script automatically fetches the latest cryptocurrency data and updates
the local SQLite database. Designed to run daily via GitHub Actions.
"""

import sqlite3
import requests
import time
import json
from datetime import datetime, timedelta, date
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/daily_update.log', mode='a') if os.path.exists('logs') else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DailyCryptoUpdater:
    def __init__(self):
        self.db_path = 'data/crypto_extended.db'
        self.api_base_url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        self.request_delay = 0.5  # Delay between API requests to respect rate limits
        
        # List of cryptocurrencies to update (matches your existing database)
        self.crypto_symbols = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'AVAX', 'SHIB',
            'MATIC', 'LTC', 'UNI', 'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'VET', 'ICP',
            'FIL', 'TRX', 'ETC', 'HBAR', 'APT', 'NEAR', 'QNT', 'IMX', 'MANA', 'SAND',
            'CRO', 'AAVE', 'GRT', 'LDO', 'STX', 'FTM', 'AXS', 'FLOW', 'XTZ', 'CHZ',
            'EOS', 'THETA', 'BSV', 'KCS', 'HNT', 'RUNE', 'EGLD', 'ZEC', 'MKR', 'SNX',
            'COMP', 'SUSHI', 'YFI', 'BAT', 'ENJ', 'ZIL', 'WAVES', 'ICX', 'REN', 'SC',
            'ONE', 'HOT', 'QTUM', 'ZRX', 'OMG', 'ANT'
        ]
        
    def get_latest_date_in_db(self, symbol):
        """Get the latest date for a symbol in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(date) FROM extended_prices 
                WHERE symbol = ?
            """, (f"{symbol}-USD",))
            
            result = cursor.fetchone()
            conn.close()
            
            if result[0]:
                return datetime.strptime(result[0], '%Y-%m-%d').date()
            else:
                # If no data exists, start from 2020-01-01
                return date(2020, 1, 1)
                
        except Exception as e:
            logger.error(f"Error getting latest date for {symbol}: {str(e)}")
            return date(2020, 1, 1)
    
    def fetch_crypto_data(self, symbol, from_date, to_date):
        """Fetch crypto data from CryptoCompare API"""
        try:
            # Convert dates to timestamps
            from_ts = int(datetime.combine(from_date, datetime.min.time()).timestamp())
            to_ts = int(datetime.combine(to_date, datetime.min.time()).timestamp())
            
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': min(2000, (to_date - from_date).days + 1),
                'toTs': to_ts
            }
            
            logger.info(f"Fetching data for {symbol} from {from_date} to {to_date}")
            
            response = requests.get(self.api_base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data['Response'] == 'Success':
                historical_data = data['Data']['Data']
                
                # Process the data
                processed_data = []
                for point in historical_data:
                    if point['close'] > 0:  # Valid data
                        point_date = datetime.fromtimestamp(point['time']).date()
                        
                        # Only include data within requested range and after the latest DB date
                        if from_date <= point_date <= to_date:
                            processed_data.append({
                                'symbol': f"{symbol}-USD",
                                'date': point_date.strftime('%Y-%m-%d'),
                                'open': point.get('open', point['close']),
                                'high': point.get('high', point['close']),
                                'low': point.get('low', point['close']),
                                'close': point['close'],
                                'volume': point.get('volumeto', 0)
                            })
                
                logger.info(f"✅ {symbol}: Retrieved {len(processed_data)} data points")
                return processed_data
            else:
                logger.warning(f"❌ {symbol}: API error - {data.get('Message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ {symbol}: Network error - {str(e)}")
            return []
        except Exception as e:
            logger.error(f"❌ {symbol}: Unexpected error - {str(e)}")
            return []
    
    def update_database(self, data_list):
        """Update the database with new data"""
        if not data_list:
            return 0
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or replace data
            cursor.executemany("""
                INSERT OR REPLACE INTO extended_prices 
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [(d['symbol'], d['date'], d['open'], d['high'], d['low'], d['close'], d['volume']) for d in data_list])
            
            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()
            
            logger.info(f"✅ Updated database with {rows_affected} records")
            return rows_affected
            
        except Exception as e:
            logger.error(f"❌ Database update error: {str(e)}")
            return 0
    
    def run_daily_update(self):
        """Main function to run the daily update"""
        logger.info("🚀 Starting daily crypto database update")
        
        # Get yesterday's date (since today's data might not be available yet)
        yesterday = date.today() - timedelta(days=1)
        
        total_updates = 0
        successful_updates = 0
        
        for symbol in self.crypto_symbols:
            try:
                # Get the latest date in database
                latest_date = self.get_latest_date_in_db(symbol)
                
                # Determine the date range to fetch
                if latest_date >= yesterday:
                    logger.info(f"ℹ️ {symbol}: Already up to date (latest: {latest_date})")
                    continue
                
                # Fetch missing data from the day after latest_date to yesterday
                from_date = latest_date + timedelta(days=1)
                to_date = yesterday
                
                # Fetch data from API
                new_data = self.fetch_crypto_data(symbol, from_date, to_date)
                
                if new_data:
                    # Update database
                    rows_updated = self.update_database(new_data)
                    if rows_updated > 0:
                        total_updates += rows_updated
                        successful_updates += 1
                        logger.info(f"✅ {symbol}: Added {rows_updated} new records")
                    else:
                        logger.warning(f"⚠️ {symbol}: No records were added to database")
                else:
                    logger.warning(f"⚠️ {symbol}: No new data available")
                
                # Rate limiting
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Update failed - {str(e)}")
                continue
        
        # Summary
        logger.info(f"📊 Update completed:")
        logger.info(f"   • Symbols processed: {len(self.crypto_symbols)}")
        logger.info(f"   • Successful updates: {successful_updates}")
        logger.info(f"   • Total records added: {total_updates}")
        
        return total_updates > 0
    
    def verify_database_integrity(self):
        """Verify database integrity after updates"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check total record count
            cursor.execute("SELECT COUNT(*) FROM extended_prices")
            total_records = cursor.fetchone()[0]
            
            # Check date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM extended_prices")
            min_date, max_date = cursor.fetchone()
            
            # Check symbols count
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM extended_prices")
            unique_symbols = cursor.fetchone()[0]
            
            conn.close()
            
            logger.info(f"📊 Database integrity check:")
            logger.info(f"   • Total records: {total_records:,}")
            logger.info(f"   • Date range: {min_date} to {max_date}")
            logger.info(f"   • Unique symbols: {unique_symbols}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database integrity check failed: {str(e)}")
            return False

def main():
    """Main entry point"""
    updater = DailyCryptoUpdater()
    
    try:
        # Run the daily update
        has_updates = updater.run_daily_update()
        
        # Verify database integrity
        updater.verify_database_integrity()
        
        if has_updates:
            logger.info("✅ Daily update completed successfully with new data")
            sys.exit(0)
        else:
            logger.info("ℹ️ Daily update completed - no new data to add")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Daily update failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()