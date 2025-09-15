#!/usr/bin/env python3
"""
Databento API Connector for V6 Real-Time Trading System
Handles real-time market data streaming and historical data requests
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
import logging
import os

# Note: You'll need to install databento: pip install databento
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logging.warning("Databento not installed. Using simulation mode.")

logger = logging.getLogger(__name__)

class DatabentoConnector:
    """Handles Databento API connections and data streaming"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.live_client = None
        self.is_connected = False
        self.data_callback = None
        
        # ES futures contract mapping - use specific contract symbols
        self.contract_mapping = {
            'ES JUN25': 'ESM6',  # ES Jun 2026
            'ES SEP25': 'ESU5',  # ES Sep 2025 (current front month)
            'ES DEC25': 'ESZ5',  # ES Dec 2025
            'ES MAR26': 'ESH6',  # ES Mar 2026
            'ES.FUT': 'ESZ5',    # Default to Dec 2025
        }
        
    async def initialize(self, api_key: str = None):
        """Initialize Databento clients"""
        if not DATABENTO_AVAILABLE:
            logger.warning("⚠️  Databento not available - running in simulation mode")
            return False
            
        try:
            if api_key:
                self.api_key = api_key
                
            if not self.api_key:
                logger.error("❌ Databento API key required")
                return False
                
            # Initialize historical data client
            self.client = db.Historical(key=self.api_key)
            
            # Initialize live data client  
            self.live_client = db.Live(key=self.api_key)
            
            logger.info("✅ Databento clients initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Databento: {e}")
            return False
    
    def set_data_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Set callback function for receiving real-time data"""
        self.data_callback = callback
    
    async def get_historical_data(self, contract: str, start_date: datetime, 
                                 end_date: datetime = None) -> pd.DataFrame:
        """Get historical OHLCV data for backtesting/initialization"""
        if not DATABENTO_AVAILABLE or not self.client:
            return self._simulate_historical_data(contract, start_date, end_date)
        
        try:
            # Map contract to Databento symbol
            symbol = self.contract_mapping.get(contract, 'ESU5')
            
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"📈 Fetching historical data: {symbol} from {start_date} to {end_date}")
            
            # Request OHLCV data
            data = self.client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=[symbol],
                schema="ohlcv-1m",  # 1-minute bars
                start=start_date,
                end=end_date,
                stype_in="raw_symbol",  # Use "raw_symbol" for specific contract symbols
                stype_out="instrument_id"
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            # Standardize column names
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Historical data retrieved: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical data: {e}")
            return self._simulate_historical_data(contract, start_date, end_date)
    
    def start_live_stream(self, contract: str):
        """Start live data streaming for specified contract using synchronous approach"""
        if not DATABENTO_AVAILABLE or not self.live_client:
            logger.info("📡 Databento not available - no live data stream")
            return
        
        try:
            # Use ES.FUT with parent stype for live data
            symbol = "ES.FUT"
            
            logger.info(f"📡 Starting live stream for {symbol}")
            
            # Subscribe to live data using the working approach
            self.live_client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="parent",
                symbols=[symbol]
            )
            
            # Add callback to process data
            def process_record(record):
                # Skip system messages and other non-data records
                if not hasattr(record, 'schema'):
                    return
                    
                if record.schema == "ohlcv-1m":
                    # Convert to DataFrame format
                    data_row = pd.DataFrame({
                        'timestamp': [pd.to_datetime(record.ts_event, unit='ns')],
                        'open': [record.open / 1e9],  # Databento price scaling
                        'high': [record.high / 1e9],
                        'low': [record.low / 1e9],
                        'close': [record.close / 1e9],
                        'volume': [record.volume]
                    })
                    data_row.set_index('timestamp', inplace=True)
                    
                    # Send to callback
                    if self.data_callback:
                        self.data_callback(data_row)
            
            self.live_client.add_callback(process_record)
            self.live_client.start()
            
            self.is_connected = True
            logger.info("✅ Live data stream started successfully")
                        
        except Exception as e:
            logger.error(f"❌ Live stream error: {e}")
            self.is_connected = False
    
    def _simulate_historical_data(self, contract: str, start_date: datetime, 
                                 end_date: datetime = None) -> pd.DataFrame:
        """Simulate historical data for testing"""
        if end_date is None:
            end_date = datetime.now()
        
        # Generate minute-by-minute data
        date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Simulate ES futures price around 6000
        base_price = 6000.0
        np.random.seed(42)  # For reproducible simulation
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.001, len(date_range))  # 0.1% volatility per minute
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(date_range):
            if i == 0:
                open_price = prices[i]
            else:
                open_price = data[-1]['close']
            
            close_price = prices[i]
            high_price = max(open_price, close_price) + np.random.exponential(0.5)
            low_price = min(open_price, close_price) - np.random.exponential(0.5)
            
            # Simulate volume (higher during market hours)
            hour = timestamp.hour
            if 9 <= hour <= 16:  # Market hours
                base_volume = 5000
            else:
                base_volume = 1000
                
            volume = int(base_volume * (1 + np.random.exponential(0.5)))
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        logger.info(f"🔮 Simulated historical data: {len(df)} records")
        return df
    
    async def _simulate_live_stream(self, contract: str):
        """Simulate live data stream for testing"""
        logger.info(f"🔮 Simulating live stream for {contract}")
        
        base_price = 6010.0
        last_price = base_price
        
        while True:
            try:
                await asyncio.sleep(60)  # 1-minute intervals
                
                # Generate realistic price movement
                change_pct = np.random.normal(0, 0.002)  # 0.2% volatility
                new_price = last_price * (1 + change_pct)
                
                # Generate OHLCV bar
                open_price = last_price
                close_price = new_price
                high_price = max(open_price, close_price) + np.random.exponential(0.3)
                low_price = min(open_price, close_price) - np.random.exponential(0.3)
                
                # Simulate varying volume (sometimes high for clusters)
                if np.random.random() < 0.05:  # 5% chance of high volume
                    volume = int(np.random.uniform(15000, 30000))  # Volume cluster
                else:
                    volume = int(np.random.uniform(3000, 8000))    # Normal volume
                
                # Create data row
                current_time = datetime.now()
                data_row = pd.DataFrame({
                    'timestamp': [current_time],
                    'open': [round(open_price, 2)],
                    'high': [round(high_price, 2)],
                    'low': [round(low_price, 2)],
                    'close': [round(close_price, 2)],
                    'volume': [volume]
                })
                data_row.set_index('timestamp', inplace=True)
                
                # Send to callback
                if self.data_callback:
                    self.data_callback(data_row)
                
                last_price = new_price
                
            except Exception as e:
                logger.error(f"❌ Simulation error: {e}")
                break
    
    async def stop_stream(self):
        """Stop live data streaming"""
        self.is_connected = False
        if self.live_client:
            try:
                await self.live_client.stop()
            except Exception as e:
                logger.error(f"⚠️  Error stopping live client: {e}")
        logger.info("⏹️  Live stream stopped")
    
    def get_contract_info(self, contract: str) -> Dict[str, Any]:
        """Get contract specifications"""
        return {
            'contract': contract,
            'symbol': self.contract_mapping.get(contract, 'ESZ5'),
            'tick_size': 0.25,
            'tick_value': 12.50,
            'contract_size': 50,  # $50 per point
            'currency': 'USD',
            'exchange': 'CME'
        }

# Example usage and testing
async def test_databento_connector():
    """Test the Databento connector"""
    print("🧪 Testing Databento Connector")
    print("=" * 40)
    
    # Get API key from environment
    api_key = os.getenv('DATABENTO_API_KEY')
    
    connector = DatabentoConnector(api_key=api_key)
    
    # Test initialization
    initialized = await connector.initialize()
    print(f"Initialization: {'✅ Success' if initialized else '⚠️  Simulation Mode'}")
    
    # Test historical data
    start_date = datetime.now() - timedelta(days=1)
    hist_data = await connector.get_historical_data('ES JUN25', start_date)
    print(f"Historical data: {len(hist_data)} records")
    print(f"Price range: ${hist_data['low'].min():.2f} - ${hist_data['high'].max():.2f}")
    
    # Set up callback for live data
    def data_received(data):
        latest = data.iloc[-1]
        print(f"📊 Live data: {latest.name} | "
              f"${latest['close']:.2f} | "
              f"Vol: {latest['volume']:,}")
    
    connector.set_data_callback(data_received)
    
    # Test live stream (run for 5 minutes)
    print("\n📡 Starting live stream test (30 seconds)...")
    stream_task = asyncio.create_task(connector.start_live_stream('ES JUN25'))
    
    # Let it run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop the stream
    await connector.stop_stream()
    stream_task.cancel()
    
    print("✅ Databento connector test completed")

if __name__ == "__main__":
    asyncio.run(test_databento_connector()) 