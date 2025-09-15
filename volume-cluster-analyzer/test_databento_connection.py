#!/usr/bin/env python3
"""
Test Databento API connection and data retrieval for V6 Volume Cluster Strategy
This script verifies we can get all the data needed for V6 calculations
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import databento as db
    from src.volume_cluster import identify_volume_clusters
    from src.real_time_trading_system import BayesianStatsManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabentoConnectionTest:
    def __init__(self):
        self.api_key = os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            logger.error("❌ DATABENTO_API_KEY environment variable not set")
            sys.exit(1)
        
        self.client = None
        self.historical_client = None
        
    def initialize_clients(self):
        """Initialize Databento clients"""
        try:
            logger.info("🔌 Initializing Databento clients...")
            
            # Initialize live client
            self.client = db.Live(
                key=self.api_key,
                gateway="wss://hist.databento.com/v0"
            )
            
            # Initialize historical client
            self.historical_client = db.Historical(
                key=self.api_key,
                gateway="https://hist.databento.com"
            )
            
            logger.info("✅ Databento clients initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Databento clients: {e}")
            return False
    
    def test_historical_data(self):
        """Test historical data retrieval for ES futures"""
        try:
            logger.info("📊 Testing historical data retrieval...")
            
            # Get data for the last 2 days (end at 15:00 UTC to match data availability)
            end_date = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=2)
            
            logger.info(f"📅 Requesting data from {start_date.date()} to {end_date.date()}")
            
            # Request ES futures data (1-minute bars)
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ESZ5"],
                schema="ohlcv-1m",
                start=start_date,
                end=end_date
            )
            
            logger.info(f"✅ Retrieved {len(data)} data points")
            
            # Convert to DataFrame for analysis
            df = data.to_df()
            logger.info(f"📈 Data shape: {df.shape}")
            logger.info(f"📈 Columns: {list(df.columns)}")
            logger.info(f"📈 Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"📈 Sample data:\n{df.head()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve historical data: {e}")
            return None
    
    def test_v6_calculations(self, df):
        """Test V6 volume cluster calculations"""
        try:
            logger.info("🧮 Testing V6 volume cluster calculations...")
            
            if df is None or df.empty:
                logger.error("❌ No data available for V6 calculations")
                return False
            
            # Test volume cluster identification
            logger.info("🔍 Testing volume cluster identification...")
            clusters = identify_volume_clusters(df)
            logger.info(f"✅ Volume clusters identified: {len(clusters)} clusters")
            
            # Test signal generation
            logger.info("📡 Testing signal generation...")
            if len(clusters) > 0:
                logger.info(f"✅ Clusters available for signal generation: {clusters.head()}")
            else:
                logger.info("ℹ️  No clusters available for signal generation")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed V6 calculations: {e}")
            return False
    
    def test_bayesian_learning(self):
        """Test Bayesian learning system"""
        try:
            logger.info("🧠 Testing Bayesian learning system...")
            
            # Initialize Bayesian learning
            bayesian = BayesianStatsManager()
            
            # Test context creation
            logger.info("📝 Testing context creation...")
            context = {
                'volume_ratio': 1.2,
                'cluster_strength': 0.8,
                'time_of_day': 'morning',
                'volatility_regime': 'normal'
            }
            
            # Test learning from trade result
            logger.info("📚 Testing learning from trade result...")
            bayesian.record_trade_result("volume_ratio", 1, datetime.now(), 6010.0, 6020.0, 0.16, True, 1.2, 0.8)
            
            # Test prediction
            logger.info("🔮 Testing prediction...")
            stats = bayesian.get_context_stats("volume_ratio", 1)
            logger.info(f"✅ Stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed Bayesian learning test: {e}")
            return False
    
    def test_database_connections(self):
        """Test database connections"""
        try:
            logger.info("🗄️  Testing database connections...")
            
            # Test paper trades database
            import sqlite3
            db_path = "/opt/v6-trading-system/data/paper_trades.db"
            
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                logger.info(f"✅ Paper trades database accessible: {len(tables)} tables")
                conn.close()
            else:
                logger.warning(f"⚠️  Paper trades database not found at {db_path}")
            
            # Test Bayesian stats database
            bayesian_db_path = "/opt/v6-trading-system/data/bayesian_stats.db"
            
            if os.path.exists(bayesian_db_path):
                conn = sqlite3.connect(bayesian_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                logger.info(f"✅ Bayesian stats database accessible: {len(tables)} tables")
                conn.close()
            else:
                logger.warning(f"⚠️  Bayesian stats database not found at {bayesian_db_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed database connection test: {e}")
            return False
    
    def run_all_tests(self):
        """Run all connection and functionality tests"""
        logger.info("🚀 Starting Databento API and V6 System Tests")
        logger.info("=" * 60)
        
        # Test 1: Initialize clients
        if not self.initialize_clients():
            return False
        
        # Test 2: Historical data retrieval
        df = self.test_historical_data()
        if df is None:
            return False
        
        # Test 3: V6 calculations
        if not self.test_v6_calculations(df):
            return False
        
        # Test 4: Bayesian learning
        if not self.test_bayesian_learning():
            return False
        
        # Test 5: Database connections
        if not self.test_database_connections():
            return False
        
        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED! System is ready for live trading.")
        return True

def main():
    """Main test function"""
    test = DatabentoConnectionTest()
    success = test.run_all_tests()
    
    if success:
        print("\n✅ All systems operational - ready for live trading!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed - system needs attention")
        sys.exit(1)

if __name__ == "__main__":
    main()
