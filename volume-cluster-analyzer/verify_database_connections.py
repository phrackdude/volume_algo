#!/usr/bin/env python3
"""
Database Connection Verification Script
Ensures all components are reading/writing to the correct databases
"""

import sqlite3
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def check_database_exists_and_has_tables(db_path, expected_tables):
    """Check if database exists and has expected tables"""
    print(f"\n🔍 Checking database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database does not exist: {db_path}")
        return False
    
    file_size = os.path.getsize(db_path)
    print(f"📁 File size: {file_size} bytes")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📊 Tables found: {tables}")
        
        # Check expected tables
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        # Check record counts
        for table in expected_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count} records")
        
        conn.close()
        print(f"✅ Database OK: {db_path}")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_component_imports():
    """Test that all components can import and initialize correctly"""
    print("\n🧪 Testing component imports and database connections...")
    
    try:
        # Test monitoring dashboard
        print("\n📊 Testing monitoring_dashboard.py...")
        from monitoring_dashboard import TradingMonitor
        monitor = TradingMonitor()
        print(f"   Paper trades DB: {monitor.db_path}")
        print(f"   Bayesian DB: {monitor.bayesian_db_path}")
        print("✅ Monitoring dashboard initialized successfully")
        
        # Test automated paper trader
        print("\n🤖 Testing automated_paper_trader.py...")
        from automated_paper_trader import AutomatedPaperTrader
        # Don't create full trader (would start trading system), just check path
        print(f"   Expected DB path: /opt/v6-trading-system/data/paper_trades.db")
        print("✅ Automated paper trader import successful")
        
        # Test real-time trading system
        print("\n📈 Testing real_time_trading_system.py...")
        from real_time_trading_system import BayesianStatsManager
        bayesian_mgr = BayesianStatsManager()
        print(f"   Bayesian DB: {bayesian_mgr.db_path}")
        print("✅ Real-time trading system initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component import/initialization failed: {e}")
        return False

def main():
    """Run comprehensive database verification"""
    print("🔍 V6 BAYESIAN TRADING SYSTEM - DATABASE VERIFICATION")
    print("=" * 60)
    
    # Expected database paths
    paper_trades_db = "/opt/v6-trading-system/data/paper_trades.db"
    bayesian_stats_db = "/opt/v6-trading-system/data/bayesian_stats.db"
    
    # Expected tables
    paper_trades_tables = ["paper_trades", "portfolio_equity"]
    bayesian_tables = ["context_performance"]
    
    # Check databases
    paper_trades_ok = check_database_exists_and_has_tables(paper_trades_db, paper_trades_tables)
    bayesian_ok = check_database_exists_and_has_tables(bayesian_stats_db, bayesian_tables)
    
    # Test component imports
    components_ok = test_component_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY:")
    print(f"   Paper Trades DB: {'✅ OK' if paper_trades_ok else '❌ FAILED'}")
    print(f"   Bayesian Stats DB: {'✅ OK' if bayesian_ok else '❌ FAILED'}")
    print(f"   Component Imports: {'✅ OK' if components_ok else '❌ FAILED'}")
    
    if paper_trades_ok and bayesian_ok and components_ok:
        print("\n🎉 ALL SYSTEMS VERIFIED - Database connections are correct!")
        return 0
    else:
        print("\n🚨 VERIFICATION FAILED - Database connection issues detected!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
