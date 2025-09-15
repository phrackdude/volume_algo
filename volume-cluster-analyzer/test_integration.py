#!/usr/bin/env python3
"""
Test script for V6 Bayesian Real-Time Trading System Integration
Tests the integration of live data, V6 strategy, and market hours detection
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_time_trading_system import RealTimeTradingSystem
from config import config, load_config_from_file
import logging

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_system_initialization():
    """Test system initialization and configuration"""
    print("🧪 Testing System Initialization")
    print("=" * 50)
    
    # Load configuration
    load_config_from_file()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   {error}")
        return False
    
    # Print configuration
    config.print_config()
    
    # Initialize system
    try:
        system = RealTimeTradingSystem()
        print("✅ System initialized successfully")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

async def test_market_hours_detection():
    """Test market hours detection"""
    print("\n🧪 Testing Market Hours Detection")
    print("=" * 50)
    
    system = RealTimeTradingSystem()
    
    # Test current market status
    is_open = system.is_market_hours()
    current_time = datetime.now(system.est_tz)
    
    print(f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market status: {'OPEN' if is_open else 'CLOSED'}")
    print(f"Market hours: {system.market_open_time} - {system.market_close_time}")
    
    return True

async def test_databento_connection():
    """Test Databento API connection"""
    print("\n🧪 Testing Databento Connection")
    print("=" * 50)
    
    system = RealTimeTradingSystem()
    
    try:
        connected = await system.connect_to_databento()
        if connected:
            print("✅ Databento connection successful")
            
            # Test historical data
            from datetime import timedelta
            start_date = datetime.now() - timedelta(days=1)
            hist_data = await system.databento_connector.get_historical_data(
                system.current_contract or "ES JUN25", 
                start_date
            )
            
            if not hist_data.empty:
                print(f"✅ Historical data retrieved: {len(hist_data)} records")
                print(f"   Price range: ${hist_data['low'].min():.2f} - ${hist_data['high'].max():.2f}")
            else:
                print("⚠️  No historical data retrieved")
            
            return True
        else:
            print("❌ Databento connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Databento connection error: {e}")
        return False

async def test_v6_strategy():
    """Test V6 Bayesian strategy components"""
    print("\n🧪 Testing V6 Bayesian Strategy")
    print("=" * 50)
    
    system = RealTimeTradingSystem()
    
    # Test Bayesian manager
    try:
        # Test with no historical data (should return conservative defaults)
        multiplier, diagnostics = system.bayesian_manager.calculate_bayesian_multiplier("modal_bin", 0)
        print(f"✅ Bayesian multiplier (no data): {multiplier:.2f}x")
        print(f"   Diagnostics: {diagnostics}")
        
        # Test modal bin context calculation
        modal_position = 0.1  # 10% from modal price
        context_value = system.get_modal_bin_context(modal_position)
        print(f"✅ Modal bin context: {modal_position:.1%} -> bin {context_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ V6 strategy test failed: {e}")
        return False

async def test_volume_cluster_detection():
    """Test volume cluster detection with simulated data"""
    print("\n🧪 Testing Volume Cluster Detection")
    print("=" * 50)
    
    system = RealTimeTradingSystem()
    
    try:
        # Create simulated data with a volume cluster
        import pandas as pd
        import numpy as np
        
        # Generate test data
        timestamps = pd.date_range(start=datetime.now(), periods=30, freq='1min')
        base_price = 6010.0
        
        # Normal volume data
        normal_data = []
        for i, ts in enumerate(timestamps[:-1]):
            price = base_price + np.random.normal(0, 2)
            normal_data.append({
                'timestamp': ts,
                'open': price,
                'high': price + np.random.uniform(0, 1),
                'low': price - np.random.uniform(0, 1),
                'close': price + np.random.normal(0, 0.5),
                'volume': np.random.randint(3000, 8000)
            })
        
        # Add volume cluster with much higher volume
        cluster_ts = timestamps[-1]
        cluster_price = base_price + np.random.normal(0, 2)
        normal_data.append({
            'timestamp': cluster_ts,
            'open': cluster_price,
            'high': cluster_price + 2,
            'low': cluster_price - 2,
            'close': cluster_price + 1,
            'volume': 50000  # Very high volume cluster (should be >4x average)
        })
        
        # Create DataFrame
        test_data = pd.DataFrame(normal_data)
        test_data.set_index('timestamp', inplace=True)
        
        # Test cluster detection
        cluster = system.detect_volume_cluster(test_data)
        
        if cluster:
            print(f"✅ Volume cluster detected!")
            print(f"   Volume ratio: {cluster.volume_ratio:.1f}x")
            print(f"   Signal strength: {cluster.signal_strength:.3f}")
            print(f"   Direction: {cluster.direction}")
            print(f"   Entry price: ${cluster.entry_price:.2f}")
            
            # Test recommendation generation
            recommendation = system.generate_trading_recommendation(cluster)
            print(f"✅ Trading recommendation generated:")
            print(f"   Action: {recommendation.action}")
            print(f"   Quantity: {recommendation.quantity}")
            print(f"   Confidence: {recommendation.confidence:.1%}")
            print(f"   Bayesian multiplier: {recommendation.bayesian_multiplier:.2f}x")
            
            return True
        else:
            print("⚠️  No volume cluster detected in test data")
            return False
            
    except Exception as e:
        print(f"❌ Volume cluster detection test failed: {e}")
        return False

async def run_integration_test():
    """Run complete integration test"""
    print("🚀 V6 BAYESIAN TRADING SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Market Hours Detection", test_market_hours_detection),
        ("Databento Connection", test_databento_connection),
        ("V6 Strategy", test_v6_strategy),
        ("Volume Cluster Detection", test_volume_cluster_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready for live trading.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before live trading.")
    
    return passed == len(results)

if __name__ == "__main__":
    try:
        success = asyncio.run(run_integration_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
