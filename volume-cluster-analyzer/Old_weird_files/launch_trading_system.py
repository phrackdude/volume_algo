#!/usr/bin/env python3
"""
V6 Bayesian Trading System Launcher
Easy-to-use launcher for the real-time trading system
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def print_banner():
    """Print system banner"""
    print("🚀 V6 BAYESIAN REAL-TIME TRADING SYSTEM")
    print("=" * 50)
    print("Based on your extraordinary 99% performance improvement")
    print("Win Rate: 64.7% | Returns: 0.813% per trade")
    print("=" * 50)

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import pandas
        import numpy
    except ImportError as e:
        missing_deps.append("pandas/numpy")
    
    try:
        import databento
        print("✅ Databento SDK available")
    except ImportError:
        print("⚠️  Databento SDK not installed - will run in simulation mode")
        print("   Install with: pip install databento")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n📦 Install with: pip install -r requirements_realtime.txt")
        return False
    
    return True

def setup_environment():
    """Setup environment and data directories"""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check configuration
    config_file = Path("trading_config.env")
    if not config_file.exists():
        print("📄 Creating sample configuration file...")
        from config import create_sample_config_file
        create_sample_config_file()
        
        print("\n⚠️  SETUP REQUIRED:")
        print("1. Edit 'trading_config.env' with your Databento API key")
        print("2. Review and adjust trading parameters if needed")
        print("3. Run this launcher again")
        return False
    
    return True

async def start_trading_system():
    """Start the main trading system"""
    try:
        from real_time_trading_system import RealTimeTradingSystem
        from config import load_config_from_file, config
        
        # Load configuration
        load_config_from_file()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   {error}")
            return False
        
        # Print configuration summary
        config.print_config()
        
        # Ask for confirmation
        print("\n🎯 READY TO START TRADING SYSTEM")
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("⏹️  System start cancelled")
            return False
        
        # Start the system
        print("\n🚀 Starting V6 Bayesian Trading System...")
        system = RealTimeTradingSystem()
        await system.run_real_time_strategy()
        
    except KeyboardInterrupt:
        print("\n⏹️  System stopped by user")
        return True
    except Exception as e:
        print(f"❌ System error: {e}")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Start the trading system
    try:
        asyncio.run(start_trading_system())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 