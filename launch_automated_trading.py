#!/usr/bin/env python3
"""
Launch Automated V6 Paper Trading System

Easy startup script for automated paper trading with realistic execution
"""

import os
import sys
import asyncio
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from automated_paper_trader import main

def check_environment():
    """Check that everything is set up correctly"""
    print("🔍 Environment Check:")
    
    # Check data directory
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✅ Created data directory")
    else:
        print("✅ Data directory exists")
    
    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
    else:
        print("⚠️  Virtual environment not detected")
    
    print()

if __name__ == "__main__":
    print("🤖 V6 AUTOMATED PAPER TRADING LAUNCHER")
    print("="*50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💰 Portfolio: $15,000 starting balance")
    print()
    
    check_environment()
    
    print("🚀 Starting automated paper trading...")
    print("   Features:")
    print("   - $15,000 portfolio simulation")
    print("   - Portfolio-based position sizing")
    print("   - Realistic 20-second execution delays")
    print("   - Offer-side fills (realistic slippage)")  
    print("   - Automatic Bayesian learning")
    print("   - Audio alerts for signals and closures")
    print("   - Continuous operation")
    print()
    print("💡 Tip: Let this run for a week to collect meaningful data!")
    print("📊 Use analyze_paper_trading.py to review portfolio progression")
    print()
    print("Press Ctrl+C to stop...")
    print("="*50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Automated trading stopped by user")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📊 Run 'python analyze_paper_trading.py' to see your portfolio results!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that all dependencies are installed and data directory exists") 