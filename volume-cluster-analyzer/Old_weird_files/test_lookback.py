#!/usr/bin/env python3
"""
Test script for the lookback tool
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from lookback import HistoricalSignalAnalyzer

async def test_lookback():
    """Test the lookback functionality"""
    print("🧪 Testing Historical Signal Lookback Tool")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = HistoricalSignalAnalyzer()
        
        # Test with a short period (last 2 hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        print(f"📊 Testing with period: {start_time} to {end_time}")
        
        # Analyze historical signals
        signals = await analyzer.analyze_historical_period(start_time, end_time)
        
        # Print results
        analyzer.print_signal_summary(signals)
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_lookback())

