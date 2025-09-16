#!/usr/bin/env python3
"""
Debug Volume Calculation
Shows the exact volume calculation logic used in the backtest
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from volume_cluster import identify_volume_clusters

def debug_volume_calculation():
    """Debug the exact volume calculation used in backtest"""
    
    print("🔍 DEBUGGING VOLUME CALCULATION")
    print("=" * 50)
    
    # Create sample data to demonstrate the calculation
    print("📊 Creating sample data to demonstrate volume calculation...")
    
    # Create 1-minute data for 1 day (9:30 AM - 4:00 PM EST)
    start_time = datetime(2025, 9, 15, 9, 30)  # 9:30 AM
    end_time = datetime(2025, 9, 15, 16, 0)    # 4:00 PM
    
    # Generate 1-minute timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Create sample OHLCV data
    np.random.seed(42)
    base_price = 6000.0
    base_volume = 5000
    
    data = []
    for i, ts in enumerate(timestamps):
        # Generate realistic price movement
        price_change = np.random.normal(0, 0.0005)  # 0.05% volatility
        price = base_price * (1 + price_change)
        
        # Generate realistic volume (higher during market hours)
        if 9 <= ts.hour <= 16:
            volume = int(base_volume * (1 + np.random.exponential(0.3)))
        else:
            volume = int(base_volume * 0.1)
        
        # Create OHLCV bar
        open_price = price
        close_price = price * (1 + np.random.normal(0, 0.0002))
        high_price = max(open_price, close_price) + np.random.exponential(0.5)
        low_price = min(open_price, close_price) - np.random.exponential(0.5)
        
        data.append({
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        base_price = close_price  # Update base price for next bar
    
    # Create DataFrame
    df = pd.DataFrame(data, index=timestamps)
    
    print(f"✅ Created sample data: {len(df)} records")
    print(f"📊 Data range: {df.index.min()} to {df.index.max()}")
    print(f"📊 Volume range: {df['volume'].min():,} - {df['volume'].max():,}")
    print(f"📊 Average 1-minute volume: {df['volume'].mean():,.0f}")
    
    # Step 1: Identify volume clusters (15-minute resampled)
    print(f"\n🔍 STEP 1: Volume Cluster Detection (15-minute resampled)")
    print("-" * 60)
    
    clusters_df = identify_volume_clusters(df, volume_multiplier=4.0)
    
    if not clusters_df.empty:
        print(f"✅ Found {len(clusters_df)} volume clusters")
        print(f"📊 Cluster volume range: {clusters_df['volume'].min():,.0f} - {clusters_df['volume'].max():,.0f}")
        print(f"📊 Average cluster strength: {clusters_df['cluster_strength'].mean():.2f}")
        
        # Step 2: Calculate volume ratios (like backtest does)
        print(f"\n🎯 STEP 2: Volume Ratio Calculation (like backtest)")
        print("-" * 60)
        
        # This is what the backtest actually does:
        daily_avg_1min = df['volume'].mean()  # 1-minute average
        print(f"📊 Daily 1-minute average volume: {daily_avg_1min:,.0f}")
        
        # Calculate volume ratios for each cluster
        volume_ratios = []
        for idx, row in clusters_df.iterrows():
            cluster_volume = row['volume']  # 15-minute cluster volume
            volume_ratio = cluster_volume / daily_avg_1min
            volume_ratios.append(volume_ratio)
            
            print(f"  Cluster at {idx.strftime('%H:%M')}: {cluster_volume:,.0f} / {daily_avg_1min:,.0f} = {volume_ratio:.1f}x")
        
        print(f"\n📊 Volume ratios: {min(volume_ratios):.1f}x - {max(volume_ratios):.1f}x")
        print(f"📊 Average volume ratio: {np.mean(volume_ratios):.1f}x")
        
        # Step 3: Show the difference
        print(f"\n🔍 STEP 3: Key Insight")
        print("-" * 60)
        print(f"🎯 Volume clusters are identified using 15-minute resampled data")
        print(f"🎯 But volume ratios are calculated against 1-minute average")
        print(f"🎯 This creates much higher volume ratios than expected!")
        print(f"")
        print(f"📊 Example:")
        print(f"  - 15-min cluster volume: {clusters_df['volume'].iloc[0]:,.0f}")
        print(f"  - Daily 1-min average: {daily_avg_1min:,.0f}")
        print(f"  - Volume ratio: {clusters_df['volume'].iloc[0] / daily_avg_1min:.1f}x")
        print(f"  - This is MUCH higher than the 4.0x cluster threshold!")
        
    else:
        print("❌ No clusters found with 4.0x threshold")
        print("💡 This means the 4.0x threshold is very selective")
        print("💡 It requires 4x the daily 15-minute average volume")
        
        # Show what the threshold actually is
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        avg_15m_volume = df_15m['volume'].mean()
        threshold = avg_15m_volume * 4.0
        
        print(f"\n📊 15-minute analysis:")
        print(f"  - Average 15-min volume: {avg_15m_volume:,.0f}")
        print(f"  - 4.0x threshold: {threshold:,.0f}")
        print(f"  - Max 15-min volume: {df_15m['volume'].max():,.0f}")
        print(f"  - Threshold met: {df_15m['volume'].max() >= threshold}")

if __name__ == "__main__":
    debug_volume_calculation()

