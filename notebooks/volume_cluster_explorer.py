#!/usr/bin/env python3
"""
Volume Cluster Analysis

This script explores the high-volume clusters identified in the OHLCV data.
It can be executed directly or run as cells in a Jupyter environment.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import our modules
sys.path.append('..')
from src.parse_zst_ohlcv import load_ohlcv_from_jsonl
from src.volume_cluster import identify_volume_clusters

# Load the data
print("Loading JSONL data...")
jsonl_path = '../data/output.jsonl'
df = load_ohlcv_from_jsonl(jsonl_path)
print("Data loaded. Preview:")
print(df.head())

# Basic statistics
print(f"\nData range: {df.index.min()} to {df.index.max()}")
print(f"Number of records: {len(df)}")
print("\nSummary statistics:")
print(df.describe())

# Resample to 15-minute intervals
print("\nResampling to 15-minute intervals...")
df_15m = df.resample('15T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
print("Resampled data preview:")
print(df_15m.head())

# Identify volume clusters with different multipliers
print("\nIdentifying volume clusters...")
clusters_2x = identify_volume_clusters(df, volume_multiplier=2)
clusters_3x = identify_volume_clusters(df, volume_multiplier=3)
clusters_4x = identify_volume_clusters(df, volume_multiplier=4)

print(f"Number of clusters at 2x avg volume: {len(clusters_2x)}")
print(f"Number of clusters at 3x avg volume: {len(clusters_3x)}")
print(f"Number of clusters at 4x avg volume: {len(clusters_4x)}")

# Visualize volume distribution
print("\nVisualizing volume distribution...")
plt.figure(figsize=(15, 6))
plt.hist(df_15m['volume'], bins=50, alpha=0.7)
plt.axvline(df_15m['volume'].mean() * 3, color='r', linestyle='dashed', linewidth=2, label='3x Avg Volume')
plt.title('Distribution of 15-Minute Volume')
plt.xlabel('Volume')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('../data/volume_distribution.png')
plt.close()
print("Volume distribution saved to ../data/volume_distribution.png")

# Plot price and volume
print("\nPlotting price and volume with clusters...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Price plot
ax1.plot(df_15m.index, df_15m['close'], label='Close Price')
for idx in clusters_3x.index:
    ax1.axvline(idx, color='r', alpha=0.3)
ax1.set_title('Price with Volume Clusters Highlighted')
ax1.set_ylabel('Price')
ax1.legend()

# Volume plot
ax2.bar(df_15m.index, df_15m['volume'], width=0.01, label='15m Volume')
ax2.bar(clusters_3x.index, clusters_3x['volume'], width=0.01, color='r', label='Cluster Volume')
ax2.axhline(df_15m['volume'].mean() * 3, color='g', linestyle='dashed', linewidth=2, label='3x Threshold')
ax2.set_title('Volume with Clusters Highlighted')
ax2.set_xlabel('Date')
ax2.set_ylabel('Volume')
ax2.legend()

plt.tight_layout()
plt.savefig('../data/price_volume_clusters.png')
plt.close()
print("Price and volume plot saved to ../data/price_volume_clusters.png")

# Function to analyze price behavior around clusters
def analyze_cluster_behavior(df_15m, clusters, periods_before=4, periods_after=8):
    results = []
    
    for idx in clusters.index:
        try:
            # Get the position in the original dataframe
            pos = df_15m.index.get_loc(idx)
            
            # Get periods before and after
            before_idx = max(0, pos - periods_before)
            after_idx = min(len(df_15m) - 1, pos + periods_after)
            
            # Get price data
            price_before = df_15m.iloc[before_idx:pos]['close'].values
            cluster_price = df_15m.iloc[pos]['close']
            price_after = df_15m.iloc[pos+1:after_idx+1]['close'].values
            
            # Calculate metrics
            price_change_before = ((cluster_price / price_before[0]) - 1) * 100 if len(price_before) > 0 else 0
            price_change_after = ((price_after[-1] / cluster_price) - 1) * 100 if len(price_after) > 0 else 0
            
            results.append({
                'timestamp': idx,
                'volume': df_15m.iloc[pos]['volume'],
                'price_before': price_before[0] if len(price_before) > 0 else None,
                'cluster_price': cluster_price,
                'price_after': price_after[-1] if len(price_after) > 0 else None,
                'price_change_before': price_change_before,
                'price_change_after': price_change_after,
                'direction': 'up' if price_change_after > 0 else 'down'
            })
        except Exception as e:
            print(f"Error analyzing cluster at {idx}: {e}")
    
    return pd.DataFrame(results)

# Analyze behavior around clusters
print("\nAnalyzing price behavior around clusters...")
cluster_analysis = analyze_cluster_behavior(df_15m, clusters_3x)
print("Cluster analysis preview:")
print(cluster_analysis.head())

# Save analysis to CSV
cluster_analysis.to_csv('../data/cluster_analysis.csv')
print("Cluster analysis saved to ../data/cluster_analysis.csv")

# Summarize cluster behavior
print("\nCluster behavior summary:")
print(f"Total clusters analyzed: {len(cluster_analysis)}")
print(f"Clusters followed by upward movement: {(cluster_analysis['direction'] == 'up').sum()} ({(cluster_analysis['direction'] == 'up').mean() * 100:.1f}%)")
print(f"Clusters followed by downward movement: {(cluster_analysis['direction'] == 'down').sum()} ({(cluster_analysis['direction'] == 'down').mean() * 100:.1f}%)")
print(f"Average price change before cluster: {cluster_analysis['price_change_before'].mean():.2f}%")
print(f"Average price change after cluster: {cluster_analysis['price_change_after'].mean():.2f}%")

print("\nAnalysis complete! Check the data directory for output files.") 