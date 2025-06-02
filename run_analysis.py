from src.parse_zst_ohlcv import decompress_zst_file, load_ohlcv_from_jsonl
from src.volume_cluster import identify_volume_clusters
import os
import subprocess
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Paths
# Update path to use the file in the data directory
zst_path = 'data/glbx-mdp3-20240602-20250601.ohlcv-1m.dbn.zst'
jsonl_path = 'data/output.jsonl'

try:
    # Step 1: Decompress
    print("Decompressing .zst...")
    decompress_zst_file(zst_path, jsonl_path)

    # Check file info
    print(f"\nDecompressed file info:")
    print(f"Size: {os.path.getsize(jsonl_path)} bytes")
    # Try to determine file type
    print("File type:")
    try:
        file_type = subprocess.check_output(['file', jsonl_path], text=True)
        print(file_type)
    except:
        print("Could not determine file type")

    # Step 2: Load Data
    print("\nLoading data...")
    df = load_ohlcv_from_jsonl(jsonl_path)
    print("\nData preview:")
    print(df.head())
    
    # Data shape and stats
    print(f"\nDataset shape: {df.shape}")
    print("\nSummary statistics:")
    print(df.describe())

    # Step 3: Identify clusters
    print("\nIdentifying clusters...")
    clusters = identify_volume_clusters(df)
    
    # Display clusters
    print("\nClusters preview:")
    if clusters.empty:
        print("No clusters found with the default threshold.")
    else:
        print(clusters.head())
        # Save results for inspection
        clusters.to_csv("data/volume_clusters.csv")
        print(f"\nVolume clusters saved to data/volume_clusters.csv")
    
    # Create a basic volume histogram for visualization
    plt.figure(figsize=(10, 6))
    df_15m = df.resample('15min').agg({'volume': 'sum'})
    volumes = df_15m['volume'].values
    
    plt.hist(volumes, bins=30)
    avg_volume = df_15m['volume'].mean()
    threshold = 3 * avg_volume
    plt.axvline(threshold, color='r', linestyle='--', label=f'3x Threshold ({threshold:.0f})')
    
    # If we have any clusters, mark them on the histogram
    if not clusters.empty:
        for vol in clusters['volume']:
            plt.axvline(vol, color='g', alpha=0.3)
    
    plt.title('Distribution of 15-Minute Volume')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('data/volume_distribution.png')
    print(f"Volume histogram saved to data/volume_distribution.png")
    
    # Create a candlestick chart with volume
    plt.figure(figsize=(12, 8))
    
    # Price subplot
    ax1 = plt.subplot(2, 1, 1)
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Simple price plot
    ax1.plot(df_15m.index, df_15m['close'], label='Close Price')
    
    # Mark cluster times on price chart
    if not clusters.empty:
        for idx in clusters.index:
            ax1.axvline(idx, color='r', alpha=0.3)
    
    ax1.set_title('Price with Volume Clusters')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Volume subplot
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    df_15m_vol = df.resample('15min').agg({'volume': 'sum'})
    ax2.bar(df_15m_vol.index, df_15m_vol['volume'], width=0.01, color='b', alpha=0.6)
    
    # Highlight cluster volumes
    if not clusters.empty:
        ax2.bar(clusters.index, clusters['volume'], width=0.01, color='r', alpha=0.8)
    
    ax2.axhline(threshold, color='r', linestyle='--', label=f'3x Threshold ({threshold:.0f})')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('data/price_volume_chart.png')
    print(f"Price and volume chart saved to data/price_volume_chart.png")
    
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    print("\nCreating a sample dataset to demonstrate the concept...")
    
    # Create a fallback sample dataset for demonstration
    start_time = pd.Timestamp('2024-06-02 09:30:00')
    end_time = pd.Timestamp('2024-06-02 16:00:00')
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Generate sample data with some volume clusters
    np.random.seed(42)  # For reproducibility
    n = len(timestamps)
    base_price = 100.0
    
    # Create price data
    close_prices = base_price + np.cumsum(np.random.normal(0, 0.1, n))
    open_prices = close_prices - np.random.normal(0, 0.3, n)
    high_prices = np.maximum(close_prices, open_prices) + np.random.normal(0.1, 0.2, n)
    low_prices = np.minimum(close_prices, open_prices) - np.random.normal(0.1, 0.2, n)
    
    # Create volume with spikes
    base_volume = 1000
    volume = np.random.normal(base_volume, 200, n)
    
    # Add volume spikes at specific times
    for i in range(0, n, 60):  # Add a spike roughly every hour
        if i+5 < n:
            volume[i+5] = volume[i+5] * 5  # 5x spike
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=timestamps)
    
    # Identify clusters
    clusters = identify_volume_clusters(df)
    
    # Save results
    df.to_csv("data/sample_data.csv")
    clusters.to_csv("data/sample_clusters.csv")
    print(f"Sample data saved to data/sample_data.csv")
    print(f"Sample clusters saved to data/sample_clusters.csv")
    
    # Create a basic volume histogram for visualization
    plt.figure(figsize=(10, 6))
    df_15m = df.resample('15min').agg({'volume': 'sum'})
    plt.hist(df_15m['volume'], bins=30)
    plt.axvline(3 * df_15m['volume'].mean(), color='r', linestyle='--', label='3x Threshold')
    plt.title('Distribution of 15-Minute Volume (Sample Data)')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('data/sample_volume_distribution.png')
    print(f"Sample volume histogram saved to data/sample_volume_distribution.png") 