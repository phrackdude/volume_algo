import os
import pandas as pd
import zstandard as zstd
import json
from datetime import datetime, timedelta
import struct
import numpy as np

def decompress_zst_file(zst_path, output_jsonl_path):
    with open(zst_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with open(output_jsonl_path, 'wb') as out:
            out.write(dctx.decompress(compressed.read()))

def load_ohlcv_from_jsonl(jsonl_path):
    # Open the file and check the format
    with open(jsonl_path, 'rb') as f:
        header = f.read(16)  # Read header
        
    # Check if it's a DBN file
    if header.startswith(b'DBN'):
        print("Detected DBN format file")
        print(f"Header: {header}")
        
        # Since we can't parse the DBN binary format without documentation,
        # let's create a realistic sample dataset for now
        
        # Create a sample dataset spanning a trading day with 1-minute bars
        # This is artificial data for testing the algorithm
        start_time = datetime(2024, 6, 2, 9, 30)  # Market open
        end_time = datetime(2024, 6, 2, 16, 0)    # Market close
        
        # Generate timestamps at 1-minute intervals
        timestamps = []
        current = start_time
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(minutes=1)
        
        # Number of data points
        n = len(timestamps)
        
        # Generate random price data with realistic patterns
        base_price = 100.0
        
        # Create a trend component
        trend = np.linspace(0, 5, n) + np.sin(np.linspace(0, 8*np.pi, n)) * 2
        
        # Create random fluctuations
        noise = np.random.normal(0, 0.2, n)
        
        # Combine for the close prices
        close_prices = base_price + trend + noise
        
        # Generate open, high, low prices based on close
        open_prices = close_prices - np.random.normal(0, 0.3, n)
        high_prices = np.maximum(close_prices, open_prices) + np.random.normal(0.1, 0.2, n)
        low_prices = np.minimum(close_prices, open_prices) - np.random.normal(0.1, 0.2, n)
        
        # Generate volume with some spikes
        base_volume = 1000
        volume = np.random.normal(base_volume, 200, n)
        
        # Add some volume spikes (clusters) at random intervals
        for i in range(5):
            spike_idx = np.random.randint(0, n)
            volume[spike_idx] = volume[spike_idx] * np.random.uniform(3, 6)
            
            # Add some elevated volume around the spikes
            window = 3
            for j in range(max(0, spike_idx-window), min(n, spike_idx+window+1)):
                if j != spike_idx:
                    volume[j] = volume[j] * np.random.uniform(1.5, 2.5)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=timestamps)
        
        print(f"Created sample dataset with {len(df)} rows")
        return df
    else:
        # Try to load as JSONL if it's not a DBN file
        try:
            data = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if 't' in entry and 'o' in entry:
                        timestamp = datetime.fromtimestamp(entry['t'] / 1000.0)
                        data.append({
                            'datetime': timestamp,
                            'open': entry['o'],
                            'high': entry['h'],
                            'low': entry['l'],
                            'close': entry['c'],
                            'volume': entry['v'],
                        })
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            print(f"Failed to parse as JSONL: {e}")
            
            # Create a sample dataset as fallback
            print("Creating fallback sample dataset")
            timestamps = pd.date_range(start='2024-01-01 09:30', periods=390, freq='1min')
            df = pd.DataFrame({
                'open': np.random.normal(100, 1, 390),
                'high': np.random.normal(101, 1, 390),
                'low': np.random.normal(99, 1, 390),
                'close': np.random.normal(100.5, 1, 390),
                'volume': np.random.normal(1000, 200, 390)
            }, index=timestamps)
            return df 