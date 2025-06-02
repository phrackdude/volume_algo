import pandas as pd
import numpy as np

# Aggregates 1-min data into 15-minute blocks, then finds high-volume clusters
def identify_volume_clusters(df, volume_multiplier=3):
    # 15-minute resample
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Calculate the average volume
    avg_volume = df_15m['volume'].mean()
    print(f"Average 15-minute volume: {avg_volume:.2f}")
    
    # Set threshold
    threshold = volume_multiplier * avg_volume
    print(f"Volume threshold ({volume_multiplier}x avg): {threshold:.2f}")
    
    # Find clusters (periods with volume >= threshold)
    clusters = df_15m[df_15m['volume'] >= threshold].copy()
    
    # If no clusters were found, investigate why
    if len(clusters) == 0:
        print("\nNo clusters found! Investigating volume distribution...")
        
        # Print volume statistics
        print("Volume statistics:")
        print(df_15m['volume'].describe())
        
        # Try with a lower threshold as a fallback
        lower_multiplier = 1.5
        lower_threshold = lower_multiplier * avg_volume
        print(f"\nTrying with a lower threshold ({lower_multiplier}x avg): {lower_threshold:.2f}")
        
        clusters = df_15m[df_15m['volume'] >= lower_threshold].copy()
        print(f"Found {len(clusters)} clusters with lower threshold")
    
    print(f"Found {len(clusters)} volume clusters out of {len(df_15m)} 15-minute periods")
    
    # Add a cluster_strength column (ratio to average volume)
    if not clusters.empty:
        clusters['cluster_strength'] = clusters['volume'] / avg_volume
        clusters = clusters.sort_values('cluster_strength', ascending=False)
        
    return clusters 