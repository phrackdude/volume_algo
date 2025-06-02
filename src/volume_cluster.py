import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import skew, kurtosis

def identify_volume_clusters(df, volume_multiplier=3.0):
    """
    Identify periods of unusually high trading volume on a day-by-day basis.
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame with datetime index and OHLCV columns
    volume_multiplier : float
        Multiplier to determine the threshold for high volume
        (e.g., 3.0 means 3x the daily average 15-minute volume)
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with information about each volume cluster including
        skewness and kurtosis metrics
    """
    # Ensure the DataFrame is sorted by time
    df = df.sort_index()
    
    # Get unique dates
    dates = pd.Series([d.date() for d in df.index]).unique()
    
    print(f"Processing {len(dates)} trading days...")
    
    # Container for clusters
    all_clusters = []
    
    # Process each day separately
    for date in dates:
        # Get data for this day
        day_data = df[df.index.date == date]
        
        # Skip if no data for this day
        if day_data.empty:
            continue
        
        # Resample to 15-minute intervals
        df_15m = day_data.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate average 15-minute volume for this day
        avg_15m_volume = df_15m['volume'].mean()
        
        # Set threshold based on multiple of average
        threshold = avg_15m_volume * volume_multiplier
        
        # Find periods with volume above threshold
        high_volume_periods = df_15m[df_15m['volume'] > threshold].copy()  # Create a copy to avoid warning
        
        # If no clusters found in this day
        if high_volume_periods.empty:
            print(f"Date {date}: No clusters found (avg 15-min vol: {avg_15m_volume:.2f}, threshold: {threshold:.2f})")
            continue
        
        # Calculate cluster strength (how many times above average)
        high_volume_periods.loc[:, 'cluster_strength'] = high_volume_periods['volume'] / avg_15m_volume
        
        # Add date, threshold and average volume columns
        high_volume_periods.loc[:, 'date'] = date
        high_volume_periods.loc[:, 'threshold'] = threshold
        high_volume_periods.loc[:, 'avg_volume'] = avg_15m_volume
        
        # Calculate skewness and kurtosis for each cluster
        skewness_values = []
        kurtosis_values = []
        
        for cluster_time in high_volume_periods.index:
            # Get the underlying volume data for this 15-minute period
            cluster_start = cluster_time
            cluster_end = cluster_time + pd.Timedelta(minutes=15)
            
            # Extract individual minute-level volumes for this cluster period
            cluster_volumes = day_data[
                (day_data.index >= cluster_start) & 
                (day_data.index < cluster_end)
            ]['volume']
            
            # Calculate skewness and kurtosis
            if len(cluster_volumes) >= 3 and cluster_volumes.std() > 0:
                # Only calculate if we have enough data points and non-constant data
                cluster_skew = skew(cluster_volumes)
                cluster_kurt = kurtosis(cluster_volumes)
            else:
                # Set to NaN if insufficient data or constant values
                cluster_skew = np.nan
                cluster_kurt = np.nan
            
            skewness_values.append(cluster_skew)
            kurtosis_values.append(cluster_kurt)
        
        # Add skewness and kurtosis columns
        high_volume_periods.loc[:, 'skewness'] = skewness_values
        high_volume_periods.loc[:, 'kurtosis'] = kurtosis_values
        
        # Add to all clusters
        all_clusters.append(high_volume_periods)
        
        # Print summary for this day
        print(f"Date {date}: Found {len(high_volume_periods)} clusters (avg 15-min vol: {avg_15m_volume:.2f}, threshold: {threshold:.2f})")
    
    # Combine all clusters
    if not all_clusters:
        print("\nNo clusters found in any day with the current threshold.")
        return pd.DataFrame()
    
    # Concatenate all clusters
    clusters_df = pd.concat(all_clusters)
    
    # Sort by time
    clusters_df = clusters_df.sort_index()
    
    return clusters_df

def plot_volume_profile(df, clusters_df, date, save_path=None):
    """
    Plot volume profile for a specific day with clusters highlighted.
    
    Parameters:
    -----------
    df : DataFrame
        Original DataFrame with OHLCV data
    clusters_df : DataFrame
        DataFrame with cluster information
    date : datetime.date
        Date to plot
    save_path : str, optional
        Path to save the plot
    """
    # Filter data for this date
    day_data = df[df.index.date == date]
    day_clusters = clusters_df[clusters_df['date'] == date]
    
    # Skip if no data
    if day_data.empty:
        return
    
    # Create figure with price and volume subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Volume Profile for {date}', fontsize=16)
    
    # Resample to 15-minute intervals
    df_15m = day_data.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Price subplot
    ax1.plot(df_15m.index, df_15m['close'], label='Close Price')
    
    # Mark cluster times on price chart
    for idx in day_clusters.index:
        ax1.axvline(idx, color='r', alpha=0.5, linestyle='--', 
                   label='Volume Cluster' if idx == day_clusters.index[0] else None)
    
    ax1.set_title('Price with Volume Clusters')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Volume subplot
    ax2.bar(df_15m.index, df_15m['volume'], width=0.01, color='b', alpha=0.6, label='Volume')
    
    # Highlight cluster volumes
    for idx in day_clusters.index:
        ax2.bar(idx, day_clusters.loc[idx, 'volume'], width=0.01, color='r', alpha=0.8, 
               label='Cluster Volume' if idx == day_clusters.index[0] else None)
    
    # Add threshold line
    if not day_clusters.empty:
        threshold = day_clusters['threshold'].iloc[0]
        avg = day_clusters['avg_volume'].iloc[0]
        ax2.axhline(threshold, color='r', linestyle='--', 
                  label=f'Threshold ({threshold:.0f})')
        ax2.axhline(avg, color='g', linestyle=':', 
                  label=f'Avg Volume ({avg:.0f})')
    
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.legend()
    
    # Save or show the figure
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def analyze_cluster_timing(clusters_df):
    """
    Analyze the timing patterns of volume clusters.
    
    Parameters:
    -----------
    clusters_df : DataFrame
        DataFrame with cluster information
        
    Returns:
    --------
    dict
        Dictionary with timing analysis results
    """
    if clusters_df.empty:
        return {}
    
    # Create a copy to avoid warnings
    clusters_copy = clusters_df.copy()
    
    # Extract hour and minute
    clusters_copy.loc[:, 'hour'] = clusters_copy.index.hour
    clusters_copy.loc[:, 'minute'] = clusters_copy.index.minute
    
    # Time of day distribution
    morning = clusters_copy[(clusters_copy['hour'] >= 9) & (clusters_copy['hour'] < 12)].shape[0]
    afternoon = clusters_copy[(clusters_copy['hour'] >= 12) & (clusters_copy['hour'] < 16)].shape[0]
    close = clusters_copy[clusters_copy['hour'] >= 16].shape[0]
    
    # Calculate hour distribution
    hour_dist = clusters_copy.groupby('hour').size()
    
    # Calculate cluster strength by time of day
    hour_strength = clusters_copy.groupby('hour')['cluster_strength'].mean()
    
    return {
        'total_clusters': len(clusters_copy),
        'morning_clusters': morning,
        'afternoon_clusters': afternoon,
        'close_clusters': close,
        'hour_distribution': hour_dist.to_dict(),
        'hour_strength': hour_strength.to_dict()
    }

if __name__ == "__main__":
    # Load synthetic data for testing
    from synthetic_data import generate_synthetic_ohlcv
    
    print("Generating synthetic OHLCV data for testing...")
    df = generate_synthetic_ohlcv(start_date='2024-06-03', num_days=5, seed=42)
    
    # Identify clusters
    print("\nIdentifying volume clusters...")
    clusters = identify_volume_clusters(df, volume_multiplier=3.0)
    
    if not clusters.empty:
        # Print results
        print(f"\nFound {len(clusters)} volume clusters:")
        print(clusters[['volume', 'cluster_strength', 'threshold', 'skewness', 'kurtosis']].head())
        
        # Analyze timing
        print("\nCluster timing analysis:")
        timing_analysis = analyze_cluster_timing(clusters)
        for key, value in timing_analysis.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Plot for each date with clusters
        print("\nCreating plots...")
        dates = clusters['date'].unique()
        for date in dates:
            os.makedirs('data', exist_ok=True)
            save_path = f'data/volume_profile_{date}.png'
            plot_volume_profile(df, clusters, date, save_path)
            print(f"Plot for {date} saved to {save_path}")
    else:
        print("No clusters found with the current threshold.") 