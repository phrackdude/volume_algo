import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import skew, kurtosis, ttest_1samp, ttest_ind

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
        modal_prices = []
        price_deltas = []
        
        # Forward return analysis
        return_15m_values = []
        return_30m_values = []
        return_60m_values = []
        
        # Retest analysis
        retested_poc_values = []
        retest_time_values = []
        
        for cluster_time in high_volume_periods.index:
            # Get the underlying volume data for this 15-minute period
            cluster_start = cluster_time
            cluster_end = cluster_time + pd.Timedelta(minutes=15)
            
            # Extract individual minute-level volumes for this cluster period
            cluster_data = day_data[
                (day_data.index >= cluster_start) & 
                (day_data.index < cluster_end)
            ]
            
            cluster_volumes = cluster_data['volume'].values
            
            # Calculate skewness and kurtosis
            if len(cluster_volumes) >= 3 and cluster_volumes.std() > 0:
                cluster_skewness = skew(cluster_volumes)
                cluster_kurtosis = kurtosis(cluster_volumes)
            else:
                cluster_skewness = np.nan
                cluster_kurtosis = np.nan
            
            skewness_values.append(cluster_skewness)
            kurtosis_values.append(cluster_kurtosis)
            
            # Modal price calculation
            if len(cluster_data) > 0:
                # Use 0.25 tick size for ES futures
                tick_size = 0.25
                
                # Create price bins
                min_price = cluster_data['close'].min()
                max_price = cluster_data['close'].max()
                
                # Ensure we have a reasonable range
                if max_price - min_price < tick_size:
                    # If price range is too small, create bins around the mean price
                    mean_price = cluster_data['close'].mean()
                    min_price = mean_price - 2 * tick_size
                    max_price = mean_price + 2 * tick_size
                
                # Create bins
                bins = np.arange(min_price, max_price + tick_size, tick_size)
                
                # Calculate volume per price bin
                hist, bin_edges = np.histogram(cluster_data['close'], bins=bins, weights=cluster_data['volume'])
                
                # Find the bin with maximum volume (modal price)
                if len(hist) > 0 and hist.max() > 0:
                    max_volume_idx = np.argmax(hist)
                    modal_price = (bin_edges[max_volume_idx] + bin_edges[max_volume_idx + 1]) / 2
                else:
                    modal_price = cluster_data['close'].mean()
                
                # Calculate price delta (modal price - closing price of cluster)
                cluster_close = cluster_data['close'].iloc[-1]  # Last close in the cluster
                price_delta = modal_price - cluster_close
            else:
                modal_price = np.nan
                price_delta = np.nan
                cluster_close = np.nan
            
            modal_prices.append(modal_price)
            price_deltas.append(price_delta)
            
            # =================================================================
            # FORWARD RETURN ANALYSIS
            # =================================================================
            
            # Get the closing price at the end of the cluster (t=0)
            if len(cluster_data) > 0:
                t0_close = cluster_data['close'].iloc[-1]  # Last close price in cluster
                
                # Calculate forward returns at different horizons
                forward_returns = {}
                
                for horizon_minutes, return_key in [(15, 'return_15m'), (30, 'return_30m'), (60, 'return_60m')]:
                    # Find the target time
                    target_time = cluster_end + pd.Timedelta(minutes=horizon_minutes)
                    
                    # Look for data at or after the target time
                    future_data = day_data[day_data.index >= target_time]
                    
                    if len(future_data) > 0:
                        # Use the first available price at or after target time
                        t_horizon_close = future_data['close'].iloc[0]
                        
                        # Calculate return: (close_t+n - close_t) / close_t
                        forward_return = (t_horizon_close - t0_close) / t0_close
                        forward_returns[return_key] = forward_return
                    else:
                        # Not enough future data (e.g., end of day)
                        forward_returns[return_key] = np.nan
                
            else:
                # No cluster data available
                forward_returns = {'return_15m': np.nan, 'return_30m': np.nan, 'return_60m': np.nan}
            
            # Store forward returns
            return_15m_values.append(forward_returns['return_15m'])
            return_30m_values.append(forward_returns['return_30m'])
            return_60m_values.append(forward_returns['return_60m'])
            
            # =================================================================
            # RETEST ANALYSIS - Check if modal price is revisited
            # =================================================================
            
            # Only perform retest analysis if we have a valid modal price
            if not np.isnan(modal_price):
                # Define retest threshold (±3 ticks for ES futures)
                retest_threshold = 0.75  # 3 * 0.25 tick size
                
                # Define the retest window: from cluster_end + 1 minute to cluster_end + 60 minutes
                retest_start = cluster_end + pd.Timedelta(minutes=1)
                retest_end = cluster_end + pd.Timedelta(minutes=60)
                
                # Get data in the retest window
                retest_data = day_data[
                    (day_data.index >= retest_start) & 
                    (day_data.index <= retest_end)
                ]
                
                # Check if price revisits modal price within ±threshold
                retested = False
                retest_time_minutes = np.nan
                
                if len(retest_data) > 0:
                    # For each minute in the retest window, check if price range overlaps with modal price ±threshold
                    # Overlap occurs when: low <= modal_price + threshold AND high >= modal_price - threshold
                    upper_bound = modal_price + retest_threshold
                    lower_bound = modal_price - retest_threshold
                    
                    # Find bars where price range overlaps with the modal price zone
                    retest_hits = retest_data[
                        (retest_data['low'] <= upper_bound) & 
                        (retest_data['high'] >= lower_bound)
                    ]
                    
                    if len(retest_hits) > 0:
                        retested = True
                        # Calculate time to first retest in minutes
                        first_retest_time = retest_hits.index[0]
                        retest_time_minutes = (first_retest_time - cluster_end).total_seconds() / 60
                
                retested_poc_values.append(retested)
                retest_time_values.append(retest_time_minutes)
            else:
                # No valid modal price, so no retest possible
                retested_poc_values.append(False)
                retest_time_values.append(np.nan)
        
        # Add all metrics to the result DataFrame
        high_volume_periods['skewness'] = skewness_values
        high_volume_periods['kurtosis'] = kurtosis_values
        high_volume_periods['modal_price'] = modal_prices
        high_volume_periods['price_delta'] = price_deltas
        high_volume_periods['return_15m'] = return_15m_values
        high_volume_periods['return_30m'] = return_30m_values
        high_volume_periods['return_60m'] = return_60m_values
        high_volume_periods['retested_poc'] = retested_poc_values
        high_volume_periods['retest_time'] = retest_time_values
        
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

def analyze_forward_returns(clusters_df):
    """
    Analyze forward returns to determine if volume clusters have predictive power.
    
    Parameters:
    -----------
    clusters_df : DataFrame
        DataFrame containing volume clusters with forward return columns
        
    Returns:
    --------
    dict
        Dictionary containing statistical analysis results
    """
    print("\n" + "="*60)
    print("DIRECTIONAL BIAS ANALYSIS")
    print("="*60)
    
    results = {}
    return_columns = ['return_15m', 'return_30m', 'return_60m']
    horizons = ['15-minute', '30-minute', '60-minute']
    
    for return_col, horizon in zip(return_columns, horizons):
        if return_col in clusters_df.columns:
            # Get non-NaN returns
            returns = clusters_df[return_col].dropna()
            
            if len(returns) > 0:
                # Basic statistics
                mean_return = returns.mean()
                std_return = returns.std()
                count = len(returns)
                
                # Convert to percentage for easier interpretation
                mean_return_pct = mean_return * 100
                std_return_pct = std_return * 100
                
                # Perform one-sample t-test against zero
                # H0: mean return = 0 (no directional bias)
                # H1: mean return ≠ 0 (directional bias exists)
                if count > 1:
                    t_stat, p_value = ttest_1samp(returns, 0)
                    
                    # Determine significance
                    is_significant = p_value < 0.05
                    significance_level = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    
                    # Determine direction
                    direction = "Bullish" if mean_return > 0 else "Bearish" if mean_return < 0 else "Neutral"
                    
                    # Store results
                    results[return_col] = {
                        'horizon': horizon,
                        'count': count,
                        'mean_return': mean_return,
                        'mean_return_pct': mean_return_pct,
                        'std_return': std_return,
                        'std_return_pct': std_return_pct,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'direction': direction,
                        'significance_stars': significance_level
                    }
                    
                    # Print results
                    print(f"\n{horizon.upper()} FORWARD RETURNS:")
                    print(f"  Sample size: {count}")
                    print(f"  Mean return: {mean_return_pct:+.2f}% {significance_level}")
                    print(f"  Std deviation: {std_return_pct:.2f}%")
                    print(f"  Direction: {direction}")
                    print(f"  t-statistic: {t_stat:.3f}")
                    print(f"  p-value: {p_value:.4f}")
                    
                    if is_significant:
                        print(f"  ✅ SIGNIFICANT: Volume clusters show {direction.lower()} bias")
                    else:
                        print(f"  ❌ NOT SIGNIFICANT: No clear directional bias detected")
                    
                    # Additional insights
                    positive_returns = returns[returns > 0]
                    negative_returns = returns[returns < 0]
                    
                    win_rate = len(positive_returns) / count * 100
                    
                    print(f"  Win rate: {win_rate:.1f}% ({len(positive_returns)}/{count})")
                    
                    if len(positive_returns) > 0:
                        avg_win = positive_returns.mean() * 100
                        print(f"  Average win: +{avg_win:.2f}%")
                    
                    if len(negative_returns) > 0:
                        avg_loss = negative_returns.mean() * 100
                        print(f"  Average loss: {avg_loss:.2f}%")
                        
                else:
                    print(f"\n{horizon.upper()} FORWARD RETURNS:")
                    print(f"  Insufficient data for t-test (n={count})")
                    
                    results[return_col] = {
                        'horizon': horizon,
                        'count': count,
                        'mean_return': mean_return,
                        'mean_return_pct': mean_return_pct,
                        'std_return': std_return,
                        'std_return_pct': std_return_pct,
                        't_stat': np.nan,
                        'p_value': np.nan,
                        'is_significant': False,
                        'direction': 'Insufficient data',
                        'significance_stars': ''
                    }
            else:
                print(f"\n{horizon.upper()} FORWARD RETURNS:")
                print(f"  No valid data available")
                
                results[return_col] = {
                    'horizon': horizon,
                    'count': 0,
                    'mean_return': np.nan,
                    'mean_return_pct': np.nan,
                    'std_return': np.nan,
                    'std_return_pct': np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan,
                    'is_significant': False,
                    'direction': 'No data',
                    'significance_stars': ''
                }
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    significant_horizons = [results[col]['horizon'] for col in return_columns 
                          if col in results and results[col]['is_significant']]
    
    if significant_horizons:
        print(f"📊 Volume clusters show significant directional bias at: {', '.join(significant_horizons)}")
        
        # Find the most significant result
        best_horizon = min([col for col in return_columns if col in results and results[col]['is_significant']], 
                          key=lambda x: results[x]['p_value'])
        best_result = results[best_horizon]
        
        print(f"🎯 Strongest signal: {best_result['horizon']} with {best_result['mean_return_pct']:+.2f}% average return")
        print(f"   (p-value: {best_result['p_value']:.4f}, direction: {best_result['direction']})")
        
    else:
        print(f"📊 No statistically significant directional bias detected at any time horizon")
        print(f"💡 This suggests volume clusters may not be predictive of future price direction")
    
    return results

def analyze_retest_bias(clusters_df):
    """
    Analyze the relationship between retest behavior and directional bias.
    
    Parameters:
    -----------
    clusters_df : DataFrame
        DataFrame containing volume clusters with retest and return columns
        
    Returns:
    --------
    dict
        Dictionary containing retest bias analysis results
    """
    if 'retested_poc' not in clusters_df.columns or 'return_30m' not in clusters_df.columns:
        return {}
    
    print("\n" + "="*60)
    print("RETEST vs DIRECTIONAL BIAS ANALYSIS")
    print("="*60)
    
    # Split clusters into retested vs non-retested
    retested_clusters = clusters_df[clusters_df['retested_poc'] == True]
    non_retested_clusters = clusters_df[clusters_df['retested_poc'] == False]
    
    results = {}
    
    for return_col in ['return_15m', 'return_30m', 'return_60m']:
        if return_col in clusters_df.columns:
            horizon = return_col.replace('return_', '').replace('m', '-minute')
            
            # Analyze returns for retested clusters
            retested_returns = retested_clusters[return_col].dropna()
            non_retested_returns = non_retested_clusters[return_col].dropna()
            
            if len(retested_returns) > 0 and len(non_retested_returns) > 0:
                retested_mean = retested_returns.mean() * 100
                non_retested_mean = non_retested_returns.mean() * 100
                
                print(f"\n{horizon.upper()} RETURNS BY RETEST STATUS:")
                print(f"  Retested clusters ({len(retested_returns)}): {retested_mean:+.2f}%")
                print(f"  Non-retested clusters ({len(non_retested_returns)}): {non_retested_mean:+.2f}%")
                print(f"  Difference: {retested_mean - non_retested_mean:+.2f}%")
                
                # Perform t-test between the two groups
                try:
                    t_stat, p_value = ttest_ind(retested_returns, non_retested_returns)
                    is_significant = p_value < 0.05
                    
                    print(f"  t-statistic: {t_stat:.3f}")
                    print(f"  p-value: {p_value:.4f}")
                    
                    if is_significant:
                        if retested_mean > non_retested_mean:
                            print(f"  ✅ SIGNIFICANT: Retested clusters show MORE positive bias")
                        else:
                            print(f"  ✅ SIGNIFICANT: Retested clusters show MORE negative bias")
                    else:
                        print(f"  ❌ NOT SIGNIFICANT: No difference between retested and non-retested clusters")
                        
                    results[return_col] = {
                        'retested_mean_pct': retested_mean,
                        'non_retested_mean_pct': non_retested_mean,
                        'difference_pct': retested_mean - non_retested_mean,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'retested_count': len(retested_returns),
                        'non_retested_count': len(non_retested_returns)
                    }
                except ImportError:
                    print(f"  (t-test requires scipy - using basic comparison)")
                    results[return_col] = {
                        'retested_mean_pct': retested_mean,
                        'non_retested_mean_pct': non_retested_mean,
                        'difference_pct': retested_mean - non_retested_mean,
                        'retested_count': len(retested_returns),
                        'non_retested_count': len(non_retested_returns)
                    }
    
    return results

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
        display_columns = ['volume', 'cluster_strength', 'threshold', 'skewness', 'kurtosis', 
                          'modal_price', 'price_delta', 'return_15m', 'return_30m', 'return_60m',
                          'retested_poc', 'retest_time']
        available_columns = [col for col in display_columns if col in clusters.columns]
        print(clusters[available_columns].head())
        
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
        
        # Print modal price analysis summary
        print(f"\nModal price analysis:")
        if 'modal_price' in clusters.columns:
            modal_prices = clusters['modal_price'].dropna()
            if len(modal_prices) > 0:
                print(f"  Average modal price: ${modal_prices.mean():.2f}")
                print(f"  Modal price range: ${modal_prices.min():.2f} - ${modal_prices.max():.2f}")
        
        if 'price_delta' in clusters.columns:
            price_deltas = clusters['price_delta'].dropna()
            if len(price_deltas) > 0:
                print(f"  Average price delta: ${price_deltas.mean():.2f}")
                print(f"  Price delta range: ${price_deltas.min():.2f} - ${price_deltas.max():.2f}")
        
        # Print retest analysis summary
        print(f"\nRetest analysis (±3 ticks within 60 minutes):")
        if 'retested_poc' in clusters.columns:
            retest_data = clusters['retested_poc'].dropna()
            if len(retest_data) > 0:
                retest_rate = retest_data.sum() / len(retest_data) * 100
                print(f"  Retest rate: {retest_rate:.1f}% ({retest_data.sum()}/{len(retest_data)})")
                
                # Analyze retest times
                if 'retest_time' in clusters.columns:
                    retest_times = clusters[clusters['retested_poc'] == True]['retest_time'].dropna()
                    if len(retest_times) > 0:
                        print(f"  Average retest time: {retest_times.mean():.1f} minutes")
                        print(f"  Median retest time: {retest_times.median():.1f} minutes")
                        print(f"  Retest time range: {retest_times.min():.1f} - {retest_times.max():.1f} minutes")
                        
                        # Quick retest analysis (within 15 minutes)
                        quick_retests = retest_times[retest_times <= 15].count()
                        if len(retest_times) > 0:
                            quick_retest_rate = quick_retests / len(retest_times) * 100
                            print(f"  Quick retests (≤15 min): {quick_retest_rate:.1f}% ({quick_retests}/{len(retest_times)})")
        
        # Perform directional bias analysis
        return_analysis = analyze_forward_returns(clusters)
        
        # Save return analysis summary to CSV
        if return_analysis:
            summary_data = []
            for return_col, stats in return_analysis.items():
                summary_data.append({
                    'horizon': stats['horizon'],
                    'sample_size': stats['count'],
                    'mean_return_pct': stats['mean_return_pct'],
                    'std_return_pct': stats['std_return_pct'],
                    't_statistic': stats['t_stat'],
                    'p_value': stats['p_value'],
                    'is_significant': stats['is_significant'],
                    'direction': stats['direction']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = 'data/cluster_return_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"\n📊 Return analysis summary saved to: {summary_path}")
        
        # Plot for each date with clusters
        print("\nCreating plots...")
        dates = clusters['date'].unique()
        for date in dates:
            os.makedirs('data', exist_ok=True)
            save_path = f'data/volume_profile_{date}.png'
            plot_volume_profile(df, clusters, date, save_path)
            print(f"Plot for {date} saved to {save_path}")
        
        # Perform retest bias analysis
        retest_bias_analysis = analyze_retest_bias(clusters)
        
        # Save retest bias analysis summary to CSV
        if retest_bias_analysis:
            retest_bias_data = []
            for return_col, stats in retest_bias_analysis.items():
                retest_bias_data.append({
                    'return_col': return_col,
                    'retested_mean_pct': stats['retested_mean_pct'],
                    'non_retested_mean_pct': stats['non_retested_mean_pct'],
                    'difference_pct': stats['difference_pct'],
                    't_stat': stats['t_stat'],
                    'p_value': stats['p_value'],
                    'is_significant': stats['is_significant'],
                    'retested_count': stats['retested_count'],
                    'non_retested_count': stats['non_retested_count']
                })
            
            retest_bias_df = pd.DataFrame(retest_bias_data)
            retest_bias_path = 'data/retest_bias_summary.csv'
            retest_bias_df.to_csv(retest_bias_path, index=False)
            print(f"\n📊 Retest bias analysis summary saved to: {retest_bias_path}")
    else:
        print("No clusters found with the current threshold.") 