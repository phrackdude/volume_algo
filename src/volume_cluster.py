import pandas as pd
import numpy as np

# Identifies high-volume clusters on a day-by-day basis
def identify_volume_clusters(df, volume_multiplier=3):
    # Create an empty list to store cluster results for each day
    all_clusters = []
    
    # Group the DataFrame by calendar date
    df['date'] = df.index.date
    grouped = df.groupby('date')
    
    print(f"Processing {len(grouped)} trading days...")
    
    # Process each day separately
    for date, day_data in grouped:
        # Remove the 'date' column from the day's data to avoid resampling issues
        day_data = day_data.drop(columns=['date'])
        
        # Resample to 15-minute intervals for this day
        df_15m = day_data.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Skip days with too few data points
        if len(df_15m) < 2:
            print(f"Skipping date {date} - insufficient data (only {len(df_15m)} 15-minute bars)")
            continue
            
        # Calculate this day's average 15-minute volume
        daily_avg_volume = df_15m['volume'].mean()
        
        # Skip days with no data or zero average volume
        if daily_avg_volume == 0 or pd.isna(daily_avg_volume):
            print(f"Skipping date {date} - zero or NaN average volume")
            continue
            
        # Set threshold for this day
        daily_threshold = volume_multiplier * daily_avg_volume
        
        # Find clusters for this day (periods with volume >= threshold)
        day_clusters = df_15m[df_15m['volume'] >= daily_threshold].copy()
        
        # Add day-specific information
        if not day_clusters.empty:
            # Add date information
            day_clusters['date'] = date
            day_clusters['avg_volume'] = daily_avg_volume
            day_clusters['threshold'] = daily_threshold
            
            # Add cluster strength (ratio to daily average)
            day_clusters['cluster_strength'] = day_clusters['volume'] / daily_avg_volume
            
            # Add to the list of all clusters
            all_clusters.append(day_clusters)
            
            print(f"Date {date}: Found {len(day_clusters)} clusters (avg 15-min vol: {daily_avg_volume:.2f}, threshold: {daily_threshold:.2f})")
        else:
            print(f"Date {date}: No clusters found (avg 15-min vol: {daily_avg_volume:.2f}, threshold: {daily_threshold:.2f})")
    
    # Combine all clusters into a single DataFrame
    if all_clusters:
        combined_clusters = pd.concat(all_clusters)
        
        # Sort by timestamp
        combined_clusters = combined_clusters.sort_index()
        
        print(f"\nTotal clusters found across all days: {len(combined_clusters)}")
        return combined_clusters
    else:
        print("\nNo clusters found in any day with the current threshold.")
        
        # Return an empty DataFrame with the right columns
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 
                                         'date', 'avg_volume', 'threshold', 'cluster_strength'])
        return empty_df 