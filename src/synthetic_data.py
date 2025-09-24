import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os

def generate_synthetic_ohlcv(start_date='2024-06-01', num_days=5, seed=None):
    """
    Generate synthetic 1-minute OHLCV data for market analysis.
    
    Parameters:
    -----------
    start_date : str
        Starting date in 'YYYY-MM-DD' format.
    num_days : int
        Number of trading days to generate.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with datetime index and OHLCV columns.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Parse start date
    start = pd.Timestamp(start_date)
    
    # Ensure start_date is adjusted to the next Monday if it falls on a weekend
    if start.weekday() >= 5:  # 5=Saturday, 6=Sunday
        days_to_add = 7 - start.weekday()
        start = start + timedelta(days=days_to_add)
    
    # Generate trading day dates (Monday-Friday only)
    trading_days = []
    current_date = start
    
    while len(trading_days) < num_days:
        if current_date.weekday() < 5:  # Monday-Friday
            trading_days.append(current_date)
        current_date = current_date + timedelta(days=1)
    
    # Define trading hours
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Generate timestamps for all trading minutes
    timestamps = []
    for day in trading_days:
        current_minute = datetime.combine(day.date(), market_open)
        end_minute = datetime.combine(day.date(), market_close)
        
        while current_minute <= end_minute:
            timestamps.append(current_minute)
            current_minute += timedelta(minutes=1)
    
    # Starting price
    base_price = 100.0
    
    # Track current price for all days
    current_price = base_price
    
    # Process each day separately to create day-specific patterns
    all_opens = []
    all_highs = []
    all_lows = []
    all_closes = []
    all_volumes = []
    
    # Track last close from previous day
    prev_day_close = base_price
    
    # Price evolution parameters
    volatility = 0.1
    
    # Volume parameters
    base_volume = 1000
    volume_noise = 0.2  # Percentage of noise
    
    # Process each day separately
    for day_idx, day in enumerate(trading_days):
        # Get timestamps for this day
        day_timestamps = [ts for ts in timestamps if ts.date() == day.date()]
        minutes_in_day = len(day_timestamps)
        
        # Create a daily bias/trend
        daily_bias = np.random.normal(0, 0.5) / 100  # Daily price drift
        
        # Initialize day arrays
        day_opens = []
        day_highs = []
        day_lows = []
        day_closes = []
        day_volumes = []
        
        # Minute-by-minute price evolution
        for minute_idx, ts in enumerate(day_timestamps):
            # Normalized time within the day (0 to 1)
            t = minute_idx / minutes_in_day
            
            # First minute of the day
            if minute_idx == 0:
                # Gap from previous day's close with some randomness
                open_price = prev_day_close * (1 + np.random.normal(daily_bias * 10, 0.003))
                current_price = open_price
            else:
                # Open at previous minute's close
                open_price = day_closes[-1]
            
            # Calculate minute's price movement
            # More volatility at open and close
            time_volatility = volatility * (1 + 0.5 * (np.exp(-10 * (t - 0.1)**2) + np.exp(-10 * (t - 0.9)**2)))
            
            # Price change for this minute
            price_change = np.random.normal(daily_bias, time_volatility)
            close_price = open_price * (1 + price_change)
            
            # Determine high and low with intrabar volatility
            intrabar_range = abs(open_price - close_price) + open_price * np.random.uniform(0.0005, 0.002)
            if open_price <= close_price:
                high_price = close_price + np.random.uniform(0, intrabar_range/2)
                low_price = open_price - np.random.uniform(0, intrabar_range/2)
            else:
                high_price = open_price + np.random.uniform(0, intrabar_range/2)
                low_price = close_price - np.random.uniform(0, intrabar_range/2)
                
            # Ensure high is always highest and low is always lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Store prices for this minute
            day_opens.append(open_price)
            day_highs.append(high_price)
            day_lows.append(low_price)
            day_closes.append(close_price)
            
            # Update current price
            current_price = close_price
        
        # Store the last close for next day's open
        prev_day_close = day_closes[-1] if day_closes else base_price
        
        # Generate U-shaped volume profile
        minute_indices = np.arange(minutes_in_day)
        
        # Normalized time in [0,1]
        normalized_time = minute_indices / minutes_in_day
        
        # U-shape function: higher at beginning and end, lower in the middle
        u_shape = 1.5 - 1.0 * np.sin(normalized_time * np.pi)**2
        
        # Base volumes for the day with U-shape
        day_base_volumes = base_volume * u_shape
        
        # Add randomness to volume
        day_volumes = day_base_volumes * np.random.lognormal(0, volume_noise, size=minutes_in_day)
        
        # Add 2-3 volume spikes per day
        num_spikes = np.random.randint(2, 4)
        
        for _ in range(num_spikes):
            # Random spike location, avoiding first and last 15 minutes
            if minutes_in_day <= 30:  # Handle short days
                spike_idx = np.random.randint(0, minutes_in_day)
            else:
                spike_idx = np.random.randint(15, minutes_in_day - 15)
            
            # Spike multiplier between 6x and 10x (increased from 3-5x)
            spike_multiplier = np.random.uniform(6, 10)
            
            # Apply spike
            day_volumes[spike_idx] *= spike_multiplier
            
            # Add elevated volume around the spike (2-3 minutes before and after)
            window = min(np.random.randint(2, 4), spike_idx, minutes_in_day - spike_idx - 1)
            for j in range(max(0, spike_idx-window), min(minutes_in_day, spike_idx+window+1)):
                if j != spike_idx:
                    # Taper off effect as we move away from spike
                    distance = abs(j - spike_idx)
                    taper = (window + 1 - distance) / (window + 1)
                    day_volumes[j] *= (1 + (spike_multiplier - 1) * taper * 0.4)
        
        # Extend our data lists
        all_opens.extend(day_opens)
        all_highs.extend(day_highs)
        all_lows.extend(day_lows)
        all_closes.extend(day_closes)
        all_volumes.extend(day_volumes)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'open': all_opens,
        'high': all_highs,
        'low': all_lows,
        'close': all_closes,
        'volume': all_volumes,
        'is_synthetic': True  # Flag as synthetic data
    }, index=timestamps)
    
    # Ensure the DataFrame is sorted by time
    df = df.sort_index()
    
    return df

def save_synthetic_data(df, filepath='data/synthetic_es_ohlcv.csv'):
    """Save synthetic data to CSV file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the data
    df.to_csv(filepath)
    print(f"Synthetic data saved to {filepath}")
    
    return filepath

if __name__ == "__main__":
    # Generate synthetic data for 5 trading days
    print("Generating synthetic OHLCV data...")
    df = generate_synthetic_ohlcv(start_date='2024-06-03', num_days=5, seed=42)
    
    # Convert index.date to list and then get unique values
    unique_dates = pd.Series([d.date() for d in df.index]).unique()
    
    print(f"Generated {len(df)} records spanning {len(unique_dates)} trading days")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Save to CSV
    save_synthetic_data(df)
    
    # Print sample
    print("\nSample data:")
    print(df.head()) 