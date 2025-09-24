"""
Simulates backtesting of the volume cluster trading strategy.
This script replays market data day-by-day, minute-by-minute, to:
- Detect volume clusters using your existing logic
- Identify modal price (POC)
- Wait for optional retest
- Enter a short trade
- Exit after fixed time window (e.g., 30 minutes)
- Log results

Outputs:
- backtest_results.csv (all trades)
- summary stats in console
"""

# Imports
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import pytz
from volume_cluster import identify_volume_clusters  # reuse this
from tqdm import tqdm

# Params
DATA_PATH = "../data/es_ohlcv_real.csv"  # Fixed path from src/ directory
VOLUME_THRESHOLD = 5.0  # Increased from 4.0 for higher quality signals
RETENTION_MINUTES = 60  # Increased from 30 for longer hold time
RETEST_ENABLED = True
RETEST_TICK_WINDOW = 3
BASE_POSITION_SIZE = 1.0  # Base position size
MAX_POSITION_SIZE = 2.0   # Maximum position multiplier
VOLATILITY_LOOKBACK = 20  # Days for volatility calculation

def is_valid_trading_time(timestamp):
    """
    Check if timestamp falls within allowed CET trading hours:
    14:00-17:30 CET (data is already in CET/CEST time)
    """
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Check if within allowed window: 14:00-17:30 CET
    if hour == 14 or hour == 15 or hour == 16 or (hour == 17 and minute < 30):
        return True
    
    return False

# Load full dataset
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")  # Changed from timestamp to datetime
# Remove symbol filtering since the data doesn't have a symbol column
# df = df[df["symbol"] == "ES"]  

# Prepare
results = []

# Iterate day-by-day
for day, group in tqdm(df.groupby(df.index.date)):
    intraday_df = group.copy()

    # Use the corrected function call - identify_volume_clusters only needs df and volume_multiplier
    clusters_df = identify_volume_clusters(intraday_df, volume_multiplier=VOLUME_THRESHOLD)
    
    # Skip if no clusters found
    if clusters_df.empty:
        continue

    # Rolling top-2 filter: Process clusters chronologically within the day
    daily_clusters = []
    
    # Sort clusters by timestamp to process chronologically
    clusters_sorted = clusters_df.sort_index()
    
    for cluster_time, cluster_row in clusters_sorted.iterrows():
        
        # Time filter: Skip clusters outside valid trading hours
        if not is_valid_trading_time(cluster_time):
            continue
            
        # Calculate volume ratio for current cluster
        cluster_volume = cluster_row['volume']
        avg_volume = intraday_df['volume'].mean()
        volume_ratio = cluster_volume / avg_volume if avg_volume > 0 else 0
        
        # Add current cluster to daily list
        current_cluster = {
            'timestamp': cluster_time,
            'volume_ratio': volume_ratio,
            'cluster_data': cluster_row
        }
        daily_clusters.append(current_cluster)
        
        # Sort clusters seen so far today by volume ratio (descending)
        daily_clusters_sorted = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        
        # Check if current cluster is in top 2 so far today
        top_2_today = daily_clusters_sorted[:2]
        current_cluster_in_top2 = any(c['timestamp'] == cluster_time for c in top_2_today)
        
        if not current_cluster_in_top2:
            continue  # Skip if not in rolling top-2
        
        cluster_slice = intraday_df.loc[cluster_time : cluster_time + timedelta(minutes=14)]
        
        # Check if cluster_slice is empty
        if cluster_slice.empty:
            continue
            
        modal_price = cluster_slice["close"].round(2).mode()
        if len(modal_price) == 0:
            continue
        modal_price = modal_price[0]

        # Determine trade direction from cluster shape
        price_low = cluster_slice["low"].min()
        price_high = cluster_slice["high"].max()

        # Normalize modal price between 0 and 1
        pos = (modal_price - price_low) / (price_high - price_low + 1e-9)

        # Tighter thresholds for higher quality signals
        if pos <= 0.25:  # Tightened from 0.35 - only strongest support signals
            direction = "long"
            signal_strength = 1.0 - (pos / 0.25)  # Stronger signal = higher strength
        elif pos >= 0.75:  # Tightened from 0.65 - only strongest resistance signals
            direction = "short"
            signal_strength = (pos - 0.75) / 0.25  # Stronger signal = higher strength
        else:
            continue  # skip ambiguous clusters

        # Calculate position size based on signal strength and volume ratio
        volume_strength = min(volume_ratio / 100.0, 2.0)  # Cap at 2x for 100x+ volume
        position_multiplier = BASE_POSITION_SIZE + (signal_strength * volume_strength * 0.5)
        position_multiplier = min(position_multiplier, MAX_POSITION_SIZE)  # Cap maximum size

        entry_time = None
        retest_time = None
        for t, row in intraday_df.loc[cluster_time:].iterrows():
            if abs(row["close"] - modal_price) <= 0.75 and RETEST_ENABLED:
                retest_time = t
                break
        if not RETEST_ENABLED:
            retest_time = cluster_time + timedelta(minutes=1)

        # Only take trades where modal price is retested
        if retest_time is None:
            continue
            
        # Confirmation logic: only required for short trades
        short_confirmed = True  # Default for longs
        if direction == "short":
            # For shorts, require confirmation candle where close is below modal price
            future_candles = intraday_df[intraday_df.index > retest_time]
            if future_candles.empty:
                continue
                
            confirmation_candle = future_candles.iloc[0]
            confirmation_close = confirmation_candle["close"]
            
            # Check if confirmation candle closes below modal price
            if confirmation_close >= modal_price:
                continue  # Skip if confirmation fails
                
            # Set entry time to confirmation candle time for shorts
            entry_time = confirmation_candle.name
            short_confirmed = True
        else:
            # For longs, enter directly after retest without confirmation
            entry_time = retest_time
            short_confirmed = False  # Not applicable for longs

        # Handle potential duplicate timestamps by taking the first match
        entry_data = intraday_df.loc[entry_time]
        if isinstance(entry_data, pd.Series):
            entry_price = float(entry_data["close"])
        else:  # DataFrame with multiple rows for same timestamp
            entry_price = float(entry_data["close"].iloc[0])

        # Calculate volatility-based stop loss
        recent_data = df.loc[:entry_time].tail(VOLATILITY_LOOKBACK * 24 * 60)  # Last N days of minute data
        if len(recent_data) > 100:  # Ensure sufficient data
            price_changes = recent_data['close'].pct_change().dropna()
            volatility = price_changes.std()
            stop_distance = 2.0 * volatility * entry_price  # 2-sigma stop
        else:
            stop_distance = 0.01 * entry_price  # 1% default stop
        
        # Set stop loss levels
        if direction == "long":
            stop_loss_price = entry_price - stop_distance
        else:
            stop_loss_price = entry_price + stop_distance

        # Dynamic exit timing based on signal strength
        base_exit_1 = 20
        base_exit_2 = 45
        
        # Stronger signals get longer holding periods
        exit_1_time = base_exit_1 + int(signal_strength * 10)  # 20-30 minutes
        exit_2_time = base_exit_2 + int(signal_strength * 15)  # 45-60 minutes

        # Define exit times for rolling exits (2-part strategy)
        exit_times = [
            entry_time + timedelta(minutes=exit_1_time),  # Dynamic 70% exit
            entry_time + timedelta(minutes=exit_2_time),  # Dynamic 30% exit
        ]
        
        # Find closest available exit times and prices
        exit_data = []
        exit_weights = [0.7, 0.3]  # Updated to 70%/30% weighting
        stopped_out = False
        
        for i, target_exit_time in enumerate(exit_times):
            # Check for stop loss hit during this period
            if not stopped_out:
                period_data = intraday_df[
                    (intraday_df.index > entry_time) & 
                    (intraday_df.index <= target_exit_time)
                ]
                
                for period_time, period_row in period_data.iterrows():
                    if direction == "long" and period_row["low"] <= stop_loss_price:
                        # Long position stopped out
                        exit_price = stop_loss_price
                        actual_exit_time = period_time
                        stopped_out = True
                        break
                    elif direction == "short" and period_row["high"] >= stop_loss_price:
                        # Short position stopped out
                        exit_price = stop_loss_price
                        actual_exit_time = period_time
                        stopped_out = True
                        break
            
            if stopped_out:
                # Use stop loss exit for remaining tranches
                actual_exit_time = actual_exit_time
                exit_price = exit_price
            else:
                # Find the closest available exit time if exact time doesn't exist
                future_data = intraday_df[intraday_df.index >= target_exit_time]
                if future_data.empty:
                    # If no future data available, use the last available price in the day
                    if i == 0:
                        # If we can't even get the first exit, skip this trade entirely
                        break
                    else:
                        # Use the last recorded price for remaining exits
                        last_available_data = intraday_df.tail(1)
                        actual_exit_time = last_available_data.index[0]
                        exit_price = float(last_available_data.iloc[0]["close"])
                else:
                    actual_exit_time = future_data.index[0]
                    exit_price = float(future_data.iloc[0]["close"])
            
            # Calculate return for this tranche based on direction
            if direction == "short":
                tranche_return = (entry_price - exit_price) / entry_price
            elif direction == "long":
                tranche_return = (exit_price - entry_price) / entry_price
            
            # Apply position sizing to returns
            position_adjusted_return = tranche_return * position_multiplier
            
            exit_data.append({
                'time': actual_exit_time,
                'price': exit_price,
                'return': tranche_return,
                'position_adjusted_return': position_adjusted_return,
                'stopped_out': stopped_out
            })
        
        # Skip only if we can't get even the first exit
        if len(exit_data) == 0:
            continue
            
        # If we have fewer than 2 exits, repeat the last exit for remaining tranches
        while len(exit_data) < 2:
            exit_data.append(exit_data[-1].copy())  # Copy the last available exit
        
        # Calculate blended return as weighted average (position-adjusted)
        blended_return = sum(exit_weights[i] * exit_data[i]['position_adjusted_return'] for i in range(2))
        raw_blended_return = sum(exit_weights[i] * exit_data[i]['return'] for i in range(2))

        results.append({
            "date": day,
            "entry_time": entry_time,
            "retest_time": retest_time,
            "entry_price": entry_price,
            "direction": direction,
            "modal_price": modal_price,
            "modal_position": pos,
            "volume_ratio": volume_ratio,
            "retested": RETEST_ENABLED,
            "short_confirmed": short_confirmed,
            "signal_strength": signal_strength,
            "position_multiplier": position_multiplier,
            "stop_loss_price": stop_loss_price,
            "stopped_out": exit_data[0]['stopped_out'] or exit_data[1]['stopped_out'],
            
            # Rolling exit data
            "exit_1_time": exit_data[0]['time'],
            "exit_1_price": exit_data[0]['price'],
            "exit_1_return": exit_data[0]['return'],
            "exit_1_position_return": exit_data[0]['position_adjusted_return'],
            
            "exit_2_time": exit_data[1]['time'],
            "exit_2_price": exit_data[1]['price'], 
            "exit_2_return": exit_data[1]['return'],
            "exit_2_position_return": exit_data[1]['position_adjusted_return'],
            
            "blended_return": blended_return,
            "raw_blended_return": raw_blended_return,
        })

# Export results
result_df = pd.DataFrame(results)
result_df.to_csv("../data/backtest_results.csv", index=False)  # Fixed path

# Summary
if len(result_df) > 0:
    mean_ret = result_df["blended_return"].mean()
    win_rate = (result_df["blended_return"] > 0).mean()
    long_trades = (result_df["direction"] == "long").sum()
    short_trades = (result_df["direction"] == "short").sum()
    
    # Win rates by direction
    long_win_rate = (result_df[result_df["direction"] == "long"]["blended_return"] > 0).mean() if long_trades > 0 else 0
    short_win_rate = (result_df[result_df["direction"] == "short"]["blended_return"] > 0).mean() if short_trades > 0 else 0
    
    # Average returns per exit tranche
    avg_exit1_return = result_df["exit_1_return"].mean()
    avg_exit2_return = result_df["exit_2_return"].mean()
    
    print(f"Simulated Trades: {len(result_df)}")
    print(f"Mean Blended Return: {mean_ret:.4%}")
    print(f"Win Rate (Blended): {win_rate:.2%}")
    print(f"Long Trades: {long_trades} (Win Rate: {long_win_rate:.2%})")
    print(f"Short Trades: {short_trades} (Win Rate: {short_win_rate:.2%})")
    print(f"")
    print(f"Average Returns by Exit Tranche:")
    print(f"  Exit 1 (70% @ 20min): {avg_exit1_return:.4%}")
    print(f"  Exit 2 (30% @ 45min): {avg_exit2_return:.4%}")
else:
    print("No trades found in backtest simulation.") 