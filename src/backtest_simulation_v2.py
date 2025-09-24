"""
Advanced Volume Cluster Trading Strategy - Version 2.0
Incorporates lessons learned from optimization analysis:
- Enhanced signal filtering (minimum signal strength threshold)
- Momentum confirmation for all trades
- Adaptive exit timing based on signal strength
- Volume momentum confirmation
- Enhanced stop loss with trailing capability
"""

# Imports
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import pytz
from volume_cluster import identify_volume_clusters
from tqdm import tqdm

# Enhanced Params
DATA_PATH = "../data/es_ohlcv_real.csv"
VOLUME_THRESHOLD = 5.0  # Keep high quality threshold
MIN_SIGNAL_STRENGTH = 0.6  # Only trade signals with >60% strength
MIN_VOLUME_RATIO = 100.0  # Only trade 100x+ volume clusters
RETENTION_MINUTES = 60
RETEST_ENABLED = True
RETEST_TICK_WINDOW = 3
BASE_POSITION_SIZE = 1.0
MAX_POSITION_SIZE = 2.0
VOLATILITY_LOOKBACK = 20
MOMENTUM_CONFIRMATION = True  # Require momentum confirmation for all trades
TRAILING_STOP = True  # Enable trailing stop loss

def calculate_momentum(df, timestamp, lookback_minutes=5):
    """Calculate short-term momentum before the signal"""
    start_time = timestamp - timedelta(minutes=lookback_minutes)
    momentum_data = df.loc[start_time:timestamp]
    
    if len(momentum_data) < 2:
        return 0
    
    price_change = (momentum_data['close'].iloc[-1] - momentum_data['close'].iloc[0]) / momentum_data['close'].iloc[0]
    return price_change

def is_valid_trading_time(timestamp):
    """Check if timestamp falls within allowed CET trading hours: 14:00-17:30 CET"""
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Check if within allowed window: 14:00-17:30 CET
    if hour == 14 or hour == 15 or hour == 16 or (hour == 17 and minute < 30):
        return True
    
    return False

def calculate_signal_strength_v2(modal_position, volume_ratio, momentum):
    """Enhanced signal strength calculation including momentum"""
    # Base strength from modal position
    if modal_position <= 0.25:
        position_strength = 1.0 - (modal_position / 0.25)
    elif modal_position >= 0.75:
        position_strength = (modal_position - 0.75) / 0.25
    else:
        return 0  # No trade
    
    # Volume strength (normalized to 0-1)
    volume_strength = min(volume_ratio / 200.0, 1.0)  # Cap at 200x
    
    # Momentum strength (for longs, positive momentum helps; for shorts, negative helps)
    if modal_position <= 0.25:  # Long signal
        momentum_strength = max(0, momentum * 10)  # Positive momentum good for longs
    else:  # Short signal
        momentum_strength = max(0, -momentum * 10)  # Negative momentum good for shorts
    
    momentum_strength = min(momentum_strength, 1.0)  # Cap at 1.0
    
    # Combined signal strength (weighted average)
    combined_strength = (0.5 * position_strength + 0.3 * volume_strength + 0.2 * momentum_strength)
    return combined_strength

# Load full dataset
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")

# Prepare
results = []

# Iterate day-by-day
for day, group in tqdm(df.groupby(df.index.date)):
    intraday_df = group.copy()

    clusters_df = identify_volume_clusters(intraday_df, volume_multiplier=VOLUME_THRESHOLD)
    
    if clusters_df.empty:
        continue

    # Rolling top-2 filter with enhanced filtering
    daily_clusters = []
    clusters_sorted = clusters_df.sort_index()
    
    for cluster_time, cluster_row in clusters_sorted.iterrows():
        
        if not is_valid_trading_time(cluster_time):
            continue
            
        cluster_volume = cluster_row['volume']
        avg_volume = intraday_df['volume'].mean()
        volume_ratio = cluster_volume / avg_volume if avg_volume > 0 else 0
        
        # Early filter: Skip low volume clusters
        if volume_ratio < MIN_VOLUME_RATIO:
            continue
        
        current_cluster = {
            'timestamp': cluster_time,
            'volume_ratio': volume_ratio,
            'cluster_data': cluster_row
        }
        daily_clusters.append(current_cluster)
        
        daily_clusters_sorted = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        top_2_today = daily_clusters_sorted[:2]
        current_cluster_in_top2 = any(c['timestamp'] == cluster_time for c in top_2_today)
        
        if not current_cluster_in_top2:
            continue
        
        cluster_slice = intraday_df.loc[cluster_time : cluster_time + timedelta(minutes=14)]
        
        if cluster_slice.empty:
            continue
            
        modal_price = cluster_slice["close"].round(2).mode()
        if len(modal_price) == 0:
            continue
        modal_price = modal_price[0]

        price_low = cluster_slice["low"].min()
        price_high = cluster_slice["high"].max()
        pos = (modal_price - price_low) / (price_high - price_low + 1e-9)

        # Calculate momentum before the cluster
        momentum = calculate_momentum(intraday_df, cluster_time)
        
        # Enhanced signal strength calculation
        signal_strength = calculate_signal_strength_v2(pos, volume_ratio, momentum)
        
        # Early filter: Skip weak signals
        if signal_strength < MIN_SIGNAL_STRENGTH:
            continue

        # Determine direction with tighter thresholds
        if pos <= 0.25:
            direction = "long"
        elif pos >= 0.75:
            direction = "short"
        else:
            continue

        # Calculate position size (enhanced)
        volume_strength = min(volume_ratio / 150.0, 2.0)
        position_multiplier = BASE_POSITION_SIZE + (signal_strength * volume_strength * 0.4)
        position_multiplier = min(position_multiplier, MAX_POSITION_SIZE)

        # Find retest
        entry_time = None
        retest_time = None
        for t, row in intraday_df.loc[cluster_time:].iterrows():
            if abs(row["close"] - modal_price) <= 0.75 and RETEST_ENABLED:
                retest_time = t
                break
        if not RETEST_ENABLED:
            retest_time = cluster_time + timedelta(minutes=1)

        if retest_time is None:
            continue

        # Enhanced confirmation logic for ALL trades
        short_confirmed = False
        if MOMENTUM_CONFIRMATION:
            future_candles = intraday_df[intraday_df.index > retest_time]
            if future_candles.empty:
                continue
                
            confirmation_candle = future_candles.iloc[0]
            confirmation_close = confirmation_candle["close"]
            
            if direction == "long":
                # For longs, require confirmation candle to close above modal price
                if confirmation_close <= modal_price:
                    continue
                entry_time = confirmation_candle.name
            else:
                # For shorts, require confirmation candle to close below modal price
                if confirmation_close >= modal_price:
                    continue
                entry_time = confirmation_candle.name
                short_confirmed = True
        else:
            entry_time = retest_time

        # Entry price
        entry_data = intraday_df.loc[entry_time]
        if isinstance(entry_data, pd.Series):
            entry_price = float(entry_data["close"])
        else:
            entry_price = float(entry_data["close"].iloc[0])

        # Enhanced stop loss calculation
        recent_data = df.loc[:entry_time].tail(VOLATILITY_LOOKBACK * 24 * 60)
        if len(recent_data) > 100:
            price_changes = recent_data['close'].pct_change().dropna()
            volatility = price_changes.std()
            stop_distance = 1.5 * volatility * entry_price  # Tighter 1.5-sigma stop
        else:
            stop_distance = 0.008 * entry_price  # 0.8% default stop

        if direction == "long":
            initial_stop_price = entry_price - stop_distance
        else:
            initial_stop_price = entry_price + stop_distance

        # Adaptive exit timing based on enhanced signal strength
        base_exit_1 = 15  # Shorter base exits
        base_exit_2 = 35
        
        exit_1_time = base_exit_1 + int(signal_strength * 15)  # 15-30 minutes
        exit_2_time = base_exit_2 + int(signal_strength * 20)  # 35-55 minutes

        exit_times = [
            entry_time + timedelta(minutes=exit_1_time),
            entry_time + timedelta(minutes=exit_2_time),
        ]
        
        # Enhanced exit processing with trailing stop
        exit_data = []
        exit_weights = [0.7, 0.3]
        stopped_out = False
        trailing_stop_price = initial_stop_price
        
        for i, target_exit_time in enumerate(exit_times):
            if not stopped_out:
                period_data = intraday_df[
                    (intraday_df.index > entry_time) & 
                    (intraday_df.index <= target_exit_time)
                ]
                
                # Enhanced stop loss with trailing capability
                for period_time, period_row in period_data.iterrows():
                    if TRAILING_STOP and direction == "long":
                        # Update trailing stop for longs
                        new_stop = period_row["close"] - stop_distance
                        trailing_stop_price = max(trailing_stop_price, new_stop)
                        
                        if period_row["low"] <= trailing_stop_price:
                            exit_price = trailing_stop_price
                            actual_exit_time = period_time
                            stopped_out = True
                            break
                    elif TRAILING_STOP and direction == "short":
                        # Update trailing stop for shorts
                        new_stop = period_row["close"] + stop_distance
                        trailing_stop_price = min(trailing_stop_price, new_stop)
                        
                        if period_row["high"] >= trailing_stop_price:
                            exit_price = trailing_stop_price
                            actual_exit_time = period_time
                            stopped_out = True
                            break
                    elif not TRAILING_STOP:
                        # Standard stop loss
                        if direction == "long" and period_row["low"] <= initial_stop_price:
                            exit_price = initial_stop_price
                            actual_exit_time = period_time
                            stopped_out = True
                            break
                        elif direction == "short" and period_row["high"] >= initial_stop_price:
                            exit_price = initial_stop_price
                            actual_exit_time = period_time
                            stopped_out = True
                            break
            
            if stopped_out:
                actual_exit_time = actual_exit_time
                exit_price = exit_price
            else:
                future_data = intraday_df[intraday_df.index >= target_exit_time]
                if future_data.empty:
                    if i == 0:
                        break
                    else:
                        last_available_data = intraday_df.tail(1)
                        actual_exit_time = last_available_data.index[0]
                        exit_price = float(last_available_data.iloc[0]["close"])
                else:
                    actual_exit_time = future_data.index[0]
                    exit_price = float(future_data.iloc[0]["close"])
            
            # Calculate returns
            if direction == "short":
                tranche_return = (entry_price - exit_price) / entry_price
            elif direction == "long":
                tranche_return = (exit_price - entry_price) / entry_price
            
            position_adjusted_return = tranche_return * position_multiplier
            
            exit_data.append({
                'time': actual_exit_time,
                'price': exit_price,
                'return': tranche_return,
                'position_adjusted_return': position_adjusted_return,
                'stopped_out': stopped_out
            })
        
        if len(exit_data) == 0:
            continue
            
        while len(exit_data) < 2:
            exit_data.append(exit_data[-1].copy())
        
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
            "momentum": momentum,
            "signal_strength": signal_strength,
            "position_multiplier": position_multiplier,
            "initial_stop_price": initial_stop_price,
            "final_stop_price": trailing_stop_price if TRAILING_STOP else initial_stop_price,
            "stopped_out": exit_data[0]['stopped_out'] or exit_data[1]['stopped_out'],
            "short_confirmed": short_confirmed,
            
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
result_df.to_csv("../data/backtest_results_v2.csv", index=False)

# Summary
if len(result_df) > 0:
    mean_ret = result_df["blended_return"].mean()
    win_rate = (result_df["blended_return"] > 0).mean()
    long_trades = (result_df["direction"] == "long").sum()
    short_trades = (result_df["direction"] == "short").sum()
    
    long_win_rate = (result_df[result_df["direction"] == "long"]["blended_return"] > 0).mean() if long_trades > 0 else 0
    short_win_rate = (result_df[result_df["direction"] == "short"]["blended_return"] > 0).mean() if short_trades > 0 else 0
    
    avg_exit1_return = result_df["exit_1_return"].mean()
    avg_exit2_return = result_df["exit_2_return"].mean()
    avg_signal_strength = result_df["signal_strength"].mean()
    avg_volume_ratio = result_df["volume_ratio"].mean()
    
    print(f"=== ENHANCED STRATEGY V2.0 RESULTS ===")
    print(f"Simulated Trades: {len(result_df)}")
    print(f"Mean Blended Return: {mean_ret:.4%}")
    print(f"Win Rate (Blended): {win_rate:.2%}")
    print(f"Long Trades: {long_trades} (Win Rate: {long_win_rate:.2%})")
    print(f"Short Trades: {short_trades} (Win Rate: {short_win_rate:.2%})")
    print(f"Average Signal Strength: {avg_signal_strength:.3f}")
    print(f"Average Volume Ratio: {avg_volume_ratio:.1f}x")
    print(f"")
    print(f"Average Returns by Exit Tranche:")
    print(f"  Exit 1 (70% @ adaptive): {avg_exit1_return:.4%}")
    print(f"  Exit 2 (30% @ adaptive): {avg_exit2_return:.4%}")
else:
    print("No trades found in enhanced backtest simulation.") 