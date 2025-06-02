"""
Volume Cluster Trading Strategy - Version 3.0 (Return Optimization Focus)
Addresses the negative mean return issue through:
- Looser selection criteria for more opportunities
- Profit targets with asymmetric risk/reward (2:1 ratio)
- Transaction cost accounting
- Enhanced exit logic with early profit taking
- Tighter stop losses for better risk management
"""

# Imports
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import pytz
from volume_cluster import identify_volume_clusters
from tqdm import tqdm

# Enhanced Params for Return Optimization
DATA_PATH = "../data/es_ohlcv_real.csv"
VOLUME_THRESHOLD = 4.0  # Reduced from 5.0 for more opportunities
MIN_SIGNAL_STRENGTH = 0.45  # Reduced from 0.6 for broader selection
MIN_VOLUME_RATIO = 60.0  # Reduced from 100.0 for more trades
RETENTION_MINUTES = 60
RETEST_ENABLED = True
RETEST_TICK_WINDOW = 3
BASE_POSITION_SIZE = 1.0
MAX_POSITION_SIZE = 2.0
VOLATILITY_LOOKBACK = 20

# New Return Optimization Features
PROFIT_TARGET_RATIO = 2.0  # 2:1 reward/risk ratio
COMMISSION_PER_CONTRACT = 2.50  # Round-trip commission
SLIPPAGE_TICKS = 0.75  # Expected slippage in ticks
TICK_VALUE = 12.50  # ES tick value in USD
USE_PROFIT_TARGETS = True
TIGHTER_STOPS = True  # Use 1.0-sigma instead of 1.5-sigma

def calculate_transaction_costs(position_size=1.0):
    """Calculate total transaction costs including commission and slippage"""
    commission_cost = COMMISSION_PER_CONTRACT * position_size
    slippage_cost = SLIPPAGE_TICKS * TICK_VALUE * position_size
    return (commission_cost + slippage_cost) / (5000 * position_size)  # As percentage of notional

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

def calculate_signal_strength_v3(modal_position, volume_ratio, momentum):
    """Enhanced signal strength calculation with looser criteria"""
    # Base strength from modal position (looser thresholds)
    if modal_position <= 0.28:  # Loosened from 0.25
        position_strength = 1.0 - (modal_position / 0.28)
    elif modal_position >= 0.72:  # Loosened from 0.75
        position_strength = (modal_position - 0.72) / 0.28
    else:
        return 0  # No trade
    
    # Volume strength (normalized to 0-1)
    volume_strength = min(volume_ratio / 150.0, 1.0)  # Reduced from 200x
    
    # Momentum strength
    if modal_position <= 0.28:  # Long signal
        momentum_strength = max(0, momentum * 8)  # Reduced sensitivity
    else:  # Short signal
        momentum_strength = max(0, -momentum * 8)
    
    momentum_strength = min(momentum_strength, 1.0)
    
    # Combined signal strength (weighted average)
    combined_strength = (0.5 * position_strength + 0.3 * volume_strength + 0.2 * momentum_strength)
    return combined_strength

def calculate_profit_target_and_stop(entry_price, direction, signal_strength, volatility):
    """Calculate profit target and stop loss with asymmetric risk/reward"""
    if TIGHTER_STOPS:
        stop_distance = 1.0 * volatility * entry_price  # 1-sigma instead of 1.5
    else:
        stop_distance = 1.5 * volatility * entry_price
    
    # Minimum stop distance
    min_stop = 0.005 * entry_price  # 0.5% minimum
    stop_distance = max(stop_distance, min_stop)
    
    # Profit target based on risk/reward ratio
    profit_distance = stop_distance * PROFIT_TARGET_RATIO
    
    if direction == "long":
        stop_price = entry_price - stop_distance
        profit_target = entry_price + profit_distance
    else:
        stop_price = entry_price + stop_distance
        profit_target = entry_price - profit_distance
    
    return profit_target, stop_price, stop_distance

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

    # Rolling top-3 filter (increased from 2 for more opportunities)
    daily_clusters = []
    clusters_sorted = clusters_df.sort_index()
    
    for cluster_time, cluster_row in clusters_sorted.iterrows():
        
        if not is_valid_trading_time(cluster_time):
            continue
            
        cluster_volume = cluster_row['volume']
        avg_volume = intraday_df['volume'].mean()
        volume_ratio = cluster_volume / avg_volume if avg_volume > 0 else 0
        
        # Early filter: Skip low volume clusters (loosened threshold)
        if volume_ratio < MIN_VOLUME_RATIO:
            continue
        
        current_cluster = {
            'timestamp': cluster_time,
            'volume_ratio': volume_ratio,
            'cluster_data': cluster_row
        }
        daily_clusters.append(current_cluster)
        
        daily_clusters_sorted = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
        top_3_today = daily_clusters_sorted[:3]  # Increased from 2
        current_cluster_in_top3 = any(c['timestamp'] == cluster_time for c in top_3_today)
        
        if not current_cluster_in_top3:
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
        signal_strength = calculate_signal_strength_v3(pos, volume_ratio, momentum)
        
        # Early filter: Skip weak signals (loosened threshold)
        if signal_strength < MIN_SIGNAL_STRENGTH:
            continue

        # Determine direction with loosened thresholds
        if pos <= 0.28:  # Loosened from 0.25
            direction = "long"
        elif pos >= 0.72:  # Loosened from 0.75
            direction = "short"
        else:
            continue

        # Calculate position size
        volume_strength = min(volume_ratio / 100.0, 2.0)  # Adjusted
        position_multiplier = BASE_POSITION_SIZE + (signal_strength * volume_strength * 0.3)
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

        # Simplified confirmation logic (optional, not mandatory)
        short_confirmed = False
        future_candles = intraday_df[intraday_df.index > retest_time]
        if future_candles.empty:
            continue
            
        # Entry after retest (simplified)
        entry_time = future_candles.index[0]

        # Entry price
        entry_data = intraday_df.loc[entry_time]
        if isinstance(entry_data, pd.Series):
            entry_price = float(entry_data["close"])
        else:
            entry_price = float(entry_data["close"].iloc[0])

        # Calculate volatility for stops and targets
        recent_data = df.loc[:entry_time].tail(VOLATILITY_LOOKBACK * 24 * 60)
        if len(recent_data) > 100:
            price_changes = recent_data['close'].pct_change().dropna()
            volatility = price_changes.std()
        else:
            volatility = 0.01  # 1% default

        # Calculate profit target and stop loss
        profit_target, stop_price, stop_distance = calculate_profit_target_and_stop(
            entry_price, direction, signal_strength, volatility
        )

        # Enhanced exit processing with profit targets
        exit_data = []
        stopped_out = False
        profit_hit = False
        
        # Define multiple exit scenarios
        max_hold_time = entry_time + timedelta(minutes=60)  # Maximum hold
        
        # Look for exit conditions
        future_data = intraday_df[intraday_df.index > entry_time]
        exit_price = None
        actual_exit_time = None
        exit_reason = "time"
        
        for period_time, period_row in future_data.iterrows():
            if period_time > max_hold_time:
                # Time exit
                exit_price = period_row["close"]
                actual_exit_time = period_time
                exit_reason = "time"
                break
                
            if USE_PROFIT_TARGETS:
                # Check profit target
                if direction == "long" and period_row["high"] >= profit_target:
                    exit_price = profit_target
                    actual_exit_time = period_time
                    exit_reason = "profit_target"
                    profit_hit = True
                    break
                elif direction == "short" and period_row["low"] <= profit_target:
                    exit_price = profit_target
                    actual_exit_time = period_time
                    exit_reason = "profit_target"
                    profit_hit = True
                    break
            
            # Check stop loss
            if direction == "long" and period_row["low"] <= stop_price:
                exit_price = stop_price
                actual_exit_time = period_time
                exit_reason = "stop_loss"
                stopped_out = True
                break
            elif direction == "short" and period_row["high"] >= stop_price:
                exit_price = stop_price
                actual_exit_time = period_time
                exit_reason = "stop_loss"
                stopped_out = True
                break
        
        # If no exit found, use end of day
        if exit_price is None:
            last_data = future_data.tail(1)
            if not last_data.empty:
                exit_price = float(last_data.iloc[0]["close"])
                actual_exit_time = last_data.index[0]
                exit_reason = "end_of_data"
            else:
                continue
        
        # Calculate returns
        if direction == "short":
            gross_return = (entry_price - exit_price) / entry_price
        elif direction == "long":
            gross_return = (exit_price - entry_price) / entry_price
        
        # Account for transaction costs
        transaction_cost = calculate_transaction_costs(position_multiplier)
        net_return = gross_return - transaction_cost
        
        # Position-adjusted return
        position_adjusted_return = net_return * position_multiplier

        results.append({
            "date": day,
            "entry_time": entry_time,
            "exit_time": actual_exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "modal_price": modal_price,
            "modal_position": pos,
            "volume_ratio": volume_ratio,
            "momentum": momentum,
            "signal_strength": signal_strength,
            "position_multiplier": position_multiplier,
            "profit_target": profit_target,
            "stop_price": stop_price,
            "exit_reason": exit_reason,
            "stopped_out": stopped_out,
            "profit_hit": profit_hit,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "position_adjusted_return": position_adjusted_return,
        })

# Export results
result_df = pd.DataFrame(results)
result_df.to_csv("../data/backtest_results_v3.csv", index=False)

# Enhanced Summary
if len(result_df) > 0:
    mean_gross_ret = result_df["gross_return"].mean()
    mean_net_ret = result_df["net_return"].mean()
    mean_pos_adj_ret = result_df["position_adjusted_return"].mean()
    win_rate = (result_df["net_return"] > 0).mean()
    
    # Exit reason analysis
    exit_reasons = result_df["exit_reason"].value_counts()
    profit_target_rate = (result_df["profit_hit"] == True).mean()
    stop_loss_rate = (result_df["stopped_out"] == True).mean()
    
    # Risk/reward analysis
    winners = result_df[result_df["net_return"] > 0]["net_return"]
    losers = result_df[result_df["net_return"] < 0]["net_return"]
    
    avg_winner = winners.mean() if len(winners) > 0 else 0
    avg_loser = losers.mean() if len(losers) > 0 else 0
    risk_reward_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else 0
    
    print(f"=== RETURN OPTIMIZATION V3.0 RESULTS ===")
    print(f"Simulated Trades: {len(result_df)}")
    print(f"")
    print(f"RETURN ANALYSIS:")
    print(f"  Mean Gross Return: {mean_gross_ret:.4%}")
    print(f"  Mean Net Return: {mean_net_ret:.4%}")
    print(f"  Mean Position-Adjusted Return: {mean_pos_adj_ret:.4%}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"")
    print(f"RISK/REWARD METRICS:")
    print(f"  Average Winner: {avg_winner:.4%}")
    print(f"  Average Loser: {avg_loser:.4%}")
    print(f"  Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
    print(f"")
    print(f"EXIT ANALYSIS:")
    print(f"  Profit Target Hit: {profit_target_rate:.2%}")
    print(f"  Stop Loss Hit: {stop_loss_rate:.2%}")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(result_df)*100:.1f}%)")
    print(f"")
    print(f"TRANSACTION COST IMPACT:")
    print(f"  Average Transaction Cost: {result_df['transaction_cost'].mean():.4%}")
    print(f"  Gross vs Net Return Difference: {(mean_gross_ret - mean_net_ret):.4%}")
    
else:
    print("No trades found in enhanced backtest simulation.") 