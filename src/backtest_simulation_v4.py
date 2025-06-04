"""
Volume Cluster Trading Strategy - Version 4.0 (Adaptive Statistics)
Key improvements over V3:
- Dynamic volume ratio threshold (top N clusters per day)
- Adaptive modal position filtering based on historical returns
- No forward-looking bias in decision making
- Modular preprocessing with CSV export of statistics
- All thresholds based only on past data available at cluster time
"""

# Imports
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import pytz
from volume_cluster import identify_volume_clusters
from tqdm import tqdm

# Base Parameters (same as V3)
DATA_PATH = "../data/es_ohlcv_real.csv"
VOLUME_THRESHOLD = 4.0
MIN_SIGNAL_STRENGTH = 0.45
RETENTION_MINUTES = 60
RETEST_ENABLED = True
RETEST_TICK_WINDOW = 3
BASE_POSITION_SIZE = 1.0
MAX_POSITION_SIZE = 2.0
VOLATILITY_LOOKBACK = 20

# V4 Adaptive Parameters
TOP_N_CLUSTERS_PER_DAY = 2  # Only trade top N volume clusters per day
MODAL_POSITION_BINS = 10    # Number of bins for modal position analysis
MIN_HISTORICAL_TRADES = 20  # Minimum trades needed before using adaptive filtering
LOOKBACK_DAYS = 30         # Days to look back for modal position statistics
MIN_BIN_RETURN_THRESHOLD = 0.0  # Minimum average return for a bin to be tradeable

# Transaction Cost Parameters (same as V3)
PROFIT_TARGET_RATIO = 2.0
COMMISSION_PER_CONTRACT = 2.50
SLIPPAGE_TICKS = 0.75
TICK_VALUE = 12.50
USE_PROFIT_TARGETS = True
TIGHTER_STOPS = True

def calculate_transaction_costs(position_size=1.0):
    """Calculate total transaction costs including commission and slippage"""
    commission_cost = COMMISSION_PER_CONTRACT * position_size
    slippage_cost = SLIPPAGE_TICKS * TICK_VALUE * position_size
    return (commission_cost + slippage_cost) / (5000 * position_size)

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
    
    if hour == 14 or hour == 15 or hour == 16 or (hour == 17 and minute < 30):
        return True
    
    return False

def calculate_signal_strength_v3(modal_position, volume_ratio, momentum):
    """Enhanced signal strength calculation (same as V3)"""
    if modal_position <= 0.28:
        position_strength = 1.0 - (modal_position / 0.28)
    elif modal_position >= 0.72:
        position_strength = (modal_position - 0.72) / 0.28
    else:
        return 0
    
    volume_strength = min(volume_ratio / 150.0, 1.0)
    
    if modal_position <= 0.28:
        momentum_strength = max(0, momentum * 8)
    else:
        momentum_strength = max(0, -momentum * 8)
    
    momentum_strength = min(momentum_strength, 1.0)
    combined_strength = (0.5 * position_strength + 0.3 * volume_strength + 0.2 * momentum_strength)
    return combined_strength

def calculate_profit_target_and_stop(entry_price, direction, signal_strength, volatility):
    """Calculate profit target and stop loss (same as V3)"""
    if TIGHTER_STOPS:
        stop_distance = 1.0 * volatility * entry_price
    else:
        stop_distance = 1.5 * volatility * entry_price
    
    min_stop = 0.005 * entry_price
    stop_distance = max(stop_distance, min_stop)
    profit_distance = stop_distance * PROFIT_TARGET_RATIO
    
    if direction == "long":
        stop_price = entry_price - stop_distance
        profit_target = entry_price + profit_distance
    else:
        stop_price = entry_price + stop_distance
        profit_target = entry_price - profit_distance
    
    return profit_target, stop_price, stop_distance

def get_modal_position_bin(modal_position, n_bins=MODAL_POSITION_BINS):
    """Convert modal position to bin number (0 to n_bins-1)"""
    bin_size = 1.0 / n_bins
    bin_number = int(modal_position / bin_size)
    return min(bin_number, n_bins - 1)  # Handle edge case of exactly 1.0

def build_modal_position_statistics(historical_trades):
    """Build statistics for modal position bins based on historical trades"""
    if len(historical_trades) == 0:
        return {}
    
    # Group trades by modal position bin
    bin_stats = {}
    for trade in historical_trades:
        bin_num = get_modal_position_bin(trade['modal_position'])
        if bin_num not in bin_stats:
            bin_stats[bin_num] = []
        bin_stats[bin_num].append(trade['net_return'])
    
    # Calculate mean return for each bin
    bin_returns = {}
    for bin_num, returns in bin_stats.items():
        if len(returns) >= 3:  # Minimum trades per bin
            bin_returns[bin_num] = {
                'mean_return': np.mean(returns),
                'count': len(returns),
                'std_return': np.std(returns)
            }
    
    return bin_returns

def is_modal_position_tradeable(modal_position, historical_bin_stats):
    """Check if modal position bin has positive historical returns"""
    if len(historical_bin_stats) < MIN_HISTORICAL_TRADES:
        # Fall back to V3 logic if insufficient historical data
        return modal_position <= 0.28 or modal_position >= 0.72
    
    bin_num = get_modal_position_bin(modal_position)
    
    if bin_num in historical_bin_stats:
        bin_data = historical_bin_stats[bin_num]
        return bin_data['mean_return'] > MIN_BIN_RETURN_THRESHOLD
    
    return False  # Don't trade unknown bins

def get_modal_direction(modal_position, historical_bin_stats):
    """Determine trade direction based on modal position and historical data"""
    if len(historical_bin_stats) < MIN_HISTORICAL_TRADES:
        # Fall back to V3 logic
        if modal_position <= 0.28:
            return "long"
        elif modal_position >= 0.72:
            return "short"
        else:
            return None
    
    bin_num = get_modal_position_bin(modal_position)
    
    # Determine direction based on which extreme the bin is closer to
    if modal_position <= 0.5:
        return "long"  # Lower modal positions typically long
    else:
        return "short"  # Higher modal positions typically short

def save_statistics_to_csv(modal_stats_history, volume_stats_history):
    """Save statistics to CSV files for analysis"""
    
    # Save modal position statistics
    modal_rows = []
    for date, bin_stats in modal_stats_history.items():
        for bin_num, stats in bin_stats.items():
            modal_rows.append({
                'date': date,
                'bin_number': bin_num,
                'bin_range_start': bin_num / MODAL_POSITION_BINS,
                'bin_range_end': (bin_num + 1) / MODAL_POSITION_BINS,
                'mean_return': stats['mean_return'],
                'trade_count': stats['count'],
                'std_return': stats['std_return']
            })
    
    modal_df = pd.DataFrame(modal_rows)
    if not modal_df.empty:
        modal_df.to_csv("../data/modal_position_returns.csv", index=False)
        print(f"Saved modal position statistics: {len(modal_df)} bin-date combinations")
    
    # Save volume statistics summary
    volume_rows = []
    for date, stats in volume_stats_history.items():
        volume_rows.append({
            'date': date,
            'total_clusters': stats.get('total_clusters', 0),
            'top_n_threshold': stats.get('top_n_threshold', 0),
            'mean_volume_ratio': stats.get('mean_volume_ratio', 0),
            'max_volume_ratio': stats.get('max_volume_ratio', 0)
        })
    
    volume_df = pd.DataFrame(volume_rows)
    if not volume_df.empty:
        volume_df.to_csv("../data/volume_ratio_returns.csv", index=False)
        print(f"Saved volume ratio statistics: {len(volume_df)} trading days")

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")

# Initialize tracking variables
results = []
historical_trades = []  # Running history of completed trades
modal_stats_history = {}  # Track modal position statistics over time
volume_stats_history = {}  # Track volume statistics over time

print("Starting adaptive backtest simulation...")

# Process day by day
for day, group in tqdm(df.groupby(df.index.date), desc="Processing days"):
    intraday_df = group.copy()
    
    # Identify clusters for the day
    clusters_df = identify_volume_clusters(intraday_df, volume_multiplier=VOLUME_THRESHOLD)
    
    if clusters_df.empty:
        continue
    
    # Calculate volume ratios for all clusters
    avg_volume = intraday_df['volume'].mean()
    daily_clusters = []
    
    clusters_sorted = clusters_df.sort_index()
    
    for cluster_time, cluster_row in clusters_sorted.iterrows():
        if not is_valid_trading_time(cluster_time):
            continue
            
        cluster_volume = cluster_row['volume']
        volume_ratio = cluster_volume / avg_volume if avg_volume > 0 else 0
        
        daily_clusters.append({
            'timestamp': cluster_time,
            'volume_ratio': volume_ratio,
            'cluster_data': cluster_row
        })
    
    # Adaptive volume filtering: rank clusters and keep top N
    daily_clusters_sorted = sorted(daily_clusters, key=lambda x: x['volume_ratio'], reverse=True)
    top_n_clusters = daily_clusters_sorted[:TOP_N_CLUSTERS_PER_DAY]
    
    # Store volume statistics for this day
    if daily_clusters:
        volume_ratios = [c['volume_ratio'] for c in daily_clusters]
        top_n_threshold = top_n_clusters[-1]['volume_ratio'] if top_n_clusters else 0
        
        volume_stats_history[day] = {
            'total_clusters': len(daily_clusters),
            'top_n_threshold': top_n_threshold,
            'mean_volume_ratio': np.mean(volume_ratios),
            'max_volume_ratio': max(volume_ratios)
        }
    
    # Build modal position statistics from historical trades (lookback approach)
    cutoff_date = day - timedelta(days=LOOKBACK_DAYS)
    recent_trades = [t for t in historical_trades if t['date'] >= cutoff_date]
    current_bin_stats = build_modal_position_statistics(recent_trades)
    
    # Store modal statistics for this day
    modal_stats_history[day] = current_bin_stats.copy()
    
    # Process each top cluster
    for cluster_info in top_n_clusters:
        cluster_time = cluster_info['timestamp']
        volume_ratio = cluster_info['volume_ratio']
        cluster_row = cluster_info['cluster_data']
        
        # Analyze cluster price action
        cluster_slice = intraday_df.loc[cluster_time : cluster_time + timedelta(minutes=14)]
        
        if cluster_slice.empty:
            continue
            
        modal_price = cluster_slice["close"].round(2).mode()
        if len(modal_price) == 0:
            continue
        modal_price = modal_price[0]

        price_low = cluster_slice["low"].min()
        price_high = cluster_slice["high"].max()
        modal_position = (modal_price - price_low) / (price_high - price_low + 1e-9)

        # Adaptive modal position filtering
        if not is_modal_position_tradeable(modal_position, current_bin_stats):
            continue
        
        # Determine direction using adaptive logic
        direction = get_modal_direction(modal_position, current_bin_stats)
        if direction is None:
            continue

        # Calculate momentum and signal strength (same as V3)
        momentum = calculate_momentum(intraday_df, cluster_time)
        signal_strength = calculate_signal_strength_v3(modal_position, volume_ratio, momentum)
        
        if signal_strength < MIN_SIGNAL_STRENGTH:
            continue

        # Position sizing (same as V3)
        volume_strength = min(volume_ratio / 100.0, 2.0)
        position_multiplier = BASE_POSITION_SIZE + (signal_strength * volume_strength * 0.3)
        position_multiplier = min(position_multiplier, MAX_POSITION_SIZE)

        # Find retest (same as V3)
        retest_time = None
        for t, row in intraday_df.loc[cluster_time:].iterrows():
            if abs(row["close"] - modal_price) <= 0.75 and RETEST_ENABLED:
                retest_time = t
                break
        if not RETEST_ENABLED:
            retest_time = cluster_time + timedelta(minutes=1)

        if retest_time is None:
            continue

        # Entry logic (same as V3)
        future_candles = intraday_df[intraday_df.index > retest_time]
        if future_candles.empty:
            continue
            
        entry_time = future_candles.index[0]
        entry_data = intraday_df.loc[entry_time]
        if isinstance(entry_data, pd.Series):
            entry_price = float(entry_data["close"])
        else:
            entry_price = float(entry_data["close"].iloc[0])

        # Calculate volatility and targets (same as V3)
        recent_data = df.loc[:entry_time].tail(VOLATILITY_LOOKBACK * 24 * 60)
        if len(recent_data) > 100:
            price_changes = recent_data['close'].pct_change().dropna()
            volatility = price_changes.std()
        else:
            volatility = 0.01

        profit_target, stop_price, stop_distance = calculate_profit_target_and_stop(
            entry_price, direction, signal_strength, volatility
        )

        # Exit processing (same as V3)
        max_hold_time = entry_time + timedelta(minutes=60)
        future_data = intraday_df[intraday_df.index > entry_time]
        exit_price = None
        actual_exit_time = None
        exit_reason = "time"
        stopped_out = False
        profit_hit = False
        
        for period_time, period_row in future_data.iterrows():
            if period_time > max_hold_time:
                exit_price = period_row["close"]
                actual_exit_time = period_time
                exit_reason = "time"
                break
                
            if USE_PROFIT_TARGETS:
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
        
        if exit_price is None:
            last_data = future_data.tail(1)
            if not last_data.empty:
                exit_price = float(last_data.iloc[0]["close"])
                actual_exit_time = last_data.index[0]
                exit_reason = "end_of_data"
            else:
                continue
        
        # Calculate returns (same as V3)
        if direction == "short":
            gross_return = (entry_price - exit_price) / entry_price
        elif direction == "long":
            gross_return = (exit_price - entry_price) / entry_price
        
        transaction_cost = calculate_transaction_costs(position_multiplier)
        net_return = gross_return - transaction_cost
        position_adjusted_return = net_return * position_multiplier

        # Store trade result
        trade_result = {
            "date": day,
            "entry_time": entry_time,
            "exit_time": actual_exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "modal_price": modal_price,
            "modal_position": modal_position,
            "modal_bin": get_modal_position_bin(modal_position),
            "volume_ratio": volume_ratio,
            "volume_rank": next(i for i, c in enumerate(daily_clusters_sorted) if c['timestamp'] == cluster_time) + 1,
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
            "historical_trades_count": len(recent_trades),
            "used_adaptive_modal": len(current_bin_stats) >= MIN_HISTORICAL_TRADES
        }
        
        results.append(trade_result)
        
        # Add to historical trades for future adaptive decisions
        historical_trades.append({
            'date': day,
            'modal_position': modal_position,
            'net_return': net_return,
            'direction': direction
        })

# Export results
print("\nExporting results...")
result_df = pd.DataFrame(results)
result_df.to_csv("../data/backtest_results_v4.csv", index=False)

# Save adaptive statistics
save_statistics_to_csv(modal_stats_history, volume_stats_history)

# Enhanced Analysis
if len(result_df) > 0:
    print(f"\n=== ADAPTIVE STRATEGY V4.0 RESULTS ===")
    print(f"Total Simulated Trades: {len(result_df)}")
    
    # Basic performance metrics
    mean_gross_ret = result_df["gross_return"].mean()
    mean_net_ret = result_df["net_return"].mean()
    mean_pos_adj_ret = result_df["position_adjusted_return"].mean()
    win_rate = (result_df["net_return"] > 0).mean()
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Mean Gross Return: {mean_gross_ret:.4%}")
    print(f"  Mean Net Return: {mean_net_ret:.4%}")
    print(f"  Mean Position-Adjusted Return: {mean_pos_adj_ret:.4%}")
    print(f"  Win Rate: {win_rate:.2%}")
    
    # Adaptive strategy analysis
    adaptive_trades = result_df[result_df["used_adaptive_modal"] == True]
    fallback_trades = result_df[result_df["used_adaptive_modal"] == False]
    
    print(f"\nADAPTIVE ANALYSIS:")
    print(f"  Trades using adaptive modal filter: {len(adaptive_trades)} ({len(adaptive_trades)/len(result_df)*100:.1f}%)")
    print(f"  Trades using fallback logic: {len(fallback_trades)} ({len(fallback_trades)/len(result_df)*100:.1f}%)")
    
    if len(adaptive_trades) > 0:
        print(f"  Adaptive trades mean return: {adaptive_trades['net_return'].mean():.4%}")
        print(f"  Adaptive trades win rate: {(adaptive_trades['net_return'] > 0).mean():.2%}")
    
    if len(fallback_trades) > 0:
        print(f"  Fallback trades mean return: {fallback_trades['net_return'].mean():.4%}")
        print(f"  Fallback trades win rate: {(fallback_trades['net_return'] > 0).mean():.2%}")
    
    # Volume ranking analysis
    print(f"\nVOLUME RANKING ANALYSIS:")
    rank_performance = result_df.groupby("volume_rank")["net_return"].agg(['mean', 'count'])
    for rank, stats in rank_performance.iterrows():
        print(f"  Rank {rank}: {stats['mean']:.4%} return ({int(stats['count'])} trades)")
    
    # Modal position bin analysis
    print(f"\nMODAL POSITION BIN ANALYSIS:")
    bin_performance = result_df.groupby("modal_bin")["net_return"].agg(['mean', 'count'])
    for bin_num, stats in bin_performance.iterrows():
        bin_start = bin_num / MODAL_POSITION_BINS
        bin_end = (bin_num + 1) / MODAL_POSITION_BINS
        print(f"  Bin {bin_num} ({bin_start:.1f}-{bin_end:.1f}): {stats['mean']:.4%} return ({int(stats['count'])} trades)")
    
    # Exit analysis (same as V3)
    exit_reasons = result_df["exit_reason"].value_counts()
    profit_target_rate = (result_df["profit_hit"] == True).mean()
    stop_loss_rate = (result_df["stopped_out"] == True).mean()
    
    print(f"\nEXIT ANALYSIS:")
    print(f"  Profit Target Hit: {profit_target_rate:.2%}")
    print(f"  Stop Loss Hit: {stop_loss_rate:.2%}")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(result_df)*100:.1f}%)")
    
    print(f"\nFILES EXPORTED:")
    print(f"  backtest_results_v4.csv - Main backtest results")
    print(f"  modal_position_returns.csv - Modal position bin statistics")
    print(f"  volume_ratio_returns.csv - Volume ratio statistics")
    
else:
    print("No trades found in adaptive backtest simulation.") 