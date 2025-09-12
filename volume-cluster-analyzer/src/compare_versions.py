import pandas as pd
import numpy as np

# Load both results
v3 = pd.read_csv('../data/backtest_results_v3.csv')
v4 = pd.read_csv('../data/backtest_results_v4.csv')

print('=== BACKTEST VERSION COMPARISON ===')
print(f'V3 Total Trades: {len(v3)}')
print(f'V4 Total Trades: {len(v4)}')
print(f'Trade Count Change: {len(v4) - len(v3)} ({(len(v4)/len(v3)-1)*100:+.1f}%)')
print()

print('PERFORMANCE COMPARISON:')
v3_mean_net = v3['net_return'].mean()
v4_mean_net = v4['net_return'].mean()
v3_win_rate = (v3['net_return'] > 0).mean()
v4_win_rate = (v4['net_return'] > 0).mean()

print(f'V3 Mean Net Return: {v3_mean_net:.4%}')
print(f'V4 Mean Net Return: {v4_mean_net:.4%}')
print(f'Return Improvement: {(v4_mean_net - v3_mean_net)*100:+.2f} basis points')
print()
print(f'V3 Win Rate: {v3_win_rate:.2%}')
print(f'V4 Win Rate: {v4_win_rate:.2%}')
print(f'Win Rate Change: {(v4_win_rate - v3_win_rate)*100:+.1f} percentage points')
print()

# Compare volume ranking vs old volume threshold
print('VOLUME FILTERING COMPARISON:')
if 'volume_rank' in v4.columns:
    v4_rank_perf = v4.groupby('volume_rank')['net_return'].agg(['mean', 'count'])
    print('V4 Volume Rank Performance:')
    for rank, stats in v4_rank_perf.iterrows():
        print(f'  Rank {rank}: {stats["mean"]:.4%} ({stats["count"]} trades)')
print()

# Compare V3 volume ratios vs V4 ranks
print('V3 Volume Ratio vs V4 Ranking:')
v3_vol_stats = v3['volume_ratio'].describe()
print(f'V3 Volume Ratio Stats: Mean={v3_vol_stats["mean"]:.1f}, Min={v3_vol_stats["min"]:.1f}, Max={v3_vol_stats["max"]:.1f}')

# Modal position analysis
print()
print('MODAL POSITION ANALYSIS:')
if 'modal_bin' in v4.columns:
    v4_modal_perf = v4.groupby('modal_bin')['net_return'].agg(['mean', 'count'])
    print('V4 Modal Bin Performance:')
    for bin_num, stats in v4_modal_perf.iterrows():
        bin_start = bin_num / 10
        bin_end = (bin_num + 1) / 10
        print(f'  Bin {bin_num} ({bin_start:.1f}-{bin_end:.1f}): {stats["mean"]:.4%} ({stats["count"]} trades)')

print()
print('V3 Modal Position Distribution (traditional thresholds):')
v3_long = v3[v3['direction'] == 'long']
v3_short = v3[v3['direction'] == 'short']
v3_modal_long = v3_long['modal_position']
v3_modal_short = v3_short['modal_position']

print(f'V3 Long trades: {len(v3_long)} (modal pos <= 0.28: {(v3_modal_long <= 0.28).sum()})')
print(f'V3 Short trades: {len(v3_short)} (modal pos >= 0.72: {(v3_modal_short >= 0.72).sum()})')
print(f'V3 Long mean return: {v3_long["net_return"].mean():.4%}')
print(f'V3 Short mean return: {v3_short["net_return"].mean():.4%}')

print()
print('V4 Direction Distribution:')
v4_long = v4[v4['direction'] == 'long']
v4_short = v4[v4['direction'] == 'short']
print(f'V4 Long trades: {len(v4_long)}')
print(f'V4 Short trades: {len(v4_short)}')
print(f'V4 Long mean return: {v4_long["net_return"].mean():.4%}')
print(f'V4 Short mean return: {v4_short["net_return"].mean():.4%}')

# Adaptive filtering analysis
print()
print('ADAPTIVE FILTERING ANALYSIS:')
if 'used_adaptive_modal' in v4.columns:
    adaptive_trades = v4[v4['used_adaptive_modal'] == True]
    fallback_trades = v4[v4['used_adaptive_modal'] == False]
    
    print(f'Adaptive trades: {len(adaptive_trades)} ({len(adaptive_trades)/len(v4)*100:.1f}%)')
    print(f'Fallback trades: {len(fallback_trades)} ({len(fallback_trades)/len(v4)*100:.1f}%)')
    
    if len(adaptive_trades) > 0:
        print(f'Adaptive mean return: {adaptive_trades["net_return"].mean():.4%}')
        print(f'Adaptive win rate: {(adaptive_trades["net_return"] > 0).mean():.2%}')
    
    if len(fallback_trades) > 0:
        print(f'Fallback mean return: {fallback_trades["net_return"].mean():.4%}')
        print(f'Fallback win rate: {(fallback_trades["net_return"] > 0).mean():.2%}')

# Trade frequency over time
print()
print('TRADE FREQUENCY ANALYSIS:')
v3['date'] = pd.to_datetime(v3['date'])
v4['date'] = pd.to_datetime(v4['date'])

v3_monthly = v3.groupby(v3['date'].dt.to_period('M')).size()
v4_monthly = v4.groupby(v4['date'].dt.to_period('M')).size()

print('Monthly trade counts:')
for month in sorted(set(v3_monthly.index).union(set(v4_monthly.index))):
    v3_count = v3_monthly.get(month, 0)
    v4_count = v4_monthly.get(month, 0)
    print(f'{month}: V3={v3_count}, V4={v4_count} (diff: {v4_count-v3_count:+d})')

# Risk metrics
print()
print('RISK METRICS:')
v3_std = v3['net_return'].std()
v4_std = v4['net_return'].std()
v3_sharpe = v3_mean_net / v3_std if v3_std > 0 else 0
v4_sharpe = v4_mean_net / v4_std if v4_std > 0 else 0

print(f'V3 Return Std Dev: {v3_std:.4%}')
print(f'V4 Return Std Dev: {v4_std:.4%}')
print(f'V3 Sharpe Ratio: {v3_sharpe:.3f}')
print(f'V4 Sharpe Ratio: {v4_sharpe:.3f}')

# Max drawdown analysis
v3_cum_returns = (1 + v3['net_return']).cumprod()
v4_cum_returns = (1 + v4['net_return']).cumprod()

v3_running_max = v3_cum_returns.expanding().max()
v4_running_max = v4_cum_returns.expanding().max()

v3_drawdown = (v3_cum_returns - v3_running_max) / v3_running_max
v4_drawdown = (v4_cum_returns - v4_running_max) / v4_running_max

print(f'V3 Max Drawdown: {v3_drawdown.min():.4%}')
print(f'V4 Max Drawdown: {v4_drawdown.min():.4%}') 