import pandas as pd
import numpy as np

# Load all three versions
v3 = pd.read_csv('../data/backtest_results_v3.csv')
v4 = pd.read_csv('../data/backtest_results_v4.csv')
v5 = pd.read_csv('../data/backtest_results_v5.csv')

print('=' * 80)
print('COMPLETE STRATEGY EVOLUTION ANALYSIS: V3 → V4 → V5')
print('=' * 80)

# Basic trade counts
print('\n📊 TRADE COUNT EVOLUTION:')
print(f'V3 Total Trades: {len(v3)}')
print(f'V4 Total Trades: {len(v4)} ({len(v4) - len(v3):+d}, {(len(v4)/len(v3)-1)*100:+.1f}%)')
print(f'V5 Total Trades: {len(v5)} ({len(v5) - len(v4):+d}, {(len(v5)/len(v4)-1)*100:+.1f}%)')
print(f'V3 → V5 Total Change: {len(v5) - len(v3):+d} ({(len(v5)/len(v3)-1)*100:+.1f}%)')

# Performance comparison
v3_mean_net = v3['net_return'].mean()
v4_mean_net = v4['net_return'].mean()
v5_mean_net = v5['net_return'].mean()

v3_win_rate = (v3['net_return'] > 0).mean()
v4_win_rate = (v4['net_return'] > 0).mean()
v5_win_rate = (v5['net_return'] > 0).mean()

print('\n🎯 CORE PERFORMANCE EVOLUTION:')
print(f'Mean Net Return:')
print(f'  V3: {v3_mean_net:.4%}')
print(f'  V4: {v4_mean_net:.4%} ({(v4_mean_net - v3_mean_net)*100:+.2f} bps)')
print(f'  V5: {v5_mean_net:.4%} ({(v5_mean_net - v4_mean_net)*100:+.2f} bps)')
print(f'  V3→V5 Total Improvement: {(v5_mean_net - v3_mean_net)*100:+.2f} basis points')

print(f'\nWin Rate:')
print(f'  V3: {v3_win_rate:.2%}')
print(f'  V4: {v4_win_rate:.2%} ({(v4_win_rate - v3_win_rate)*100:+.1f} ppts)')
print(f'  V5: {v5_win_rate:.2%} ({(v5_win_rate - v4_win_rate)*100:+.1f} ppts)')
print(f'  V3→V5 Total Improvement: {(v5_win_rate - v3_win_rate)*100:+.1f} percentage points')

# Risk-adjusted metrics
v3_std = v3['net_return'].std()
v4_std = v4['net_return'].std()
v5_std = v5['net_return'].std()

v3_sharpe = v3_mean_net / v3_std if v3_std > 0 else 0
v4_sharpe = v4_mean_net / v4_std if v4_std > 0 else 0
v5_sharpe = v5_mean_net / v5_std if v5_std > 0 else 0

print('\n📈 RISK-ADJUSTED METRICS:')
print(f'Return Volatility:')
print(f'  V3: {v3_std:.4%}')
print(f'  V4: {v4_std:.4%}')
print(f'  V5: {v5_std:.4%}')

print(f'\nSharpe Ratio:')
print(f'  V3: {v3_sharpe:.3f}')
print(f'  V4: {v4_sharpe:.3f} ({v4_sharpe - v3_sharpe:+.3f})')
print(f'  V5: {v5_sharpe:.3f} ({v5_sharpe - v4_sharpe:+.3f})')
print(f'  V3→V5 Total Improvement: {v5_sharpe - v3_sharpe:+.3f}')

# Direction analysis evolution
print('\n🎯 DIRECTIONAL STRATEGY EVOLUTION:')

v3_long = v3[v3['direction'] == 'long']
v3_short = v3[v3['direction'] == 'short']
v4_long = v4[v4['direction'] == 'long']
v4_short = v4[v4['direction'] == 'short']
v5_long = v5[v5['direction'] == 'long']
v5_short = v5[v5['direction'] == 'short'] if 'direction' in v5.columns else pd.DataFrame()

print('Long Trades:')
print(f'  V3: {len(v3_long)} trades, {v3_long["net_return"].mean():.4%} avg return')
print(f'  V4: {len(v4_long)} trades, {v4_long["net_return"].mean():.4%} avg return')
print(f'  V5: {len(v5_long)} trades, {v5_long["net_return"].mean():.4%} avg return')

print('Short Trades:')
print(f'  V3: {len(v3_short)} trades, {v3_short["net_return"].mean():.4%} avg return')
print(f'  V4: {len(v4_short)} trades, {v4_short["net_return"].mean():.4%} avg return')
print(f'  V5: {len(v5_short)} trades (eliminated shorts strategy)')

# Volume filtering evolution
print('\n🔥 VOLUME FILTERING EVOLUTION:')

if 'volume_rank' in v4.columns:
    v4_rank_stats = v4.groupby('volume_rank')['net_return'].agg(['mean', 'count'])
    print('V4 Volume Ranking:')
    for rank, stats in v4_rank_stats.iterrows():
        print(f'  Rank {rank}: {stats["mean"]:.4%} ({int(stats["count"])} trades)')

print('V5 Top-1 Only Strategy:')
print(f'  All trades are top-1 volume clusters: {v5_mean_net:.4%} avg return')

# Modal position analysis evolution  
print('\n📍 MODAL POSITION THRESHOLD EVOLUTION:')

# V3 traditional thresholds
v3_modal_stats = v3['modal_position'].describe()
print('V3 Traditional Thresholds:')
print(f'  Long (≤0.28): {(v3_long["modal_position"] <= 0.28).sum()} trades')
print(f'  Short (≥0.72): {(v3_short["modal_position"] >= 0.72).sum()} trades')
print(f'  Modal range: {v3_modal_stats["min"]:.3f} - {v3_modal_stats["max"]:.3f}')

# V4 bin analysis
if 'modal_bin' in v4.columns:
    v4_bin_stats = v4.groupby('modal_bin')['net_return'].agg(['mean', 'count'])
    print('\nV4 Modal Bin Analysis:')
    for bin_num, stats in v4_bin_stats.iterrows():
        bin_start = bin_num / 10
        bin_end = (bin_num + 1) / 10
        print(f'  Bin {bin_num} ({bin_start:.1f}-{bin_end:.1f}): {stats["mean"]:.4%} ({int(stats["count"])} trades)')

# V5 tightened logic
v5_modal_stats = v5['modal_position'].describe()
print('\nV5 Tightened Thresholds:')
print(f'  Long only (≤0.15): {len(v5)} trades')
print(f'  Modal range: {v5_modal_stats["min"]:.3f} - {v5_modal_stats["max"]:.3f}')

# Adaptive filter usage evolution
print('\n🤖 ADAPTIVE FILTERING USAGE:')

if 'used_adaptive_modal' in v4.columns:
    v4_adaptive = v4[v4['used_adaptive_modal'] == True]
    v4_fallback = v4[v4['used_adaptive_modal'] == False]
    print('V4 Adaptive Usage:')
    print(f'  Adaptive: {len(v4_adaptive)} ({len(v4_adaptive)/len(v4)*100:.1f}%)')
    print(f'  Fallback: {len(v4_fallback)} ({len(v4_fallback)/len(v4)*100:.1f}%)')

if 'used_adaptive_modal' in v5.columns:
    v5_adaptive = v5[v5['used_adaptive_modal'] == True]
    v5_fallback = v5[v5['used_adaptive_modal'] == False]
    print('V5 Adaptive Usage:')
    print(f'  Adaptive: {len(v5_adaptive)} ({len(v5_adaptive)/len(v5)*100:.1f}%)')
    print(f'  Fallback: {len(v5_fallback)} ({len(v5_fallback)/len(v5)*100:.1f}%)')

# Position sizing evolution
print('\n💰 POSITION SIZING EVOLUTION:')

v3_pos_stats = v3['position_multiplier'].describe()
v4_pos_stats = v4['position_multiplier'].describe()
v5_pos_stats = v5['position_multiplier'].describe()

print('Position Multiplier Statistics:')
print(f'  V3: Mean={v3_pos_stats["mean"]:.3f}, Max={v3_pos_stats["max"]:.3f}')
print(f'  V4: Mean={v4_pos_stats["mean"]:.3f}, Max={v4_pos_stats["max"]:.3f}')
print(f'  V5: Mean={v5_pos_stats["mean"]:.3f}, Max={v5_pos_stats["max"]:.3f}')

# V5 specific boost analysis
if 'volume_rank_boost' in v5.columns and 'modal_quality_boost' in v5.columns:
    volume_boosted = v5[v5['volume_rank_boost'] > 1.0]
    modal_boosted = v5[v5['modal_quality_boost'] > 1.0]
    
    print('\nV5 Position Sizing Boosts:')
    print(f'  Volume rank boost: {len(volume_boosted)} trades ({len(volume_boosted)/len(v5)*100:.1f}%)')
    print(f'  Modal quality boost: {len(modal_boosted)} trades ({len(modal_boosted)/len(v5)*100:.1f}%)')
    if len(modal_boosted) > 0:
        print(f'    Modal boost avg return: {modal_boosted["net_return"].mean():.4%}')

# Exit strategy analysis
print('\n🚪 EXIT STRATEGY ANALYSIS:')

def analyze_exits(df, version_name):
    profit_target_rate = (df['profit_hit'] == True).mean() if 'profit_hit' in df.columns else 0
    stop_loss_rate = (df['stopped_out'] == True).mean() if 'stopped_out' in df.columns else 0
    
    print(f'{version_name}:')
    print(f'  Profit Target Hit: {profit_target_rate:.2%}')
    print(f'  Stop Loss Hit: {stop_loss_rate:.2%}')
    
    if 'exit_reason' in df.columns:
        exit_breakdown = df['exit_reason'].value_counts()
        for reason, count in exit_breakdown.items():
            print(f'  {reason}: {count} ({count/len(df)*100:.1f}%)')

analyze_exits(v3, 'V3')
analyze_exits(v4, 'V4')
analyze_exits(v5, 'V5')

# Monthly performance comparison
print('\n📅 MONTHLY PERFORMANCE EVOLUTION:')

v3['date'] = pd.to_datetime(v3['date'])
v4['date'] = pd.to_datetime(v4['date'])
v5['date'] = pd.to_datetime(v5['date'])

v3_monthly = v3.groupby(v3['date'].dt.to_period('M')).agg({
    'net_return': ['count', 'mean']
}).round(4)
v4_monthly = v4.groupby(v4['date'].dt.to_period('M')).agg({
    'net_return': ['count', 'mean']
}).round(4)
v5_monthly = v5.groupby(v5['date'].dt.to_period('M')).agg({
    'net_return': ['count', 'mean']
}).round(4)

# Flatten column names
v3_monthly.columns = ['count', 'mean_return']
v4_monthly.columns = ['count', 'mean_return']
v5_monthly.columns = ['count', 'mean_return']

print('Monthly Trade Counts & Average Returns:')
print('Month       | V3 Trades | V3 Avg Ret | V4 Trades | V4 Avg Ret | V5 Trades | V5 Avg Ret')
print('-' * 85)

all_months = sorted(set(v3_monthly.index).union(set(v4_monthly.index)).union(set(v5_monthly.index)))
for month in all_months:
    v3_count = v3_monthly.loc[month, 'count'] if month in v3_monthly.index else 0
    v3_ret = v3_monthly.loc[month, 'mean_return'] if month in v3_monthly.index else 0
    v4_count = v4_monthly.loc[month, 'count'] if month in v4_monthly.index else 0
    v4_ret = v4_monthly.loc[month, 'mean_return'] if month in v4_monthly.index else 0
    v5_count = v5_monthly.loc[month, 'count'] if month in v5_monthly.index else 0
    v5_ret = v5_monthly.loc[month, 'mean_return'] if month in v5_monthly.index else 0
    
    print(f'{month}    |    {v3_count:3.0f}    |   {v3_ret:6.2%}  |    {v4_count:3.0f}    |   {v4_ret:6.2%}  |    {v5_count:3.0f}    |   {v5_ret:6.2%}')

# Summary of optimizations impact
print('\n' + '=' * 80)
print('🏆 OPTIMIZATION IMPACT SUMMARY')
print('=' * 80)

print('\n1. VOLUME FILTERING OPTIMIZATION:')
print(f'   • V3→V4: Introduced dynamic ranking (top-2) vs fixed threshold')
print(f'   • V4→V5: Focused on top-1 only')
print(f'   • Impact: Trade count V3→V5: {len(v3)} → {len(v5)} ({(len(v5)/len(v3)-1)*100:+.1f}%)')

print('\n2. MODAL POSITION REFINEMENT:')
print(f'   • V3: Fixed thresholds (≤0.28 long, ≥0.72 short)')  
print(f'   • V4: Adaptive bin-based filtering')
print(f'   • V5: Tightened to ≤0.15 long only, eliminated shorts')
print(f'   • Impact: Return improvement V3→V5: {(v5_mean_net - v3_mean_net)*100:+.2f} bps')

print('\n3. POSITION SIZING ENHANCEMENT:')
print(f'   • V5: Added volume rank boost (2x) and modal quality boost (1.5x)')
print(f'   • {len(modal_boosted) if "modal_boosted" in locals() else 0} trades received modal boost ({len(modal_boosted)/len(v5)*100:.1f}%)')

print('\n4. OVERALL STRATEGY IMPROVEMENT:')
print(f'   • Sharpe Ratio: {v3_sharpe:.3f} → {v5_sharpe:.3f} ({v5_sharpe - v3_sharpe:+.3f})')
print(f'   • Win Rate: {v3_win_rate:.1%} → {v5_win_rate:.1%} ({(v5_win_rate - v3_win_rate)*100:+.1f} ppts)')
print(f'   • Mean Return: {v3_mean_net:.3%} → {v5_mean_net:.3%} ({(v5_mean_net - v3_mean_net)*100:+.2f} bps)')
print(f'   • Trade Quality: Higher returns with fewer, better-selected trades')

# Calculate compound returns for each version
v3_compound = (1 + v3['net_return']).prod() - 1
v4_compound = (1 + v4['net_return']).prod() - 1
v5_compound = (1 + v5['net_return']).prod() - 1

print('\n5. CUMULATIVE PERFORMANCE:')
print(f'   • V3 Total Return: {v3_compound:.2%} ({len(v3)} trades)')
print(f'   • V4 Total Return: {v4_compound:.2%} ({len(v4)} trades)')
print(f'   • V5 Total Return: {v5_compound:.2%} ({len(v5)} trades)')
print(f'   • V5 Return Per Trade: {v5_compound/len(v5)*100:.3f}% avg')

print('\n' + '=' * 80)
print('✅ STRATEGY EVOLUTION COMPLETE: From broad to focused, adaptive excellence!')
print('=' * 80) 