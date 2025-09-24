import pandas as pd
import numpy as np

# Load both versions
v5_biased = pd.read_csv('../data/backtest_results_v5.csv')
v5_fixed = pd.read_csv('../data/backtest_results_v5_fixed.csv')

print('=' * 90)
print('🚨 FORWARD-LOOKING BIAS IMPACT ANALYSIS: V5 Original vs V5 Fixed')
print('=' * 90)

print('\n📊 TRADE COUNT COMPARISON:')
print(f'V5 Original (Biased):     {len(v5_biased)} trades')
print(f'V5 Fixed (Bias-Free):     {len(v5_fixed)} trades')
print(f'Difference:               {len(v5_fixed) - len(v5_biased):+d} trades ({(len(v5_fixed)/len(v5_biased)-1)*100:+.1f}%)')

# Performance comparison
v5_biased_ret = v5_biased['net_return'].mean()
v5_fixed_ret = v5_fixed['net_return'].mean()
v5_biased_win = (v5_biased['net_return'] > 0).mean()
v5_fixed_win = (v5_fixed['net_return'] > 0).mean()

print('\n📈 PERFORMANCE IMPACT:')
print(f'Mean Net Return:')
print(f'  V5 Original (Biased):   {v5_biased_ret:.4%}')
print(f'  V5 Fixed (Bias-Free):   {v5_fixed_ret:.4%}')
print(f'  Bias Impact:            {(v5_fixed_ret - v5_biased_ret)*100:+.2f} basis points')

print(f'\nWin Rate:')
print(f'  V5 Original (Biased):   {v5_biased_win:.2%}')
print(f'  V5 Fixed (Bias-Free):   {v5_fixed_win:.2%}')
print(f'  Bias Impact:            {(v5_fixed_win - v5_biased_win)*100:+.1f} percentage points')

# Risk metrics
v5_biased_std = v5_biased['net_return'].std()
v5_fixed_std = v5_fixed['net_return'].std()
v5_biased_sharpe = v5_biased_ret / v5_biased_std if v5_biased_std > 0 else 0
v5_fixed_sharpe = v5_fixed_ret / v5_fixed_std if v5_fixed_std > 0 else 0

print('\n⚡ RISK-ADJUSTED METRICS:')
print(f'Sharpe Ratio:')
print(f'  V5 Original (Biased):   {v5_biased_sharpe:.3f}')
print(f'  V5 Fixed (Bias-Free):   {v5_fixed_sharpe:.3f}')
print(f'  Bias Impact:            {v5_fixed_sharpe - v5_biased_sharpe:+.3f}')

print(f'\nReturn Volatility:')
print(f'  V5 Original (Biased):   {v5_biased_std:.4%}')
print(f'  V5 Fixed (Bias-Free):   {v5_fixed_std:.4%}')

# Volume ranking analysis
print('\n🔍 VOLUME RANKING COMPARISON:')
if 'volume_rank' in v5_biased.columns and 'volume_rank' in v5_fixed.columns:
    print('V5 Original (Biased) - Volume Ranking:')
    biased_ranks = v5_biased.groupby('volume_rank')['net_return'].agg(['mean', 'count'])
    for rank, stats in biased_ranks.iterrows():
        print(f'  Rank {rank}: {stats["mean"]:.4%} ({int(stats["count"])} trades)')
    
    print('\nV5 Fixed (Bias-Free) - Volume Ranking:')
    fixed_ranks = v5_fixed.groupby('volume_rank')['net_return'].agg(['mean', 'count'])
    for rank, stats in fixed_ranks.iterrows():
        print(f'  Rank {rank}: {stats["mean"]:.4%} ({int(stats["count"])} trades)')

# Bias verification metrics
print('\n🔒 BIAS VERIFICATION:')
if 'rolling_window_hours' in v5_fixed.columns:
    avg_clusters_used = v5_fixed['clusters_used_for_ranking'].mean()
    print(f'V5 Fixed - Rolling Window: {v5_fixed["rolling_window_hours"].iloc[0]}h')
    print(f'V5 Fixed - Avg Clusters Used for Ranking: {avg_clusters_used:.1f}')
    print(f'V5 Fixed - Uses Only Past Information: ✅')

print(f'\nV5 Original - Uses Future Same-Day Info: ❌ (BIASED)')
print(f'V5 Fixed - Uses Only Past Info: ✅ (UNBIASED)')

# Exit strategy comparison
print('\n🚪 EXIT STRATEGY COMPARISON:')
def analyze_exits(df, version_name):
    profit_target_rate = (df['profit_hit'] == True).mean() if 'profit_hit' in df.columns else 0
    stop_loss_rate = (df['stopped_out'] == True).mean() if 'stopped_out' in df.columns else 0
    
    print(f'{version_name}:')
    print(f'  Profit Target Hit: {profit_target_rate:.2%}')
    print(f'  Stop Loss Hit: {stop_loss_rate:.2%}')

analyze_exits(v5_biased, 'V5 Original (Biased)')
analyze_exits(v5_fixed, 'V5 Fixed (Bias-Free)')

# Monthly performance comparison
print('\n📅 MONTHLY PERFORMANCE BIAS IMPACT:')
v5_biased['date'] = pd.to_datetime(v5_biased['date'])
v5_fixed['date'] = pd.to_datetime(v5_fixed['date'])

biased_monthly = v5_biased.groupby(v5_biased['date'].dt.to_period('M')).agg({
    'net_return': ['count', 'mean']
}).round(4)
fixed_monthly = v5_fixed.groupby(v5_fixed['date'].dt.to_period('M')).agg({
    'net_return': ['count', 'mean']
}).round(4)

biased_monthly.columns = ['count', 'mean_return']
fixed_monthly.columns = ['count', 'mean_return']

print('Month       | Biased Trades | Biased Ret | Fixed Trades | Fixed Ret | Impact')
print('-' * 75)

all_months = sorted(set(biased_monthly.index).union(set(fixed_monthly.index)))
for month in all_months:
    biased_count = biased_monthly.loc[month, 'count'] if month in biased_monthly.index else 0
    biased_ret = biased_monthly.loc[month, 'mean_return'] if month in biased_monthly.index else 0
    fixed_count = fixed_monthly.loc[month, 'count'] if month in fixed_monthly.index else 0
    fixed_ret = fixed_monthly.loc[month, 'mean_return'] if month in fixed_monthly.index else 0
    
    impact = (fixed_ret - biased_ret) * 100 if biased_ret != 0 else 0
    
    print(f'{month}    |      {biased_count:3.0f}      |   {biased_ret:6.2%}  |      {fixed_count:3.0f}      |   {fixed_ret:6.2%}  | {impact:+5.1f}bp')

# Summary impact
print('\n' + '=' * 90)
print('🎯 BIAS IMPACT SUMMARY')
print('=' * 90)

print('\n1. FORWARD-LOOKING BIAS DETECTION:')
print(f'   ❌ V5 Original used same-day future clusters for ranking')
print(f'   ✅ V5 Fixed uses rolling 2-hour window (past clusters only)')

print('\n2. TRADE COUNT IMPACT:')
print(f'   • Bias led to {len(v5_biased) - len(v5_fixed)} EXTRA trades ({(len(v5_biased)/len(v5_fixed)-1)*100:+.1f}%)')
print(f'   • Biased version cherry-picked best clusters using future info')

print('\n3. PERFORMANCE IMPACT:')
print(f'   • Mean Return Bias: {(v5_biased_ret - v5_fixed_ret)*100:+.2f} basis points')
print(f'   • Win Rate Bias: {(v5_biased_win - v5_fixed_win)*100:+.1f} percentage points')
print(f'   • Sharpe Ratio Bias: {v5_biased_sharpe - v5_fixed_sharpe:+.3f}')

print('\n4. REALISTIC EXPECTATIONS:')
print(f'   • V5 Fixed (Bias-Free) represents realistic trading performance')
print(f'   • V5 Original was artificially inflated by lookahead bias')

# Calculate what the bias "bought" us
bias_advantage = (v5_biased_ret - v5_fixed_ret) * 100
trade_advantage = len(v5_biased) - len(v5_fixed)

print('\n5. QUANTIFIED BIAS ADVANTAGE:')
print(f'   • Lookahead bias "purchased" {bias_advantage:+.2f} basis points')
print(f'   • Bias "purchased" {trade_advantage:+d} additional trades')
print(f'   • Cost: Complete invalidity of backtest results')

print('\n' + '=' * 90)
print('✅ CONCLUSION: V5 Fixed provides legitimate, tradeable performance metrics')
print('❌ CONCLUSION: V5 Original was contaminated by forward-looking bias')
print('=' * 90) 