import pandas as pd
import numpy as np

# Load the optimized results
df = pd.read_csv('../data/backtest_results.csv')

print('=== OPTIMIZED STRATEGY ANALYSIS ===')
print()

# Basic performance metrics
print(f'📊 PERFORMANCE OVERVIEW:')
print(f'Total Trades: {len(df)}')
print(f'Date Range: {df["date"].min()} to {df["date"].max()}')
print(f'Mean Blended Return: {df["blended_return"].mean():.4%}')
print(f'Mean Raw Return: {df["raw_blended_return"].mean():.4%}')
print(f'Median Blended Return: {df["blended_return"].median():.4%}')
print(f'Standard Deviation: {df["blended_return"].std():.4%}')
print(f'Overall Win Rate: {(df["blended_return"] > 0).mean():.2%}')
print()

# Position sizing impact
print(f'🎯 POSITION SIZING IMPACT:')
print(f'Average Position Multiplier: {df["position_multiplier"].mean():.2f}x')
print(f'Max Position Multiplier: {df["position_multiplier"].max():.2f}x')
print(f'Min Position Multiplier: {df["position_multiplier"].min():.2f}x')
print(f'Position-Adjusted vs Raw Return Ratio: {df["blended_return"].mean() / df["raw_blended_return"].mean():.2f}')
print()

# Signal strength analysis
print(f'⚡ SIGNAL STRENGTH ANALYSIS:')
print(f'Average Signal Strength: {df["signal_strength"].mean():.3f}')
print(f'Max Signal Strength: {df["signal_strength"].max():.3f}')
print(f'Min Signal Strength: {df["signal_strength"].min():.3f}')

# Performance by signal strength quartiles
df['signal_quartile'] = pd.qcut(df['signal_strength'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = df[df['signal_quartile'] == quartile]
    if len(subset) > 0:
        print(f'  {quartile} (Strength {subset["signal_strength"].min():.3f}-{subset["signal_strength"].max():.3f}): {subset["blended_return"].mean():.4%} ({(subset["blended_return"] > 0).mean():.1%} win rate)')
print()

# Stop loss analysis
stopped_out_trades = df[df['stopped_out'] == True]
normal_exits = df[df['stopped_out'] == False]

print(f'🛑 STOP LOSS ANALYSIS:')
print(f'Trades Stopped Out: {len(stopped_out_trades)} ({len(stopped_out_trades)/len(df)*100:.1f}%)')
if len(stopped_out_trades) > 0:
    print(f'  - Average Return (Stopped): {stopped_out_trades["blended_return"].mean():.4%}')
    print(f'  - Average Position Size (Stopped): {stopped_out_trades["position_multiplier"].mean():.2f}x')
print(f'Normal Exits: {len(normal_exits)} ({len(normal_exits)/len(df)*100:.1f}%)')
if len(normal_exits) > 0:
    print(f'  - Average Return (Normal): {normal_exits["blended_return"].mean():.4%}')
    print(f'  - Average Position Size (Normal): {normal_exits["position_multiplier"].mean():.2f}x')
print()

# Direction analysis
long_df = df[df['direction'] == 'long']
short_df = df[df['direction'] == 'short']

print(f'📈 DIRECTIONAL ANALYSIS:')
print(f'Long Trades: {len(long_df)} ({len(long_df)/len(df)*100:.1f}%)')
print(f'  - Mean Return: {long_df["blended_return"].mean():.4%}')
print(f'  - Win Rate: {(long_df["blended_return"] > 0).mean():.2%}')
print(f'  - Avg Position Size: {long_df["position_multiplier"].mean():.2f}x')

print(f'Short Trades: {len(short_df)} ({len(short_df)/len(df)*100:.1f}%)')
if len(short_df) > 0:
    print(f'  - Mean Return: {short_df["blended_return"].mean():.4%}')
    print(f'  - Win Rate: {(short_df["blended_return"] > 0).mean():.2%}')
    print(f'  - Avg Position Size: {short_df["position_multiplier"].mean():.2f}x')
print()

# Volume ratio analysis
print(f'📊 VOLUME QUALITY ANALYSIS:')
print(f'Average Volume Ratio: {df["volume_ratio"].mean():.1f}x')
print(f'Median Volume Ratio: {df["volume_ratio"].median():.1f}x')
print(f'Min Volume Ratio: {df["volume_ratio"].min():.1f}x')
print(f'Max Volume Ratio: {df["volume_ratio"].max():.1f}x')

# Performance by volume ratio quartiles
df['volume_quartile'] = pd.qcut(df['volume_ratio'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    subset = df[df['volume_quartile'] == quartile]
    if len(subset) > 0:
        vol_range = f"{subset['volume_ratio'].min():.0f}-{subset['volume_ratio'].max():.0f}x"
        print(f'  {quartile} ({vol_range}): {subset["blended_return"].mean():.4%} ({(subset["blended_return"] > 0).mean():.1%} win rate)')
print()

# Risk metrics
print(f'⚠️ RISK METRICS:')
cumulative_returns = df['blended_return'].cumsum()
max_cumulative = cumulative_returns.expanding().max()
drawdown = cumulative_returns - max_cumulative
max_drawdown = drawdown.min()
print(f'Maximum Drawdown: {max_drawdown:.4%}')

sharpe_ratio = df['blended_return'].mean() / df['blended_return'].std() * np.sqrt(252) if df['blended_return'].std() > 0 else 0
print(f'Annualized Sharpe Ratio: {sharpe_ratio:.2f}')

# Calculate VaR (95% confidence)
var_95 = np.percentile(df['blended_return'], 5)
print(f'95% VaR (daily): {var_95:.4%}')
print()

# Best and worst trades
print(f'🏆 TOP 5 BEST TRADES:')
best_trades = df.nlargest(5, 'blended_return')[['date', 'direction', 'blended_return', 'position_multiplier', 'signal_strength', 'volume_ratio']]
for _, trade in best_trades.iterrows():
    print(f'  {trade["date"]}: {trade["direction"]} {trade["blended_return"]:.3%} (pos: {trade["position_multiplier"]:.2f}x, sig: {trade["signal_strength"]:.3f}, vol: {trade["volume_ratio"]:.0f}x)')
print()

print(f'📉 TOP 5 WORST TRADES:')
worst_trades = df.nsmallest(5, 'blended_return')[['date', 'direction', 'blended_return', 'position_multiplier', 'signal_strength', 'volume_ratio']]
for _, trade in worst_trades.iterrows():
    print(f'  {trade["date"]}: {trade["direction"]} {trade["blended_return"]:.3%} (pos: {trade["position_multiplier"]:.2f}x, sig: {trade["signal_strength"]:.3f}, vol: {trade["volume_ratio"]:.0f}x)')
print()

# Strategy comparison
print(f'📊 OPTIMIZATION IMPACT SUMMARY:')
print(f'Tighter Thresholds (0.25/0.75 vs 0.35/0.65):')
print(f'  - Trade Count: 187 vs 400 (-53% fewer trades)')
print(f'  - Quality Focus: Higher volume ratios, stronger signals')
print(f'  - Win Rate: 44.92% vs 48.75% (-3.83pp)')
print(f'  - Mean Return: -0.1064% vs -0.0887% (-0.018pp worse)')
print()
print(f'Position Sizing Benefits:')
print(f'  - Dynamic sizing based on signal + volume strength')
print(f'  - Range: {df["position_multiplier"].min():.2f}x to {df["position_multiplier"].max():.2f}x')
print(f'  - Leverages strongest signals for better risk-adjusted returns')
print()
print(f'Stop Loss Implementation:')
print(f'  - {len(stopped_out_trades)} trades ({len(stopped_out_trades)/len(df)*100:.1f}%) hit stop loss')
print(f'  - Volatility-based stops (2-sigma)')
print(f'  - Helps limit maximum loss per trade') 