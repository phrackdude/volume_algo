import pandas as pd
import numpy as np

# Load the results
df = pd.read_csv('../data/backtest_results.csv')

print('=== ROLLING TOP-2 FILTER DETAILED ANALYSIS ===')
print()

# Short confirmation analysis
short_df = df[df['direction'] == 'short']
if len(short_df) > 0:
    confirmed_shorts = short_df[short_df['short_confirmed'] == True]
    print(f'Short Confirmation Analysis:')
    print(f'  - Total Short Signals: {len(short_df)}')
    print(f'  - Confirmed Shorts: {len(confirmed_shorts)} ({len(confirmed_shorts)/len(short_df)*100:.1f}%)')
    print(f'  - Confirmed Short Win Rate: {(confirmed_shorts["blended_return"] > 0).mean():.2%}')
    print()

# Best and worst trades
print('Top 5 Best Trades:')
best_trades = df.nlargest(5, 'blended_return')[['date', 'direction', 'blended_return', 'volume_ratio']]
for _, trade in best_trades.iterrows():
    print(f'  {trade["date"]}: {trade["direction"]} {trade["blended_return"]:.3%} (vol: {trade["volume_ratio"]:.1f}x)')
print()

print('Top 5 Worst Trades:')
worst_trades = df.nsmallest(5, 'blended_return')[['date', 'direction', 'blended_return', 'volume_ratio']]
for _, trade in worst_trades.iterrows():
    print(f'  {trade["date"]}: {trade["direction"]} {trade["blended_return"]:.3%} (vol: {trade["volume_ratio"]:.1f}x)')
print()

# Risk metrics
print('Risk Metrics:')
cumulative_returns = df['blended_return'].cumsum()
max_cumulative = cumulative_returns.expanding().max()
drawdown = cumulative_returns - max_cumulative
max_drawdown = drawdown.min()
print(f'  - Maximum Drawdown: {max_drawdown:.4%}')

sharpe_ratio = df['blended_return'].mean() / df['blended_return'].std() * np.sqrt(252) if df['blended_return'].std() > 0 else 0
print(f'  - Annualized Sharpe Ratio: {sharpe_ratio:.2f}')
print()

# Strategy comparison vs previous version
print('=== STRATEGY COMPARISON ===')
print('Rolling Top-2 vs Fixed Top-2 Daily:')
print(f'  - Trade Count: 400 vs 328 (+22% more trades)')
print(f'  - Win Rate: 48.75% vs 50.61% (-1.86pp)')
print(f'  - Mean Return: -0.0887% vs -0.0628% (-0.026pp)')
print(f'  - Long Bias: 96.5% vs 96.6% (similar)')
print()

# Trading frequency analysis
daily_trades = df.groupby('date').size()
print(f'Trading Frequency:')
print(f'  - Average trades per day: {daily_trades.mean():.1f}')
print(f'  - Days with trades: {len(daily_trades)} out of ~312 days')
print(f'  - Max trades in a day: {daily_trades.max()}')
print(f'  - Days with 0 trades: {312 - len(daily_trades)}') 