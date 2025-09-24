"""
Volume Cluster Strategy - Equity Curve Analysis
Shows how $1,000 grows over time using V3.0 backtest results
Professional visualization for asset manager presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
STARTING_CAPITAL = 1000.0
RESULTS_FILE = "../data/backtest_results_v3.csv"
RISK_PER_TRADE = 0.02  # 2% risk per trade
CONTRACT_VALUE = 5000  # ES contract value approximation

def load_and_prepare_data():
    """Load backtest results and prepare for equity curve calculation"""
    try:
        df = pd.read_csv(RESULTS_FILE)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df = df.sort_values('entry_time').reset_index(drop=True)
        
        print(f"Loaded {len(df)} trades from {df['entry_time'].min().date()} to {df['entry_time'].max().date()}")
        return df
    except FileNotFoundError:
        print(f"Results file not found: {RESULTS_FILE}")
        print("Please run backtest_simulation_v3.py first.")
        return None

def calculate_position_sizes(df, starting_capital):
    """Calculate realistic position sizes based on available capital and risk management"""
    df = df.copy()
    
    # Calculate position size based on risk management
    # Using the stop distance to determine position size
    df['stop_distance_pct'] = abs(df['entry_price'] - df['stop_price']) / df['entry_price']
    
    # Position size = (Risk per trade) / (Stop distance)
    df['capital_at_risk'] = starting_capital * RISK_PER_TRADE
    df['theoretical_position_size'] = df['capital_at_risk'] / (df['stop_distance_pct'] * df['entry_price'])
    
    # Convert to contract equivalent (for ES futures)
    df['contracts'] = df['theoretical_position_size'] / CONTRACT_VALUE
    df['contracts'] = df['contracts'].clip(lower=0.1, upper=10.0)  # Reasonable limits
    
    return df

def calculate_equity_curve(df, starting_capital):
    """Calculate the equity curve over time"""
    equity_data = []
    current_capital = starting_capital
    peak_capital = starting_capital
    max_drawdown = 0.0
    
    for i, trade in df.iterrows():
        # Calculate P&L for this trade
        position_value = trade['contracts'] * CONTRACT_VALUE
        pnl_dollars = trade['net_return'] * position_value
        
        # Update capital
        current_capital += pnl_dollars
        
        # Track drawdown
        if current_capital > peak_capital:
            peak_capital = current_capital
        else:
            drawdown = (peak_capital - current_capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        equity_data.append({
            'date': trade['exit_time'],
            'trade_num': i + 1,
            'capital': current_capital,
            'pnl': pnl_dollars,
            'return_pct': pnl_dollars / (current_capital - pnl_dollars),
            'cumulative_return': (current_capital - starting_capital) / starting_capital,
            'drawdown': (peak_capital - current_capital) / peak_capital,
            'contracts': trade['contracts'],
            'direction': trade['direction'],
            'exit_reason': trade['exit_reason']
        })
    
    equity_df = pd.DataFrame(equity_data)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    return equity_df, max_drawdown

def create_comprehensive_chart(equity_df, starting_capital, max_drawdown):
    """Create a comprehensive equity curve chart with multiple panels"""
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Main equity curve
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(equity_df['date'], equity_df['capital'], linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.axhline(y=starting_capital, color='gray', linestyle='--', alpha=0.7, label='Starting Capital')
    ax1.fill_between(equity_df['date'], starting_capital, equity_df['capital'], 
                     where=(equity_df['capital'] >= starting_capital), 
                     color='green', alpha=0.3, interpolate=True)
    ax1.fill_between(equity_df['date'], starting_capital, equity_df['capital'], 
                     where=(equity_df['capital'] < starting_capital), 
                     color='red', alpha=0.3, interpolate=True)
    
    ax1.set_title('Volume Cluster Strategy - Portfolio Growth ($1,000 Initial Capital)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add performance metrics as text
    final_value = equity_df['capital'].iloc[-1]
    total_return = (final_value - starting_capital) / starting_capital
    
    textstr = f'''Performance Summary:
    Final Value: ${final_value:,.0f}
    Total Return: {total_return:.1%}
    Max Drawdown: {max_drawdown:.1%}
    Trades: {len(equity_df)}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Drawdown chart
    ax2 = plt.subplot(3, 1, 2)
    ax2.fill_between(equity_df['date'], 0, -equity_df['drawdown'] * 100, 
                     color='red', alpha=0.7, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Portfolio Drawdowns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Monthly returns chart
    ax3 = plt.subplot(3, 1, 3)
    
    # Calculate monthly returns
    equity_df_monthly = equity_df.set_index('date').resample('M').last()
    equity_df_monthly['monthly_return'] = equity_df_monthly['capital'].pct_change() * 100
    equity_df_monthly = equity_df_monthly.dropna()
    
    colors = ['green' if x > 0 else 'red' for x in equity_df_monthly['monthly_return']]
    bars = ax3.bar(equity_df_monthly.index, equity_df_monthly['monthly_return'], 
                   color=colors, alpha=0.7)
    
    ax3.set_ylabel('Monthly Return (%)', fontsize=12)
    ax3.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def calculate_performance_metrics(equity_df, starting_capital):
    """Calculate comprehensive performance metrics"""
    final_value = equity_df['capital'].iloc[-1]
    total_return = (final_value - starting_capital) / starting_capital
    
    # Calculate daily returns for Sharpe ratio
    equity_df['daily_return'] = equity_df['capital'].pct_change()
    daily_returns = equity_df['daily_return'].dropna()
    
    # Annualized metrics
    trading_days = len(equity_df)
    time_period = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days / 365.25
    annualized_return = (final_value / starting_capital) ** (1/time_period) - 1
    
    # Risk metrics
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    max_drawdown = equity_df['drawdown'].max()
    
    # Win/Loss analysis
    wins = (equity_df['pnl'] > 0).sum()
    losses = (equity_df['pnl'] < 0).sum()
    win_rate = wins / len(equity_df) if len(equity_df) > 0 else 0
    
    avg_win = equity_df[equity_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = equity_df[equity_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    metrics = {
        'Starting Capital': f"${starting_capital:,.0f}",
        'Final Value': f"${final_value:,.0f}",
        'Total Return': f"{total_return:.1%}",
        'Annualized Return': f"{annualized_return:.1%}",
        'Max Drawdown': f"{max_drawdown:.1%}",
        'Volatility (Annual)': f"{volatility:.1%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Win Rate': f"{win_rate:.1%}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Total Trades': len(equity_df),
        'Winning Trades': wins,
        'Losing Trades': losses,
        'Average Win': f"${avg_win:.2f}",
        'Average Loss': f"${avg_loss:.2f}",
        'Time Period': f"{time_period:.1f} years"
    }
    
    return metrics

def create_summary_table(metrics):
    """Create a formatted summary table"""
    print("\n" + "="*60)
    print("VOLUME CLUSTER STRATEGY - PERFORMANCE SUMMARY")
    print("="*60)
    print(f"📈 PORTFOLIO GROWTH ANALYSIS (Starting: ${STARTING_CAPITAL:,.0f})")
    print("-"*60)
    
    # Key metrics in two columns
    left_metrics = [
        ('Final Portfolio Value', metrics['Final Value']),
        ('Total Return', metrics['Total Return']),
        ('Annualized Return', metrics['Annualized Return']),
        ('Max Drawdown', metrics['Max Drawdown']),
        ('Sharpe Ratio', metrics['Sharpe Ratio']),
        ('Win Rate', metrics['Win Rate']),
        ('Profit Factor', metrics['Profit Factor'])
    ]
    
    right_metrics = [
        ('Time Period', metrics['Time Period']),
        ('Total Trades', str(metrics['Total Trades'])),
        ('Winning Trades', str(metrics['Winning Trades'])),
        ('Losing Trades', str(metrics['Losing Trades'])),
        ('Average Win', metrics['Average Win']),
        ('Average Loss', metrics['Average Loss']),
        ('Annual Volatility', metrics['Volatility (Annual)'])
    ]
    
    for i in range(max(len(left_metrics), len(right_metrics))):
        left = left_metrics[i] if i < len(left_metrics) else ('', '')
        right = right_metrics[i] if i < len(right_metrics) else ('', '')
        print(f"{left[0]:<25} {left[1]:<15} {right[0]:<25} {right[1]}")
    
    print("\n" + "="*60)
    print("INTERPRETATION FOR ASSET MANAGER:")
    print("="*60)
    
    final_value = float(metrics['Final Value'].replace('$', '').replace(',', ''))
    total_return = float(metrics['Total Return'].replace('%', '')) / 100
    
    if total_return > 0:
        print(f"✅ POSITIVE PERFORMANCE: Strategy generated {metrics['Total Return']} return")
        print(f"✅ PORTFOLIO GROWTH: ${STARTING_CAPITAL:,.0f} grew to {metrics['Final Value']}")
        print(f"✅ CONSISTENCY: {metrics['Win Rate']} win rate over {metrics['Total Trades']} trades")
        print(f"✅ RISK MANAGEMENT: {metrics['Max Drawdown']} maximum drawdown")
        print(f"✅ RISK-ADJUSTED RETURN: {metrics['Sharpe Ratio']} Sharpe ratio")
    else:
        print(f"⚠️  NEGATIVE PERFORMANCE: Strategy lost {abs(total_return):.1%}")
        
    print(f"\n🎯 BOTTOM LINE: This strategy demonstrates {'profitable' if total_return > 0 else 'unprofitable'}")
    print(f"   systematic trading with clear risk management and realistic transaction costs.")

def main():
    """Main execution function"""
    print("Volume Cluster Strategy - Equity Curve Analysis")
    print("=" * 50)
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Calculate position sizes
    df = calculate_position_sizes(df, STARTING_CAPITAL)
    
    # Calculate equity curve
    equity_df, max_drawdown = calculate_equity_curve(df, STARTING_CAPITAL)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity_df, STARTING_CAPITAL)
    
    # Create comprehensive chart
    fig = create_comprehensive_chart(equity_df, STARTING_CAPITAL, max_drawdown)
    
    # Save the chart
    chart_filename = f"../data/equity_curve_analysis_{datetime.now().strftime('%Y%m%d')}.png"
    fig.savefig(chart_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nChart saved as: {chart_filename}")
    
    # Display summary
    create_summary_table(metrics)
    
    # Save detailed equity data
    equity_filename = f"../data/equity_curve_data_{datetime.now().strftime('%Y%m%d')}.csv"
    equity_df.to_csv(equity_filename, index=False)
    print(f"\nDetailed equity data saved as: {equity_filename}")
    
    # Show the plot
    plt.show()
    
    return equity_df, metrics

if __name__ == "__main__":
    equity_df, metrics = main() 