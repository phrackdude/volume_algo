#!/usr/bin/env python3
"""
Simple V6 Analysis without external dependencies
Analyzes your 156 trades using only built-in Python
"""

import csv
import statistics
from collections import Counter

def analyze_v6_data():
    print("🚀 V6 SIMPLE ANALYSIS (No External Dependencies)")
    print("=" * 60)
    
    # Read CSV data
    trades = []
    try:
        with open('./data/backtest_results_v6.csv', 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        print(f"✅ Loaded {len(trades)} trades successfully")
    except FileNotFoundError:
        print("❌ Could not find backtest_results_v6.csv")
        return
    
    # Convert strings to numbers
    for trade in trades:
        trade['position_adjusted_return'] = float(trade['position_adjusted_return'])
        trade['net_return'] = float(trade['net_return'])
        trade['bayesian_multiplier'] = float(trade['bayesian_multiplier'])
        trade['profit_hit'] = trade['profit_hit'] == 'True'
        trade['stopped_out'] = trade['stopped_out'] == 'True'
    
    # Basic performance metrics
    returns = [t['position_adjusted_return'] for t in trades]
    net_returns = [t['net_return'] for t in trades]
    multipliers = [t['bayesian_multiplier'] for t in trades]
    
    # Calculate stats
    avg_return = statistics.mean(returns)
    median_return = statistics.median(returns)
    std_return = statistics.stdev(returns)
    
    win_count = sum(1 for r in net_returns if r > 0)
    win_rate = win_count / len(trades)
    
    avg_multiplier = statistics.mean(multipliers)
    max_multiplier = max(multipliers)
    
    print(f"\n📊 CORE PERFORMANCE METRICS:")
    print(f"   Average Return per Trade: {avg_return:.4%}")
    print(f"   Median Return: {median_return:.4%}")
    print(f"   Standard Deviation: {std_return:.4%}")
    print(f"   Win Rate: {win_rate:.2%} ({win_count}/{len(trades)} trades)")
    print(f"   Average Bayesian Multiplier: {avg_multiplier:.3f}")
    print(f"   Maximum Multiplier Used: {max_multiplier:.3f}")
    
    # Bayesian utilization
    bayesian_trades = sum(1 for t in trades if t['bayesian_diagnostic_method'] == 'bayesian')
    utilization = bayesian_trades / len(trades)
    print(f"   Bayesian Utilization: {utilization:.1%} ({bayesian_trades} trades)")
    
    # Performance distribution
    print(f"\n📈 RETURN DISTRIBUTION:")
    profit_targets = sum(1 for t in trades if t['profit_hit'])
    stop_losses = sum(1 for t in trades if t['stopped_out'])
    time_exits = len(trades) - profit_targets - stop_losses
    
    print(f"   Profit Targets Hit: {profit_targets} ({profit_targets/len(trades):.1%})")
    print(f"   Stop Losses Hit: {stop_losses} ({stop_losses/len(trades):.1%})")
    print(f"   Time Exits: {time_exits} ({time_exits/len(trades):.1%})")
    
    # Robustness indicators
    print(f"\n🎯 ROBUSTNESS INDICATORS:")
    
    # Check for consecutive losses
    consecutive_losses = 0
    max_consecutive_losses = 0
    for trade in trades:
        if trade['net_return'] <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    print(f"   Maximum Consecutive Losses: {max_consecutive_losses}")
    
    # Monthly performance consistency
    monthly_returns = {}
    for trade in trades:
        month = trade['date'][:7]  # YYYY-MM
        if month not in monthly_returns:
            monthly_returns[month] = []
        monthly_returns[month].append(trade['position_adjusted_return'])
    
    positive_months = sum(1 for month_trades in monthly_returns.values() 
                         if statistics.mean(month_trades) > 0)
    
    print(f"   Positive Months: {positive_months}/{len(monthly_returns)} ({positive_months/len(monthly_returns):.1%})")
    
    # Risk metrics
    negative_returns = [r for r in returns if r < 0]
    if negative_returns:
        avg_loss = statistics.mean(negative_returns)
        worst_loss = min(negative_returns)
        print(f"   Average Loss: {avg_loss:.4%}")
        print(f"   Worst Single Loss: {worst_loss:.4%}")
    
    positive_returns = [r for r in returns if r > 0]
    if positive_returns:
        avg_win = statistics.mean(positive_returns)
        best_win = max(positive_returns)
        print(f"   Average Win: {avg_win:.4%}")
        print(f"   Best Single Win: {best_win:.4%}")
        
        if negative_returns:
            reward_risk_ratio = abs(avg_win / avg_loss)
            print(f"   Reward/Risk Ratio: {reward_risk_ratio:.2f}")
    
    # Final assessment
    print(f"\n🏆 STRATEGY ASSESSMENT:")
    
    # Performance thresholds
    excellent_return = avg_return > 0.007  # >0.7% per trade
    excellent_winrate = win_rate > 0.6     # >60% win rate
    high_utilization = utilization > 0.9   # >90% Bayesian usage
    low_drawdown = max_consecutive_losses <= 3
    
    score = sum([excellent_return, excellent_winrate, high_utilization, low_drawdown])
    
    if score >= 4:
        assessment = "🚀 EXCEPTIONAL - Ready for Production"
    elif score >= 3:
        assessment = "✅ EXCELLENT - Proceed with Confidence"
    elif score >= 2:
        assessment = "👍 GOOD - Monitor Closely"
    else:
        assessment = "⚠️  NEEDS IMPROVEMENT"
    
    print(f"   Overall Rating: {assessment}")
    print(f"   Performance Score: {score}/4")
    
    return {
        'avg_return': avg_return,
        'win_rate': win_rate,
        'utilization': utilization,
        'max_consecutive_losses': max_consecutive_losses,
        'assessment': assessment
    }

if __name__ == "__main__":
    analyze_v6_data() 