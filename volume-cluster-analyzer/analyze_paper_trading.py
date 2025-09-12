#!/usr/bin/env python3
"""
Analyze Paper Trading Results

Extract and analyze automated paper trading performance from database
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class PaperTradingAnalyzer:
    """Analyze paper trading results from database"""
    
    def __init__(self, db_path="data/paper_trades.db"):
        self.db_path = db_path
        
    def load_trades(self):
        """Load all trades from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("""
                SELECT * FROM paper_trades 
                ORDER BY signal_time DESC
            """, conn)
            conn.close()
            
            if not df.empty:
                # Convert datetime columns
                df['signal_time'] = pd.to_datetime(df['signal_time'])
                df['execution_time'] = pd.to_datetime(df['execution_time'])
                df['exit_time'] = pd.to_datetime(df['exit_time'])
                
            return df
        except Exception as e:
            print(f"Error loading trades: {e}")
            return pd.DataFrame()
    
    def load_bayesian_stats(self):
        """Load Bayesian learning statistics"""
        try:
            conn = sqlite3.connect("data/bayesian_stats.db")
            df = pd.read_sql_query("""
                SELECT context_type, context_value, 
                       COUNT(*) as total_trades,
                       SUM(win) as wins,
                       AVG(return_pct) as avg_return,
                       MIN(trade_timestamp) as first_trade,
                       MAX(trade_timestamp) as last_trade
                FROM context_performance 
                GROUP BY context_type, context_value
                ORDER BY context_value
            """, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading Bayesian stats: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, df):
        """Calculate comprehensive performance metrics"""
        if df.empty:
            return {}
        
        closed_trades = df[df['status'] == 'CLOSED'].copy()
        
        if closed_trades.empty:
            return {"error": "No closed trades found"}
        
        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100
        
        # P&L metrics
        total_pnl = closed_trades['pnl'].sum()
        avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        avg_trade = total_pnl / total_trades
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(closed_trades)
        sharpe_ratio = self.calculate_sharpe_ratio(closed_trades)
        
        # Execution metrics
        avg_slippage = closed_trades['slippage'].mean()
        total_slippage_cost = (closed_trades['slippage'].abs() * closed_trades['quantity'] * 50).sum()
        
        # Time metrics
        trade_durations = (closed_trades['exit_time'] - closed_trades['execution_time']).dt.total_seconds() / 3600
        avg_duration = trade_durations.mean()
        
        # Bayesian metrics
        avg_confidence = closed_trades['confidence'].mean()
        avg_multiplier = closed_trades['bayesian_multiplier'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'profit_factor': avg_win / abs(avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_slippage': avg_slippage,
            'total_slippage_cost': total_slippage_cost,
            'avg_duration_hours': avg_duration,
            'avg_confidence': avg_confidence,
            'avg_bayesian_multiplier': avg_multiplier
        }
    
    def calculate_max_drawdown(self, df):
        """Calculate maximum drawdown"""
        df = df.sort_values('exit_time')
        cumulative_pnl = df['pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, df):
        """Calculate Sharpe ratio (simplified)"""
        if len(df) < 2:
            return 0
        
        returns = df['pnl'] / 100000  # Normalize by account size
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        trades_df = self.load_trades()
        bayesian_df = self.load_bayesian_stats()
        
        print("📊 V6 AUTOMATED PAPER TRADING ANALYSIS")
        print("="*60)
        
        if trades_df.empty:
            print("❌ No trading data found")
            print(f"   Looking for database: {self.db_path}")
            return
        
        metrics = self.calculate_performance_metrics(trades_df)
        
        if 'error' in metrics:
            print(f"❌ {metrics['error']}")
            return
        
        # Portfolio Analysis
        closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
        if not closed_trades.empty and 'portfolio_balance_before' in closed_trades.columns:
            starting_balance = closed_trades['portfolio_balance_before'].iloc[-1]  # First trade
            final_balance = closed_trades['portfolio_balance_after'].iloc[0]       # Last trade
            total_return = ((final_balance - starting_balance) / starting_balance) * 100
            
            print("💰 PORTFOLIO PERFORMANCE")
            print("-" * 25)
            print(f"Starting Balance: ${starting_balance:,.2f}")
            print(f"Final Balance: ${final_balance:,.2f}")
            print(f"Total Return: {total_return:+.2f}%")
            print(f"Total P&L: ${final_balance - starting_balance:+,.2f}")
            
            # Annualized return (if more than a few days)
            days_traded = (closed_trades['exit_time'].max() - closed_trades['exit_time'].min()).days
            if days_traded > 0:
                daily_return = total_return / days_traded
                annualized_return = daily_return * 252  # Trading days per year
                print(f"Annualized Return: {annualized_return:+.1f}%")
            print()
        
        # Trading Summary
        print("🎯 TRADING PERFORMANCE SUMMARY")
        print("-" * 30)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}% ({metrics['winning_trades']} wins, {metrics['losing_trades']} losses)")
        print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"Average P&L per Trade: ${metrics['avg_trade']:.2f}")
        print(f"Average Winning Trade: ${metrics['avg_win']:.2f}")
        print(f"Average Losing Trade: ${metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print()
        
        # Risk Analysis
        print("📉 RISK ANALYSIS")
        print("-" * 15)
        print(f"Maximum Drawdown: ${metrics['max_drawdown']:.2f}")
        if not closed_trades.empty and 'portfolio_balance_before' in closed_trades.columns:
            max_portfolio = closed_trades['portfolio_balance_after'].max()
            current_portfolio = closed_trades['portfolio_balance_after'].iloc[0]
            portfolio_drawdown = ((max_portfolio - current_portfolio) / max_portfolio) * 100
            print(f"Portfolio Drawdown: {portfolio_drawdown:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Average Trade Duration: {metrics['avg_duration_hours']:.1f} hours")
        print()
        
        # Execution Analysis
        print("⚡ EXECUTION ANALYSIS")
        print("-" * 20)
        print(f"Average Slippage: ${metrics['avg_slippage']:.2f} per contract")
        print(f"Total Slippage Cost: ${metrics['total_slippage_cost']:,.2f}")
        print()
        
        # Bayesian Learning Analysis
        print("🧠 BAYESIAN LEARNING ANALYSIS")
        print("-" * 30)
        print(f"Average Confidence: {metrics['avg_confidence']:.1%}")
        print(f"Average Position Multiplier: {metrics['avg_bayesian_multiplier']:.2f}x")
        
        if not bayesian_df.empty:
            print("\n📊 Learning Progress by Modal Bin:")
            for _, row in bayesian_df.iterrows():
                win_rate = (row['wins'] / row['total_trades']) * 100
                print(f"   Bin {row['context_value']}: {row['total_trades']} trades, {win_rate:.1f}% wins")
        
        print()
        
        # Comparison to Backtesting
        print("🎯 COMPARISON TO BACKTEST TARGETS")
        print("-" * 35)
        target_win_rate = 64.7
        target_per_trade = 0.813
        
        actual_win_rate = metrics['win_rate']
        actual_per_trade = (metrics['avg_trade'] / 50) / 40  # Convert to percentage (assuming $40 ES move = 1%)
        
        print(f"Win Rate: {actual_win_rate:.1f}% vs {target_win_rate:.1f}% target")
        print(f"  {'✅ EXCEEDING' if actual_win_rate >= target_win_rate else '⚠️ BELOW'} target")
        
        print(f"Avg Return: {actual_per_trade:.3f}% vs {target_per_trade:.3f}% target")
        print(f"  {'✅ EXCEEDING' if actual_per_trade >= target_per_trade else '⚠️ BELOW'} target")
        
        # Portfolio return comparison
        if not closed_trades.empty and 'portfolio_balance_before' in closed_trades.columns:
            print(f"Portfolio Return: {total_return:.2f}% over {days_traded} days")
            
        # Fun comparison
        print("\n🎉 FUN FACTS")
        print("-" * 10)
        if not closed_trades.empty and 'portfolio_balance_before' in closed_trades.columns:
            profit = final_balance - starting_balance
            if profit > 0:
                print(f"💰 Profit could buy: {int(profit/5)} Starbucks lattes! ☕")
                print(f"📈 Portfolio grew by: {profit/starting_balance*100:.2f}%")
            
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 15)
        
        if metrics['total_trades'] < 30:
            print("📈 Continue trading - need more data for statistical significance")
        
        if metrics['avg_bayesian_multiplier'] < 1.5:
            print("🧠 Bayesian learning still early - expect improvement over time")
        
        if metrics['win_rate'] < 60:
            print("⚠️  Win rate below expectation - review signal quality")
        
        if metrics['avg_slippage'] > 0.5:
            print("⚡ High slippage - consider market order execution for urgent signals")
        
        print("\n" + "="*60)
    
    def export_to_csv(self, filename="paper_trading_results.csv"):
        """Export trading results to CSV"""
        trades_df = self.load_trades()
        if not trades_df.empty:
            trades_df.to_csv(filename, index=False)
            print(f"📁 Trading data exported to {filename}")
        else:
            print("❌ No data to export")
    
    def get_daily_pnl(self):
        """Get daily P&L summary"""
        trades_df = self.load_trades()
        if trades_df.empty:
            return pd.DataFrame()
        
        closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
        if closed_trades.empty:
            return pd.DataFrame()
        
        closed_trades['date'] = closed_trades['exit_time'].dt.date
        daily_pnl = closed_trades.groupby('date').agg({
            'pnl': 'sum',
            'trade_id': 'count'
        }).rename(columns={'trade_id': 'num_trades'})
        
        return daily_pnl

def main():
    """Main analysis function"""
    analyzer = PaperTradingAnalyzer()
    
    print(f"Analyzing paper trading results...")
    print(f"Database: {analyzer.db_path}")
    print()
    
    # Print comprehensive report
    analyzer.print_performance_report()
    
    # Export data
    analyzer.export_to_csv()
    
    # Show daily P&L
    daily_pnl = analyzer.get_daily_pnl()
    if not daily_pnl.empty:
        print("\n📅 DAILY P&L SUMMARY")
        print("-" * 20)
        print(daily_pnl.to_string())
    
    print("\n🎯 Analysis complete!")

if __name__ == "__main__":
    main() 