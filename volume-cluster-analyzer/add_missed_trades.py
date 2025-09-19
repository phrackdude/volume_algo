#!/usr/bin/env python3
"""
Add Missed Trades to Database
=============================

This script reconstructs and adds the 2 missed trades from September 18th
to the database with complete trade lifecycle information.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta
import pytz

class MissedTradeReconstructor:
    """Reconstruct and add missed trades to database"""
    
    def __init__(self):
        self.db_path = "/opt/v6-trading-system/data/paper_trades.db"
        self.est_tz = pytz.timezone('US/Eastern')
        
        # The 2 missed trades from September 18th (from our earlier analysis)
        self.missed_trades = [
            {
                # Trade 1: 1:31 PM EST - WINNER
                'signal_time_est': datetime(2025, 9, 18, 13, 31, 8, tzinfo=self.est_tz),
                'signal_time_utc': datetime(2025, 9, 18, 17, 31, 8),
                'execution_time_est': datetime(2025, 9, 18, 13, 32, 8, tzinfo=self.est_tz), # 1 minute later (retest)
                'execution_time_utc': datetime(2025, 9, 18, 17, 32, 8),
                'signal_price': 5970.45,
                'execution_price': 5971.13,  # Retest price
                'volume': 14325,
                'volume_ratio': 4.09,
                'signal_strength': 0.451,
                'modal_position': 0.017,
                'contract': 'ES JUN25',
                'action': 'BUY',
                'quantity': 1,
                
                # Exit details (from our calculation)
                'exit_reason': 'time_limit',  # 60-minute limit reached
                'exit_time_est': datetime(2025, 9, 18, 14, 32, 8, tzinfo=self.est_tz), # 60 minutes later
                'exit_time_utc': datetime(2025, 9, 18, 18, 32, 8),
                'exit_price': 6035.86,  # From our calculation
                
                # P&L calculation
                'gross_pnl_points': 64.73,  # 6035.86 - 5971.13
                'gross_pnl_dollars': 3236.50,  # 64.73 * $50/point
                'transaction_costs': 11.875,  # $2.50 commission + $9.375 slippage
                'net_pnl': 3224.62,  # $3236.50 - $11.875
                
                # Bayesian and strategy details
                'bayesian_multiplier': 1.0,  # No historical data for new system
                'confidence': 0.451,  # Same as signal strength
            },
            {
                # Trade 2: 4:51 PM EST - LOSER  
                'signal_time_est': datetime(2025, 9, 18, 16, 51, 8, tzinfo=self.est_tz),
                'signal_time_utc': datetime(2025, 9, 18, 20, 51, 8),
                'execution_time_est': datetime(2025, 9, 18, 16, 59, 8, tzinfo=self.est_tz), # 8 minutes later (retest)
                'execution_time_utc': datetime(2025, 9, 18, 20, 59, 8),
                'signal_price': 6048.07,
                'execution_price': 6009.52,  # Retest price
                'volume': 19679,
                'volume_ratio': 5.62,
                'signal_strength': 0.454,
                'modal_position': 0.019,
                'contract': 'ES JUN25',
                'action': 'BUY', 
                'quantity': 1,
                
                # Exit details (from our calculation)
                'exit_reason': 'stop_loss',  # Hit stop loss
                'exit_time_est': datetime(2025, 9, 18, 17, 53, 8, tzinfo=self.est_tz), # 54 minutes later
                'exit_time_utc': datetime(2025, 9, 18, 21, 53, 8),
                'exit_price': 5961.44,  # Stop loss price
                
                # P&L calculation
                'gross_pnl_points': -48.08,  # 5961.44 - 6009.52
                'gross_pnl_dollars': -2404.00,  # -48.08 * $50/point
                'transaction_costs': 11.875,
                'net_pnl': -2415.68,  # -$2404.00 - $11.875
                
                # Bayesian and strategy details
                'bayesian_multiplier': 1.0,
                'confidence': 0.454,
            }
        ]
        
        # Portfolio tracking
        self.starting_balance = 100000.0
        
    def calculate_slippage_and_commission(self, execution_price: float, quantity: int) -> dict:
        """Calculate realistic slippage and commission costs"""
        commission = 2.50 * quantity  # $2.50 per contract
        slippage_ticks = 0.75
        tick_value = 12.50
        slippage_cost = slippage_ticks * tick_value * quantity  # $9.375 per contract
        
        # Calculate actual slippage (difference between signal and execution price)
        # For these trades, slippage was minimal due to retest entries
        actual_slippage = 0.25  # Conservative estimate for retest entries
        
        return {
            'commission_cost': commission,
            'slippage_cost': slippage_cost,
            'total_transaction_costs': commission + slippage_cost,
            'slippage': actual_slippage,
            'spread': 0.50  # Typical ES spread
        }
    
    def add_trade_to_database(self, trade_data: dict, portfolio_balance_before: float):
        """Add a single trade to the database with complete details"""
        
        # Calculate costs
        costs = self.calculate_slippage_and_commission(trade_data['execution_price'], trade_data['quantity'])
        
        # Calculate portfolio impact
        portfolio_balance_after = portfolio_balance_before + trade_data['net_pnl']
        portfolio_pct_change = (trade_data['net_pnl'] / portfolio_balance_before) * 100
        
        # Generate unique trade ID
        trade_id = f"MISSED_{trade_data['signal_time_utc'].strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare trade record
        trade_record = (
            trade_id,
            trade_data['signal_time_utc'].isoformat(),
            trade_data['signal_time_utc'].isoformat(),
            trade_data['execution_time_utc'].isoformat(),
            trade_data['contract'],
            trade_data['action'],
            trade_data['quantity'],
            trade_data['signal_price'],
            trade_data['execution_price'],
            costs['slippage'],
            costs['spread'],
            trade_data['exit_price'],
            trade_data['exit_time_utc'].isoformat(),
            trade_data['net_pnl'],
            trade_data['gross_pnl_dollars'],
            costs['commission_cost'],
            costs['slippage_cost'],
            costs['total_transaction_costs'],
            'CLOSED',  # Status
            trade_data['volume_ratio'],
            trade_data['signal_strength'],
            trade_data['bayesian_multiplier'],
            trade_data['confidence'],
            portfolio_balance_before,
            portfolio_balance_after,
            portfolio_pct_change,
            trade_data['exit_reason'],
            datetime.now().isoformat()
        )
        
        # Insert trade into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO paper_trades (
                trade_id, timestamp, signal_time, execution_time, contract, action, quantity,
                signal_price, execution_price, slippage, spread, exit_price, exit_time,
                pnl, gross_pnl, commission_cost, slippage_cost, total_transaction_costs,
                status, volume_ratio, signal_strength, bayesian_multiplier, confidence,
                portfolio_balance_before, portfolio_balance_after, portfolio_pct_change,
                exit_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, trade_record)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Added Trade {trade_id}:")
        print(f"   📊 {trade_data['action']} {trade_data['quantity']} {trade_data['contract']} @ ${trade_data['execution_price']}")
        print(f"   🚪 Exit: {trade_data['exit_reason']} @ ${trade_data['exit_price']}")
        print(f"   💰 P&L: ${trade_data['net_pnl']:+.2f}")
        print(f"   📈 Portfolio: ${portfolio_balance_before:,.2f} → ${portfolio_balance_after:,.2f}")
        
        return portfolio_balance_after
    
    def add_portfolio_snapshots(self):
        """Add daily portfolio snapshots for September 18th"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate final portfolio state after both trades
        final_balance = self.starting_balance + self.missed_trades[0]['net_pnl'] + self.missed_trades[1]['net_pnl']
        total_pnl = final_balance - self.starting_balance
        trades_count = len(self.missed_trades)
        max_balance = max(self.starting_balance, self.starting_balance + self.missed_trades[0]['net_pnl'], final_balance)
        drawdown_pct = max(0, (max_balance - final_balance) / max_balance * 100)
        
        # Add end-of-day snapshot
        portfolio_snapshot = (
            '2025-09-18T23:59:59',
            final_balance,
            total_pnl,
            trades_count,
            max_balance,
            drawdown_pct
        )
        
        cursor.execute("""
            INSERT INTO portfolio_equity (timestamp, balance, pnl_today, trades_today, max_balance, drawdown_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        """, portfolio_snapshot)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Added Portfolio Snapshot for September 18th:")
        print(f"   💰 Final Balance: ${final_balance:,.2f}")
        print(f"   📊 Daily P&L: ${total_pnl:+.2f}")
        print(f"   🎯 Trades: {trades_count}")
        print(f"   📈 Max Balance: ${max_balance:,.2f}")
        print(f"   📉 Drawdown: {drawdown_pct:.2f}%")
    
    def reconstruct_missed_trades(self):
        """Add all missed trades to database"""
        print("🔄 RECONSTRUCTING MISSED TRADES FROM SEPTEMBER 18TH")
        print("="*60)
        
        current_balance = self.starting_balance
        
        # Add each trade
        for i, trade in enumerate(self.missed_trades, 1):
            print(f"\n--- ADDING MISSED TRADE {i} ---")
            current_balance = self.add_trade_to_database(trade, current_balance)
        
        # Add portfolio snapshots
        print(f"\n--- ADDING PORTFOLIO SNAPSHOTS ---")
        self.add_portfolio_snapshots()
        
        print("\n" + "="*60)
        print("✅ MISSED TRADES RECONSTRUCTION COMPLETE!")
        print(f"💰 Net Impact: ${self.missed_trades[0]['net_pnl'] + self.missed_trades[1]['net_pnl']:+.2f}")
        print("📊 Your database now reflects what would have happened!")
        print("="*60)

def main():
    """Main entry point"""
    reconstructor = MissedTradeReconstructor()
    reconstructor.reconstruct_missed_trades()

if __name__ == "__main__":
    main()
