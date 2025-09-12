#!/usr/bin/env python3
"""
Trade Feedback Interface for V6 Bayesian Trading System
Allows manual input of trade execution results for adaptive learning
"""

import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from real_time_trading_system import BayesianStatsManager

class TradeFeedbackInterface:
    """Interface for recording trade execution results"""
    
    def __init__(self):
        self.bayesian_manager = BayesianStatsManager()
        self.pending_trades_file = "../data/pending_trades.json"
        
    def record_order_execution(self, 
                             recommendation_timestamp: str,
                             executed: bool,
                             fill_price: Optional[float] = None,
                             fill_quantity: Optional[int] = None,
                             execution_notes: str = ""):
        """Record whether an order was executed and at what price"""
        
        # Load the original recommendation
        recommendation = self._load_recommendation(recommendation_timestamp)
        if not recommendation:
            print(f"❌ Could not find recommendation for {recommendation_timestamp}")
            return False
            
        if executed and fill_price:
            # Store as pending trade awaiting exit
            pending_trade = {
                'entry_timestamp': recommendation_timestamp,
                'contract': recommendation['contract'],
                'action': recommendation['action'],
                'quantity': fill_quantity or recommendation['quantity'],
                'entry_price': fill_price,
                'stop_loss': recommendation['stop_loss'],
                'profit_target': recommendation['profit_target'],
                'volume_ratio': recommendation.get('volume_ratio', 0),
                'signal_strength': recommendation['signal_strength'],
                'modal_bin_context': self._calculate_modal_bin_context(recommendation),
                'execution_notes': execution_notes
            }
            
            self._save_pending_trade(pending_trade)
            print(f"✅ Trade recorded: {recommendation['action']} {fill_quantity or recommendation['quantity']} {recommendation['contract']} @ ${fill_price}")
            print(f"📊 Awaiting exit to complete Bayesian learning cycle")
            
        else:
            print(f"⏹️  Trade not executed: {recommendation['action']} {recommendation['contract']}")
            
        return True
    
    def record_trade_exit(self,
                         entry_timestamp: str,
                         exit_price: float,
                         exit_reason: str = "manual",
                         exit_notes: str = ""):
        """Record trade exit for Bayesian learning"""
        
        # Load pending trade
        pending_trade = self._load_pending_trade(entry_timestamp)
        if not pending_trade:
            print(f"❌ Could not find pending trade for {entry_timestamp}")
            return False
            
        # Calculate return
        entry_price = pending_trade['entry_price']
        if pending_trade['action'] in ['BUY', 'LONG']:
            return_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            return_pct = (entry_price - exit_price) / entry_price
            
        # Record in Bayesian database
        self.bayesian_manager.record_trade_result(
            context_type="modal_bin",
            context_value=pending_trade['modal_bin_context'],
            entry_price=entry_price,
            exit_price=exit_price,
            volume_ratio=pending_trade['volume_ratio'],
            signal_strength=pending_trade['signal_strength']
        )
        
        # Remove from pending trades
        self._remove_pending_trade(entry_timestamp)
        
        # Display results
        win_loss = "WIN 🎉" if return_pct > 0 else "LOSS 📉"
        print(f"✅ Trade completed: {win_loss}")
        print(f"📊 Return: {return_pct:.2%}")
        print(f"🧠 Bayesian database updated")
        
        return True
    
    def show_pending_trades(self):
        """Display all pending trades awaiting exit"""
        pending_trades = self._load_all_pending_trades()
        
        if not pending_trades:
            print("📭 No pending trades")
            return
            
        print("📋 PENDING TRADES (Awaiting Exit)")
        print("=" * 60)
        
        for trade in pending_trades:
            entry_time = datetime.fromisoformat(trade['entry_timestamp'])
            current_time = datetime.now()
            duration = current_time - entry_time
            
            print(f"⏰ Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 Trade: {trade['action']} {trade['quantity']} {trade['contract']}")
            print(f"💰 Entry Price: ${trade['entry_price']:.2f}")
            print(f"🎯 Targets: Stop ${trade['stop_loss']:.2f} | Profit ${trade['profit_target']:.2f}")
            print(f"⏱️  Duration: {duration.total_seconds() / 3600:.1f} hours")
            print("-" * 40)
    
    def _load_recommendation(self, timestamp: str) -> Optional[Dict]:
        """Load recommendation from JSON log"""
        try:
            with open("../data/recommendations_log.jsonl", 'r') as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec['timestamp'] == timestamp:
                        return rec
        except FileNotFoundError:
            pass
        return None
    
    def _save_pending_trade(self, trade: Dict):
        """Save pending trade to JSON file"""
        pending_trades = self._load_all_pending_trades()
        pending_trades.append(trade)
        
        with open(self.pending_trades_file, 'w') as f:
            json.dump(pending_trades, f, indent=2)
    
    def _load_pending_trade(self, entry_timestamp: str) -> Optional[Dict]:
        """Load specific pending trade"""
        pending_trades = self._load_all_pending_trades()
        for trade in pending_trades:
            if trade['entry_timestamp'] == entry_timestamp:
                return trade
        return None
    
    def _load_all_pending_trades(self) -> list:
        """Load all pending trades"""
        try:
            with open(self.pending_trades_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _remove_pending_trade(self, entry_timestamp: str):
        """Remove completed trade from pending list"""
        pending_trades = self._load_all_pending_trades()
        pending_trades = [t for t in pending_trades if t['entry_timestamp'] != entry_timestamp]
        
        with open(self.pending_trades_file, 'w') as f:
            json.dump(pending_trades, f, indent=2)
    
    def _calculate_modal_bin_context(self, recommendation: Dict) -> int:
        """Calculate modal bin context from recommendation"""
        # Simplified - in real implementation would use actual calculation
        return 5  # Default middle bin

def interactive_feedback():
    """Interactive command-line interface for trade feedback"""
    interface = TradeFeedbackInterface()
    
    print("🔄 V6 TRADE FEEDBACK INTERFACE")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Record Order Execution")
        print("2. Record Trade Exit") 
        print("3. Show Pending Trades")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n📝 Recording Order Execution")
            timestamp = input("Recommendation timestamp (YYYY-MM-DDTHH:MM:SS): ")
            executed = input("Was order executed? (y/n): ").lower() == 'y'
            
            if executed:
                fill_price = float(input("Fill price: $"))
                fill_quantity = input("Fill quantity (press enter for recommended): ")
                fill_quantity = int(fill_quantity) if fill_quantity else None
                notes = input("Execution notes (optional): ")
                
                interface.record_order_execution(timestamp, True, fill_price, fill_quantity, notes)
            else:
                interface.record_order_execution(timestamp, False)
                
        elif choice == "2":
            print("\n📤 Recording Trade Exit")
            interface.show_pending_trades()
            if interface._load_all_pending_trades():
                timestamp = input("\nEntry timestamp to close: ")
                exit_price = float(input("Exit price: $"))
                exit_reason = input("Exit reason (target/stop/manual/time): ")
                notes = input("Exit notes (optional): ")
                
                interface.record_trade_exit(timestamp, exit_price, exit_reason, notes)
            
        elif choice == "3":
            interface.show_pending_trades()
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    interactive_feedback() 