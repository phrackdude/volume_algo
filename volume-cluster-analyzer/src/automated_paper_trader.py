#!/usr/bin/env python3
"""
Automated Paper Trading System with Realistic Execution

Simulates real trading conditions:
- 20-second order entry delays
- Offer-side fills (realistic slippage)
- Bid/ask spread simulation
- Automatic Bayesian feedback
- Continuous learning and adaptation
"""

import asyncio
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import logging
from pathlib import Path
import random

# Import our existing components
from real_time_trading_system import (
    RealTimeTradingSystem, 
    TradingRecommendation, 
    VolumeCluster,
    play_alert_sound
)

logger = logging.getLogger(__name__)

@dataclass
class PaperTrade:
    """Paper trade execution record with V6 transaction cost tracking"""
    trade_id: str
    timestamp: datetime
    signal_time: datetime
    execution_time: datetime
    contract: str
    action: str
    quantity: int
    signal_price: float
    execution_price: float
    slippage: float
    spread: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    gross_pnl: Optional[float] = None
    commission_cost: Optional[float] = None
    slippage_cost: Optional[float] = None
    total_transaction_costs: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED
    volume_ratio: float = 0.0
    signal_strength: float = 0.0
    bayesian_multiplier: float = 1.0
    confidence: float = 0.5
    portfolio_balance_before: Optional[float] = None
    portfolio_balance_after: Optional[float] = None
    portfolio_pct_change: Optional[float] = None
    exit_reason: Optional[str] = None

class RealisticExecutionSimulator:
    """Simulates realistic order execution with market microstructure"""
    
    def __init__(self):
        self.typical_spread = 0.25  # ES typical spread (0.25 points = $12.50)
        self.execution_delay = 20   # 20 seconds as requested
        
    def simulate_bid_ask_spread(self, mid_price: float, volatility: float = 0.01) -> tuple:
        """Simulate realistic bid/ask spread based on market conditions"""
        
        # Spread widens during high volatility
        base_spread = self.typical_spread
        volatility_multiplier = 1.0 + (volatility * 10)  # Higher vol = wider spreads
        
        # Add some randomness (spreads vary)
        spread_noise = random.uniform(0.8, 1.2)
        actual_spread = base_spread * volatility_multiplier * spread_noise
        
        half_spread = actual_spread / 2
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        
        return bid, ask, actual_spread
    
    def simulate_execution_price(self, signal_price: float, action: str, 
                                volatility: float = 0.01) -> tuple:
        """Simulate realistic execution price (always on offer side)"""
        
        bid, ask, spread = self.simulate_bid_ask_spread(signal_price, volatility)
        
        if action in ["BUY", "LONG"]:
            # Buying - pay the ask (worse price)
            execution_price = ask
            slippage = execution_price - signal_price
        else:  # SHORT/SELL
            # Selling - hit the bid (worse price)  
            execution_price = bid
            slippage = signal_price - execution_price
            
        return execution_price, slippage, spread
    
    async def execute_order_with_delay(self, recommendation: TradingRecommendation) -> PaperTrade:
        """Execute order with realistic 20-second delay"""
        
        signal_time = recommendation.timestamp
        trade_id = f"PT_{signal_time.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
        
        logger.info(f"📝 Order received: {recommendation.action} {recommendation.quantity} @ ${recommendation.price:.2f}")
        logger.info(f"⏱️  Simulating 20-second order entry delay...")
        
        # 20-second delay for order entry
        await asyncio.sleep(self.execution_delay)
        
        execution_time = datetime.now()
        
        # Simulate realistic execution price
        execution_price, slippage, spread = self.simulate_execution_price(
            recommendation.price, 
            recommendation.action
        )
        
        trade = PaperTrade(
            trade_id=trade_id,
            timestamp=signal_time,
            signal_time=signal_time,
            execution_time=execution_time,
            contract=recommendation.contract,
            action=recommendation.action,
            quantity=recommendation.quantity,
            signal_price=recommendation.price,
            execution_price=execution_price,
            slippage=slippage,
            spread=spread,
            volume_ratio=0.0,  # Will be filled by caller
            signal_strength=recommendation.signal_strength,
            bayesian_multiplier=recommendation.bayesian_multiplier,
            confidence=recommendation.confidence
        )
        
        logger.info(f"✅ Order executed: {trade.action} {trade.quantity} @ ${trade.execution_price:.2f}")
        logger.info(f"   Slippage: ${trade.slippage:.2f} | Spread: ${trade.spread:.2f}")
        
        return trade

class AutomatedPaperTrader:
    """Fully automated paper trading system with $100,000 portfolio"""
    
    def __init__(self, starting_balance: float = 100000.0):
        self.trading_system = RealTimeTradingSystem()
        self.executor = RealisticExecutionSimulator()
        self.open_trades: List[PaperTrade] = []
        self.closed_trades: List[PaperTrade] = []
        self.db_path = "../data/paper_trades.db"
        
        # Portfolio tracking
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.equity_curve: List[Dict] = []
        self.max_balance = starting_balance
        self.max_drawdown_pct = 0.0
        
        # V6 Transaction costs (matching backtest structure)
        self.commission_per_contract = 2.50  # $2.50 per contract per side
        self.slippage_ticks = 0.75  # 0.75 ticks slippage
        self.tick_value = 12.50  # $12.50 per tick for ES
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        
        self.init_paper_trading_db()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_slippage = 0.0
        
        # Advanced performance metrics
        self.daily_returns: List[float] = []
        self.trade_returns: List[float] = []
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.current_consecutive_wins = 0
        self.current_consecutive_losses = 0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        
        logger.info(f"💰 Starting Portfolio: ${self.starting_balance:,.2f}")
        logger.info(f"💸 Transaction Costs: ${self.commission_per_contract}/contract + {self.slippage_ticks} ticks slippage")
    
    def calculate_advanced_metrics(self):
        """Calculate advanced performance metrics"""
        if len(self.trade_returns) < 2:
            return
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
        returns_array = np.array(self.trade_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return > 0:
            self.sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                self.sortino_ratio = mean_return / downside_std * np.sqrt(252)
        
        # Calculate Calmar ratio (return / max drawdown)
        if self.max_drawdown_pct > 0:
            annual_return = ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
            self.calmar_ratio = annual_return / self.max_drawdown_pct
    
    def update_consecutive_stats(self, trade_pnl: float):
        """Update consecutive win/loss statistics"""
        if trade_pnl > 0:
            self.current_consecutive_wins += 1
            self.current_consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.current_consecutive_wins)
        else:
            self.current_consecutive_losses += 1
            self.current_consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_consecutive_losses)
    
    def init_paper_trading_db(self):
        """Initialize paper trading database with schema migration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists and get its schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if new columns exist
            cursor.execute("PRAGMA table_info(paper_trades)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add missing columns if they don't exist
            new_columns = [
                ('gross_pnl', 'REAL'),
                ('commission_cost', 'REAL'),
                ('slippage_cost', 'REAL'),
                ('total_transaction_costs', 'REAL'),
                ('exit_reason', 'TEXT')
            ]
            
            for col_name, col_type in new_columns:
                if col_name not in columns:
                    cursor.execute(f'ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}')
                    logger.info(f"Added column {col_name} to paper_trades table")
        else:
            # Create new table with full schema
            cursor.execute('''
                CREATE TABLE paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    signal_time DATETIME,
                    execution_time DATETIME,
                    contract TEXT,
                    action TEXT,
                    quantity INTEGER,
                    signal_price REAL,
                    execution_price REAL,
                    slippage REAL,
                    spread REAL,
                    exit_price REAL,
                    exit_time DATETIME,
                    pnl REAL,
                    gross_pnl REAL,
                    commission_cost REAL,
                    slippage_cost REAL,
                    total_transaction_costs REAL,
                    status TEXT,
                    volume_ratio REAL,
                    signal_strength REAL,
                    bayesian_multiplier REAL,
                    confidence REAL,
                    portfolio_balance_before REAL,
                    portfolio_balance_after REAL,
                    portfolio_pct_change REAL,
                    exit_reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Portfolio equity curve table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_equity (
                timestamp DATETIME PRIMARY KEY,
                balance REAL,
                pnl_today REAL,
                trades_today INTEGER,
                max_balance REAL,
                drawdown_pct REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Paper trading database initialized: {self.db_path}")
    
    def calculate_position_size(self, recommendation: TradingRecommendation) -> int:
        """Calculate position size based on portfolio balance and V6 Bayesian risk management"""
        
        # V6 Risk management: 1-2% of portfolio per trade based on Bayesian confidence
        risk_pct = 0.01 + (recommendation.confidence * 0.01)  # 1-2% risk
        risk_amount = self.current_balance * risk_pct
        
        # ES futures: $50 per point, typical stop is ~15-20 points (tighter stops from V6)
        stop_loss_points = 15  # Tighter stops as per V6 strategy
        dollars_per_contract = stop_loss_points * 50  # $750 risk per contract
        
        # Calculate max contracts based on risk
        max_contracts = int(risk_amount / dollars_per_contract)
        
        # Apply V6 Bayesian multiplier with proper scaling
        base_quantity = recommendation.quantity
        bayesian_quantity = int(base_quantity * recommendation.bayesian_multiplier)
        
        # Cap at risk-based maximum
        contracts = max(1, min(max_contracts, bayesian_quantity))
        
        # Additional safety: don't risk more than 5% of portfolio on a single trade
        max_risk_amount = self.current_balance * 0.05
        max_contracts_by_risk = int(max_risk_amount / dollars_per_contract)
        contracts = min(contracts, max_contracts_by_risk)
        
        # For $100k portfolio, reasonable position sizes are 1-5 contracts typically
        contracts = min(contracts, 5)  # Cap at 5 contracts max
        
        logger.info(f"📊 Position sizing: Risk={risk_pct:.1%} (${risk_amount:.0f}), "
                   f"Bayesian={recommendation.bayesian_multiplier:.2f}x, "
                   f"Final size={contracts} contracts")
        
        return max(1, contracts)
    
    def update_portfolio_balance(self, trade: PaperTrade):
        """Update portfolio balance and track equity curve"""
        
        # Update balance
        old_balance = self.current_balance
        self.current_balance += trade.pnl
        
        # Track max balance and drawdown
        if self.current_balance > self.max_balance:
            self.max_balance = self.current_balance
        
        current_drawdown = (self.max_balance - self.current_balance) / self.max_balance * 100
        if current_drawdown > self.max_drawdown_pct:
            self.max_drawdown_pct = current_drawdown
        
        # Record in trade
        trade.portfolio_balance_before = old_balance
        trade.portfolio_balance_after = self.current_balance
        trade.portfolio_pct_change = (trade.pnl / old_balance) * 100
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': trade.exit_time,
            'balance': self.current_balance,
            'pnl': trade.pnl,
            'pct_change': trade.portfolio_pct_change,
            'trade_id': trade.trade_id
        })
        
        # Save to database
        self.save_equity_point(trade.exit_time)
    
    def save_equity_point(self, timestamp: datetime):
        """Save equity curve point to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate today's stats
        today = timestamp.date()
        today_trades = len([t for t in self.closed_trades if t.exit_time.date() == today])
        today_pnl = sum([t.pnl for t in self.closed_trades if t.exit_time.date() == today])
        
        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_equity 
            (timestamp, balance, pnl_today, trades_today, max_balance, drawdown_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, self.current_balance, today_pnl, today_trades, 
              self.max_balance, self.max_drawdown_pct))
        
        conn.commit()
        conn.close()
    
    def save_trade_to_db(self, trade: PaperTrade):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        trade_dict = asdict(trade)
        
        # Prepare values with proper handling of None values
        values = (
            trade.trade_id, trade.timestamp, trade.signal_time, trade.execution_time,
            trade.contract, trade.action, trade.quantity, trade.signal_price,
            trade.execution_price, trade.slippage, trade.spread, trade.exit_price,
            trade.exit_time, trade.pnl, trade.gross_pnl, trade.commission_cost, 
            trade.slippage_cost, trade.total_transaction_costs, trade.status, 
            trade.volume_ratio, trade.signal_strength, trade.bayesian_multiplier, 
            trade.confidence, trade.portfolio_balance_before, trade.portfolio_balance_after, 
            trade.portfolio_pct_change, trade.exit_reason
        )
        
        cursor.execute('''
            INSERT OR REPLACE INTO paper_trades 
            (trade_id, timestamp, signal_time, execution_time, contract, action, 
             quantity, signal_price, execution_price, slippage, spread, 
             exit_price, exit_time, pnl, gross_pnl, commission_cost, slippage_cost, 
             total_transaction_costs, status, volume_ratio, signal_strength, 
             bayesian_multiplier, confidence, portfolio_balance_before, portfolio_balance_after, 
             portfolio_pct_change, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)
        
        conn.commit()
        conn.close()
    
    def simulate_exit_conditions(self, trade: PaperTrade, current_price: float) -> Optional[tuple]:
        """Enhanced V6 risk management with dynamic stop losses and profit targets"""
        
        # Calculate dynamic risk parameters based on V6 strategy
        volatility = 0.01  # Simplified volatility estimate
        base_stop_distance = 0.015  # 1.5% base stop
        base_profit_distance = 0.03  # 3% base profit target (2:1 ratio)
        
        # Adjust based on signal strength and confidence
        signal_adjustment = 1.0 + (trade.signal_strength - 0.5) * 0.5  # 0.75x to 1.25x
        confidence_adjustment = 1.0 + (trade.confidence - 0.5) * 0.5   # 0.75x to 1.25x
        
        # Apply adjustments
        stop_distance = base_stop_distance * signal_adjustment * confidence_adjustment
        profit_distance = base_profit_distance * signal_adjustment * confidence_adjustment
        
        # Ensure minimum and maximum bounds
        stop_distance = max(0.005, min(stop_distance, 0.025))  # 0.5% to 2.5%
        profit_distance = max(0.01, min(profit_distance, 0.05))  # 1% to 5%
        
        if trade.action in ["BUY", "LONG"]:
            # Long position
            profit_target = trade.execution_price * (1 + profit_distance)
            stop_loss = trade.execution_price * (1 - stop_distance)
            
            if current_price >= profit_target:
                return current_price, "PROFIT_TARGET"
            elif current_price <= stop_loss:
                return current_price, "STOP_LOSS"
                
        else:  # SHORT position
            profit_target = trade.execution_price * (1 - profit_distance)
            stop_loss = trade.execution_price * (1 + stop_distance)
            
            if current_price <= profit_target:
                return current_price, "PROFIT_TARGET"
            elif current_price >= stop_loss:
                return current_price, "STOP_LOSS"
        
        # Time-based exit (hold for max 6 hours for V6 strategy)
        max_hold_time = 21600  # 6 hours
        if (datetime.now() - trade.execution_time).total_seconds() > max_hold_time:
            return current_price, "TIME_EXIT"
        
        # Portfolio protection: emergency stop if portfolio drawdown exceeds 5%
        if self.max_drawdown_pct > 5.0:
            return current_price, "EMERGENCY_STOP"
            
        return None
    
    def should_stop_trading(self) -> bool:
        """Check if trading should be stopped due to risk management rules"""
        
        # Stop if portfolio drawdown exceeds 10%
        if self.max_drawdown_pct > 10.0:
            logger.warning(f"🛑 Portfolio drawdown {self.max_drawdown_pct:.1f}% exceeds 10% limit")
            return True
        
        # Stop if too many consecutive losses
        if self.current_consecutive_losses >= 5:
            logger.warning(f"🛑 {self.current_consecutive_losses} consecutive losses - stopping trading")
            return True
        
        # Stop if daily loss exceeds 3% of portfolio
        today = datetime.now().date()
        today_trades = [t for t in self.closed_trades if t.exit_time and t.exit_time.date() == today]
        today_pnl = sum([t.pnl for t in today_trades])
        today_loss_pct = abs(today_pnl) / self.starting_balance * 100 if today_pnl < 0 else 0
        
        if today_loss_pct > 3.0:
            logger.warning(f"🛑 Daily loss {today_loss_pct:.1f}% exceeds 3% limit")
            return True
        
        # Stop if too many open positions (risk concentration)
        if len(self.open_trades) >= 3:
            logger.warning(f"🛑 Too many open positions ({len(self.open_trades)}) - stopping new trades")
            return True
        
        return False
    
    async def close_trade(self, trade: PaperTrade, exit_price: float, exit_reason: str):
        """Close a trade and record results with proper V6 transaction costs"""
        
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = "CLOSED"
        
        # Calculate gross P&L (before transaction costs)
        if trade.action in ["BUY", "LONG"]:
            gross_pnl = (exit_price - trade.execution_price) * trade.quantity * 50  # ES = $50/point
        else:  # SHORT
            gross_pnl = (trade.execution_price - exit_price) * trade.quantity * 50
        
        # Calculate V6 transaction costs
        # Entry costs: commission + slippage
        entry_commission = self.commission_per_contract * trade.quantity
        entry_slippage_cost = abs(trade.slippage) * self.tick_value * trade.quantity
        
        # Exit costs: commission + slippage (simulate exit slippage)
        exit_commission = self.commission_per_contract * trade.quantity
        exit_slippage = self.slippage_ticks * 0.25  # 0.75 ticks * $0.25 per tick
        exit_slippage_cost = exit_slippage * self.tick_value * trade.quantity
        
        # Total transaction costs
        total_commission = entry_commission + exit_commission
        total_slippage_cost = entry_slippage_cost + exit_slippage_cost
        total_transaction_costs = total_commission + total_slippage_cost
        
        # Net P&L after transaction costs
        trade.pnl = gross_pnl - total_transaction_costs
        
        # Store transaction cost details in trade record
        trade.gross_pnl = gross_pnl
        trade.commission_cost = total_commission
        trade.slippage_cost = total_slippage_cost
        trade.total_transaction_costs = total_transaction_costs
        trade.exit_reason = exit_reason
        
        # Update transaction cost tracking
        self.total_commission_paid += total_commission
        self.total_slippage_cost += total_slippage_cost
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.total_slippage += total_slippage_cost
        
        if trade.pnl > 0:
            self.winning_trades += 1
        
        # Track trade return for advanced metrics
        trade_return_pct = trade.pnl / trade.portfolio_balance_before if trade.portfolio_balance_before else 0
        self.trade_returns.append(trade_return_pct)
        
        # Update consecutive statistics
        self.update_consecutive_stats(trade.pnl)
        
        # Calculate advanced metrics
        self.calculate_advanced_metrics()
        
        # Calculate modal bin context for Bayesian feedback
        modal_position = abs(trade.execution_price - trade.signal_price) / trade.execution_price
        context_value = min(int(modal_position * 10), 9)
        
        # Feed back to Bayesian system
        self.trading_system.bayesian_manager.record_trade_result(
            "modal_bin", 
            context_value,
            trade.execution_price,
            exit_price,
            trade.volume_ratio,
            trade.signal_strength
        )
        
        # Update portfolio balance
        self.update_portfolio_balance(trade)
        
        # Save to database
        self.save_trade_to_db(trade)
        
        # Remove from open trades
        self.open_trades = [t for t in self.open_trades if t.trade_id != trade.trade_id]
        self.closed_trades.append(trade)
        
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        logger.info(f"🔚 Trade closed: {trade.trade_id}")
        logger.info(f"   P&L: ${trade.pnl:.2f} | Reason: {exit_reason}")
        logger.info(f"   📊 Running Stats: {self.winning_trades}/{self.total_trades} wins ({win_rate:.1f}%)")
        logger.info(f"   💰 Total P&L: ${self.total_pnl:.2f} | Avg: ${avg_pnl:.2f}")
        
        # Play audio alert for trade closure
        if trade.pnl > 0:
            play_alert_sound("signal")  # Winning trade
        
    def print_performance_summary(self):
        """Print current performance summary with portfolio tracking and V6 transaction costs"""
        if self.total_trades == 0:
            return
            
        win_rate = (self.winning_trades / self.total_trades) * 100
        avg_pnl = self.total_pnl / self.total_trades
        avg_commission = self.total_commission_paid / self.total_trades if self.total_trades > 0 else 0
        avg_slippage_cost = self.total_slippage_cost / self.total_trades if self.total_trades > 0 else 0
        
        # Portfolio metrics
        total_return = ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
        
        # Calculate gross P&L (before transaction costs)
        gross_pnl = self.total_pnl + self.total_commission_paid + self.total_slippage_cost
        transaction_cost_impact = (self.total_commission_paid + self.total_slippage_cost) / abs(gross_pnl) * 100 if gross_pnl != 0 else 0
        
        print("\n" + "="*70)
        print("📊 V6 BAYESIAN PAPER TRADING PERFORMANCE")
        print("="*70)
        print(f"💰 PORTFOLIO STATUS:")
        print(f"   Starting Balance: ${self.starting_balance:,.2f}")
        print(f"   Current Balance:  ${self.current_balance:,.2f}")
        print(f"   Total Return:     {total_return:+.2f}%")
        print(f"   Max Drawdown:     -{self.max_drawdown_pct:.2f}%")
        print()
        print(f"📈 TRADING STATS:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Win Rate: {win_rate:.1f}% ({self.winning_trades} wins)")
        print(f"   Net P&L: ${self.total_pnl:,.2f}")
        print(f"   Gross P&L: ${gross_pnl:,.2f}")
        print(f"   Average P&L: ${avg_pnl:.2f}")
        print(f"   Open Positions: {len(self.open_trades)}")
        print()
        print(f"💸 V6 TRANSACTION COSTS:")
        print(f"   Total Commission: ${self.total_commission_paid:,.2f}")
        print(f"   Total Slippage:   ${self.total_slippage_cost:,.2f}")
        print(f"   Avg Commission:   ${avg_commission:.2f}/trade")
        print(f"   Avg Slippage:     ${avg_slippage_cost:.2f}/trade")
        print(f"   Cost Impact:      {transaction_cost_impact:.1f}% of gross P&L")
        print()
        print(f"📊 ADVANCED METRICS:")
        print(f"   Sharpe Ratio:     {self.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio:    {self.sortino_ratio:.2f}")
        print(f"   Calmar Ratio:     {self.calmar_ratio:.2f}")
        print(f"   Max Consecutive Wins:  {self.max_consecutive_wins}")
        print(f"   Max Consecutive Losses: {self.max_consecutive_losses}")
        print("="*70)
    
    async def run_automated_trading(self):
        """Main automated trading loop with V6 Bayesian strategy"""
        logger.info("🤖 Starting V6 Bayesian Automated Paper Trading System")
        logger.info(f"💰 Initial Portfolio: ${self.starting_balance:,.2f}")
        logger.info("   - 20-second execution delays")
        logger.info("   - Realistic offer-side fills")
        logger.info("   - V6 Bayesian position sizing")
        logger.info("   - $2.50 commission + 0.75 tick slippage per contract")
        logger.info("   - Automatic Bayesian learning from trade results")
        logger.info("   - Portfolio-based risk management (1-2% per trade)")
        logger.info("   - Continuous learning enabled")
        
        # Connect to data feed
        connected = await self.trading_system.connect_to_databento()
        if not connected:
            logger.error("Failed to connect to data feed")
            return
        
        while True:
            try:
                # Simulate market data (replace with real Databento feed)
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                current_price = 6010.0 + random.uniform(-5, 5)  # Simulate price movement
                
                # Simulate market data
                simulated_data = pd.DataFrame({
                    'timestamp': [current_time],
                    'open': [current_price - 1],
                    'high': [current_price + 2],
                    'low': [current_price - 2],
                    'close': [current_price],
                    'volume': [random.randint(8000, 20000)]  # Variable volume
                })
                simulated_data.set_index('timestamp', inplace=True)
                
                # Update trading system data
                self.trading_system.current_data = pd.concat([
                    self.trading_system.current_data, 
                    simulated_data
                ]).tail(100)
                
                # Risk management checks
                if self.should_stop_trading():
                    logger.warning("🛑 Risk management triggered - stopping new trades")
                    await asyncio.sleep(300)  # Wait 5 minutes before checking again
                    continue
                
                # Check for new signals
                cluster = self.trading_system.detect_volume_cluster(self.trading_system.current_data)
                
                if cluster and (self.trading_system.last_cluster_time is None or 
                               (cluster.timestamp - self.trading_system.last_cluster_time).total_seconds() > 1800):
                    
                    # Generate recommendation
                    recommendation = self.trading_system.generate_trading_recommendation(cluster)
                    
                    # Calculate portfolio-based position size
                    portfolio_quantity = self.calculate_position_size(recommendation)
                    recommendation.quantity = portfolio_quantity
                    
                    # Print recommendation (with audio)
                    self.trading_system.print_recommendation(recommendation)
                    
                    logger.info(f"💰 Portfolio balance: ${self.current_balance:,.2f}")
                    logger.info(f"📊 Position size: {portfolio_quantity} contracts (portfolio-based)")
                    
                    # Execute trade automatically with delay
                    paper_trade = await self.executor.execute_order_with_delay(recommendation)
                    paper_trade.volume_ratio = cluster.volume_ratio
                    
                    # Add to open trades
                    self.open_trades.append(paper_trade)
                    self.save_trade_to_db(paper_trade)
                    
                    self.trading_system.last_cluster_time = cluster.timestamp
                
                # Check exit conditions for open trades
                for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
                    exit_result = self.simulate_exit_conditions(trade, current_price)
                    if exit_result:
                        exit_price, exit_reason = exit_result
                        await self.close_trade(trade, exit_price, exit_reason)
                
                # Print performance summary every 5 trades
                if self.total_trades > 0 and self.total_trades % 5 == 0:
                    self.print_performance_summary()
                    
                    # Print Bayesian learning summary every 10 trades
                    if self.total_trades % 10 == 0:
                        self.trading_system.bayesian_manager.print_bayesian_summary()
                
            except Exception as e:
                logger.error(f"Error in automated trading loop: {e}")
                await asyncio.sleep(10)

async def main():
    """Main entry point for automated paper trading"""
    trader = AutomatedPaperTrader()
    await trader.run_automated_trading()

if __name__ == "__main__":
    print("🤖 V6 BAYESIAN AUTOMATED PAPER TRADING SYSTEM")
    print("="*60)
    print("💰 Portfolio: $100,000")
    print("📊 Strategy: V6 Bayesian Volume Cluster")
    print("⏱️  Features:")
    print("   - 20-second execution delays")
    print("   - Realistic offer-side fills")
    print("   - $2.50 commission + 0.75 tick slippage")
    print("   - V6 Bayesian position sizing")
    print("   - Automatic learning from trade results")
    print("   - Portfolio-based risk management")
    print("   - Continuous operation")
    print("="*60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Automated trading stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}") 