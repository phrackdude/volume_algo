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
    """Paper trade execution record"""
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
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED
    volume_ratio: float = 0.0
    signal_strength: float = 0.0
    bayesian_multiplier: float = 1.0
    confidence: float = 0.5
    portfolio_balance_before: Optional[float] = None
    portfolio_balance_after: Optional[float] = None
    portfolio_pct_change: Optional[float] = None

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
    """Fully automated paper trading system"""
    
    def __init__(self, starting_balance: float = 15000.0):
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
        
        self.init_paper_trading_db()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_slippage = 0.0
        
        logger.info(f"💰 Starting Portfolio: ${self.starting_balance:,.2f}")
    
    def init_paper_trading_db(self):
        """Initialize paper trading database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
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
                status TEXT,
                volume_ratio REAL,
                signal_strength REAL,
                bayesian_multiplier REAL,
                confidence REAL,
                portfolio_balance_before REAL,
                portfolio_balance_after REAL,
                portfolio_pct_change REAL,
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
        """Calculate position size based on portfolio balance and risk management"""
        
        # Risk 1-2% of portfolio per trade based on Bayesian confidence
        risk_pct = 0.01 + (recommendation.confidence * 0.01)  # 1-2% risk
        risk_amount = self.current_balance * risk_pct
        
        # ES futures: $50 per point, typical stop is ~20 points
        stop_loss_points = 20
        dollars_per_contract = stop_loss_points * 50  # $1000 risk per contract
        
        # Calculate max contracts based on risk
        max_contracts = int(risk_amount / dollars_per_contract)
        
        # Apply Bayesian multiplier but cap at reasonable size
        contracts = max(1, min(max_contracts, int(recommendation.quantity * recommendation.bayesian_multiplier)))
        
        # Don't risk more than 10% of portfolio on a single trade
        max_position_value = self.current_balance * 0.10
        contracts = min(contracts, int(max_position_value / (recommendation.price * 50)))
        
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
        
        cursor.execute('''
            INSERT OR REPLACE INTO paper_trades 
            (trade_id, timestamp, signal_time, execution_time, contract, action, 
             quantity, signal_price, execution_price, slippage, spread, 
             exit_price, exit_time, pnl, status, volume_ratio, signal_strength, 
             bayesian_multiplier, confidence, portfolio_balance_before, portfolio_balance_after, 
             portfolio_pct_change)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.timestamp, trade.signal_time, trade.execution_time,
            trade.contract, trade.action, trade.quantity, trade.signal_price,
            trade.execution_price, trade.slippage, trade.spread, trade.exit_price,
            trade.exit_time, trade.pnl, trade.status, trade.volume_ratio,
            trade.signal_strength, trade.bayesian_multiplier, trade.confidence,
            trade.portfolio_balance_before, trade.portfolio_balance_after, trade.portfolio_pct_change
        ))
        
        conn.commit()
        conn.close()
    
    def simulate_exit_conditions(self, trade: PaperTrade, current_price: float) -> Optional[tuple]:
        """Check if trade should be closed (profit target or stop loss)"""
        
        if trade.action in ["BUY", "LONG"]:
            # Long position
            profit_target = trade.execution_price * 1.015  # 1.5% profit target
            stop_loss = trade.execution_price * 0.985      # 1.5% stop loss
            
            if current_price >= profit_target:
                return current_price, "PROFIT_TARGET"
            elif current_price <= stop_loss:
                return current_price, "STOP_LOSS"
                
        else:  # SHORT position
            profit_target = trade.execution_price * 0.985  # 1.5% profit target
            stop_loss = trade.execution_price * 1.015      # 1.5% stop loss
            
            if current_price <= profit_target:
                return current_price, "PROFIT_TARGET"
            elif current_price >= stop_loss:
                return current_price, "STOP_LOSS"
        
        # Time-based exit (hold for max 4 hours)
        if (datetime.now() - trade.execution_time).total_seconds() > 14400:  # 4 hours
            return current_price, "TIME_EXIT"
            
        return None
    
    async def close_trade(self, trade: PaperTrade, exit_price: float, exit_reason: str):
        """Close a trade and record results"""
        
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = "CLOSED"
        
        # Calculate P&L
        if trade.action in ["BUY", "LONG"]:
            trade.pnl = (exit_price - trade.execution_price) * trade.quantity * 50  # ES = $50/point
        else:  # SHORT
            trade.pnl = (trade.execution_price - exit_price) * trade.quantity * 50
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.total_slippage += abs(trade.slippage) * trade.quantity * 50
        
        if trade.pnl > 0:
            self.winning_trades += 1
        
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
        """Print current performance summary with portfolio tracking"""
        if self.total_trades == 0:
            return
            
        win_rate = (self.winning_trades / self.total_trades) * 100
        avg_pnl = self.total_pnl / self.total_trades
        avg_slippage = self.total_slippage / self.total_trades
        
        # Portfolio metrics
        total_return = ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
        
        print("\n" + "="*60)
        print("📊 AUTOMATED PAPER TRADING PERFORMANCE")
        print("="*60)
        print(f"💰 PORTFOLIO STATUS:")
        print(f"   Starting Balance: ${self.starting_balance:,.2f}")
        print(f"   Current Balance:  ${self.current_balance:,.2f}")
        print(f"   Total Return:     {total_return:+.2f}%")
        print(f"   Max Drawdown:     -{self.max_drawdown_pct:.2f}%")
        print()
        print(f"📈 TRADING STATS:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Win Rate: {win_rate:.1f}% ({self.winning_trades} wins)")
        print(f"   Total P&L: ${self.total_pnl:,.2f}")
        print(f"   Average P&L: ${avg_pnl:.2f}")
        print(f"   Average Slippage: ${avg_slippage:.2f}")
        print(f"   Open Positions: {len(self.open_trades)}")
        print("="*60)
    
    async def run_automated_trading(self):
        """Main automated trading loop"""
        logger.info("🤖 Starting Automated Paper Trading System")
        logger.info(f"💰 Initial Portfolio: ${self.starting_balance:,.2f}")
        logger.info("   - 20-second execution delays")
        logger.info("   - Realistic offer-side fills")
        logger.info("   - Portfolio-based position sizing")
        logger.info("   - Automatic Bayesian feedback")
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
                
                # Print performance summary every 10 trades
                if self.total_trades > 0 and self.total_trades % 5 == 0:
                    self.print_performance_summary()
                
            except Exception as e:
                logger.error(f"Error in automated trading loop: {e}")
                await asyncio.sleep(10)

async def main():
    """Main entry point for automated paper trading"""
    trader = AutomatedPaperTrader()
    await trader.run_automated_trading()

if __name__ == "__main__":
    print("🤖 AUTOMATED V6 PAPER TRADING SYSTEM")
    print("="*50)
    print("Features:")
    print("- 20-second execution delays")
    print("- Realistic offer-side fills")
    print("- Automatic Bayesian learning")
    print("- Continuous operation")
    print("="*50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Automated trading stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}") 