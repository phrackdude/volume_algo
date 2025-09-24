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
    
    def __init__(self, starting_balance: float = 100000.0, trading_system: RealTimeTradingSystem = None):
        # Use provided trading system or create new one
        self.trading_system = trading_system or RealTimeTradingSystem()
        self.executor = RealisticExecutionSimulator()
        self.open_trades: List[PaperTrade] = []
        self.closed_trades: List[PaperTrade] = []
        self.db_path = "/opt/v6-trading-system/data/paper_trades.db"
        
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
        
        # The recommendation already contains signal + Bayesian scaling, just apply portfolio risk limits
        recommended_quantity = recommendation.quantity
        
        # Cap at risk-based maximum
        contracts = max(1, min(max_contracts, recommended_quantity))
        
        # Additional safety: don't risk more than 5% of portfolio on a single trade
        max_risk_amount = self.current_balance * 0.05
        max_contracts_by_risk = int(max_risk_amount / dollars_per_contract)
        contracts = min(contracts, max_contracts_by_risk)
        
        # For $100k portfolio, reasonable position sizes are 1-5 contracts typically
        contracts = min(contracts, 5)  # Cap at 5 contracts max
        
        logger.info(f"📊 Position sizing: Risk={risk_pct:.1%} (${risk_amount:.0f}), "
                   f"Recommended={recommended_quantity} (signal+Bayesian scaled), "
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
    
    def calculate_robust_volatility(self, entry_time: datetime, entry_price: float) -> float:
        """
        Production-ready volatility calculation with multiple fallbacks and cost optimization
        Uses shorter lookback periods and robust estimation methods
        """
        try:
            if self.trading_system.current_data.empty:
                return self._get_fallback_volatility(entry_price)
            
            # COST OPTIMIZATION: Use much shorter lookbacks for live trading
            # Try progressively shorter periods if data is insufficient
            lookback_options = [
                480,   # 8 hours (1 trading day) - PREFERRED for live trading
                240,   # 4 hours (half day) 
                120,   # 2 hours (minimum for reasonable estimate)
                60     # 1 hour (emergency fallback)
            ]
            
            for lookback_minutes in lookback_options:
                historical_data = self.trading_system.current_data[
                    self.trading_system.current_data.index <= entry_time
                ].tail(lookback_minutes)
                
                if len(historical_data) >= 60:  # Minimum 60 bars for reasonable estimate
                    # Calculate multiple volatility measures for robustness
                    close_prices = historical_data['close']
                    
                    # Method 1: Close-to-close volatility (standard)
                    returns = close_prices.pct_change().dropna()
                    close_vol = returns.std() if len(returns) > 10 else None
                    
                    # Method 2: Garman-Klass volatility (uses OHLC - more accurate)
                    if all(col in historical_data.columns for col in ['high', 'low', 'open']):
                        gk_vol = self._calculate_garman_klass_volatility(historical_data)
                    else:
                        gk_vol = None
                    
                    # Method 3: ATR-based volatility (robust to gaps)
                    atr_vol = self._calculate_atr_volatility(historical_data, entry_price)
                    
                    # Choose best available method
                    volatility = self._select_best_volatility(close_vol, gk_vol, atr_vol, lookback_minutes)
                    
                    if volatility and 0.001 <= volatility <= 0.05:  # Sanity check: 0.1% to 5%
                        logger.info(f"📊 Volatility: {volatility:.4f} ({volatility*100:.2f}%) from {len(historical_data)} bars ({lookback_minutes/60:.1f}h)")
                        return volatility
                    
            logger.warning("⚠️  All volatility calculations failed - using fallback")
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility: {e}")
        
        return self._get_fallback_volatility(entry_price)
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame) -> Optional[float]:
        """Garman-Klass volatility estimator - more efficient than close-to-close"""
        try:
            if len(data) < 10:
                return None
                
            # Garman-Klass formula: 0.5 * log(H/L)^2 - (2*log(2)-1) * log(C/O)^2
            log_hl = np.log(data['high'] / data['low'])
            log_co = np.log(data['close'] / data['open'])
            
            gk_values = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
            return np.sqrt(gk_values.mean()) if len(gk_values) > 0 else None
            
        except Exception as e:
            logger.debug(f"GK volatility calculation failed: {e}")
            return None
    
    def _calculate_atr_volatility(self, data: pd.DataFrame, entry_price: float) -> Optional[float]:
        """ATR-based volatility - robust to price gaps and missing data"""
        try:
            if len(data) < 14:  # Need minimum data for ATR
                return None
                
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close_prev = np.abs(data['high'] - data['close'].shift(1))
            low_close_prev = np.abs(data['low'] - data['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = true_range.rolling(window=14).mean().iloc[-1]  # 14-period ATR
            
            # Convert ATR to volatility (percentage of entry price)
            atr_volatility = atr / entry_price if entry_price > 0 else None
            return atr_volatility
            
        except Exception as e:
            logger.debug(f"ATR volatility calculation failed: {e}")
            return None
    
    def _select_best_volatility(self, close_vol: Optional[float], gk_vol: Optional[float], 
                               atr_vol: Optional[float], lookback_minutes: int) -> Optional[float]:
        """Select the most reliable volatility measure"""
        
        # Preference order: Garman-Klass > Close-to-close > ATR
        candidates = []
        
        if gk_vol and 0.001 <= gk_vol <= 0.05:
            candidates.append(('garman_klass', gk_vol, 1.0))  # Highest weight
            
        if close_vol and 0.001 <= close_vol <= 0.05:
            candidates.append(('close_to_close', close_vol, 0.8))
            
        if atr_vol and 0.001 <= atr_vol <= 0.05:
            candidates.append(('atr', atr_vol, 0.6))  # Lowest weight but most robust
        
        if not candidates:
            return None
            
        # For short lookbacks, prefer more robust methods
        if lookback_minutes <= 120:  # Less than 2 hours
            # Weight ATR higher for short periods
            for i, (method, vol, weight) in enumerate(candidates):
                if method == 'atr':
                    candidates[i] = (method, vol, weight * 1.5)
        
        # Return highest weighted method
        best_method, best_vol, _ = max(candidates, key=lambda x: x[2])
        logger.debug(f"Selected {best_method} volatility: {best_vol:.4f}")
        
        return best_vol
    
    def _get_fallback_volatility(self, entry_price: float) -> float:
        """Intelligent fallback volatility based on ES futures characteristics"""
        
        # ES futures typical volatility ranges (based on price level)
        if entry_price >= 6000:      # High price levels
            return 0.008  # 0.8% (more volatile at high prices)
        elif entry_price >= 4000:    # Medium price levels  
            return 0.006  # 0.6%
        else:                        # Lower price levels
            return 0.004  # 0.4%
    
    def calculate_profit_target_and_stop(self, entry_price: float, direction: str, volatility: float) -> tuple:
        """Calculate profit target and stop loss matching your V6 specification"""
        # V6 Parameters
        TIGHTER_STOPS = True  # Use 1.0x volatility instead of 1.5x
        PROFIT_TARGET_RATIO = 2.0  # 2:1 risk/reward
        
        # Stop distance calculation
        if TIGHTER_STOPS:
            stop_distance = 1.0 * volatility * entry_price
        else:
            stop_distance = 1.5 * volatility * entry_price
        
        # Minimum stop distance (0.5% of entry price)
        min_stop = 0.005 * entry_price
        stop_distance = max(stop_distance, min_stop)
        
        # Profit target based on risk/reward ratio
        profit_distance = stop_distance * PROFIT_TARGET_RATIO
        
        if direction == "long":
            stop_price = entry_price - stop_distance
            profit_target = entry_price + profit_distance
        else:  # short
            stop_price = entry_price + stop_distance
            profit_target = entry_price - profit_distance
        
        return profit_target, stop_price, stop_distance
    
    def check_exit_conditions_with_highs_lows(self, trade: PaperTrade, current_bar: dict) -> Optional[tuple]:
        """
        Check exit conditions using bar high/low data as per your specification
        Returns (exit_price, exit_reason) or None
        """
        # Calculate stops and targets if not already done
        if not hasattr(trade, 'profit_target'):
            volatility = self.calculate_robust_volatility(trade.execution_time, trade.execution_price)
            direction = "long" if trade.action in ["BUY", "LONG"] else "short"
            
            profit_target, stop_price, stop_distance = self.calculate_profit_target_and_stop(
                trade.execution_price, direction, volatility
            )
            
            # Store in trade object for future reference
            trade.profit_target = profit_target
            trade.stop_price = stop_price
            trade.stop_distance = stop_distance
            
            logger.info(f"📊 Trade {trade.trade_id} stops set:")
            logger.info(f"   Entry: ${trade.execution_price:.2f}")
            logger.info(f"   Stop: ${stop_price:.2f} (${stop_distance:.2f} distance)")
            logger.info(f"   Target: ${profit_target:.2f} (2:1 R/R)")
        
        # Check time-based exit first (60 minutes maximum)
        time_elapsed = (datetime.now() - trade.execution_time).total_seconds() / 60  # minutes
        if time_elapsed > 60:
            return current_bar['close'], "time"
        
        # Check stops and targets using high/low data (hard levels)
        if trade.action in ["BUY", "LONG"]:
            # Long position: check profit target first, then stop
            if current_bar['high'] >= trade.profit_target:
                return trade.profit_target, "profit_target"
            elif current_bar['low'] <= trade.stop_price:
                return trade.stop_price, "stop_loss"
        else:  # SHORT position
            # Short position: check profit target first, then stop  
            if current_bar['low'] <= trade.profit_target:
                return trade.profit_target, "profit_target"
            elif current_bar['high'] >= trade.stop_price:
                return trade.stop_price, "stop_loss"
        
        # Portfolio protection: emergency stop if portfolio drawdown exceeds 5%
        if self.max_drawdown_pct > 5.0:
            return current_bar['close'], "emergency_stop"
            
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
        
        # Calculate V6 transaction costs (matching your specification: $11.875 per contract per trade)
        # Your spec: Commission $2.50 + Slippage 0.75 ticks * $12.50 = $11.875 per contract per trade
        cost_per_contract = self.commission_per_contract + (self.slippage_ticks * self.tick_value)  # $2.50 + $9.375 = $11.875
        total_transaction_costs = cost_per_contract * trade.quantity
        
        # Break down for tracking
        total_commission = self.commission_per_contract * trade.quantity
        total_slippage_cost = (self.slippage_ticks * self.tick_value) * trade.quantity
        
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
                # Check for new market data every minute
                await asyncio.sleep(60)  # Check every minute
                
                # Get latest market data from Databento
                current_time = datetime.now()
                logger.info(f"🔍 Checking for market data at {current_time}")
                
                # The trading system should have real data from Databento connector
                # If no real data is available, log a warning but continue
                if self.trading_system.current_data.empty:
                    logger.warning("⚠️  No market data available from Databento - system waiting for data")
                    continue
                
                # Get the latest data point
                latest_data = self.trading_system.current_data.tail(1)
                if latest_data.empty:
                    logger.warning("⚠️  No recent market data available")
                    continue
                
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
                
                # Check exit conditions for open trades using proper high/low monitoring
                if not latest_data.empty:
                    current_bar = {
                        'high': latest_data['high'].iloc[-1],
                        'low': latest_data['low'].iloc[-1], 
                        'close': latest_data['close'].iloc[-1],
                        'timestamp': latest_data.index[-1]
                    }
                    
                    for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
                        exit_result = self.check_exit_conditions_with_highs_lows(trade, current_bar)
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