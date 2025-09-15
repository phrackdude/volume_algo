#!/usr/bin/env python3
"""
V6 Bayesian Real-Time Trading System
Connects to Databento API for live market data and produces trading recommendations
based on the extraordinary V6 Bayesian volume cluster strategy
"""

import asyncio
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging
from pathlib import Path
import os
import platform
import subprocess
import pytz

# Import our custom modules
from databento_connector import DatabentoConnector
from config import config, load_config_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Audio alert configuration
AUDIO_ENABLED = True  # Set to False to disable sounds

def play_alert_sound(alert_type: str = "signal"):
    """Play audio alert for trading signals"""
    if not AUDIO_ENABLED:
        return
    
    try:
        system = platform.system()
        
        if alert_type == "signal":
            if system == "Darwin":  # macOS
                # Play system alert sound
                subprocess.run(['afplay', '/System/Library/Sounds/Glass.aiff'], 
                             check=False, capture_output=True)
            elif system == "Windows":
                # Windows alert sound
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            elif system == "Linux":
                # Linux bell sound
                subprocess.run(['paplay', '/usr/share/sounds/alsa/Front_Left.wav'], 
                             check=False, capture_output=True)
                
        elif alert_type == "urgent":
            # More urgent sound for high-confidence signals
            if system == "Darwin":
                subprocess.run(['afplay', '/System/Library/Sounds/Sosumi.aiff'], 
                             check=False, capture_output=True)
            elif system == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONHAND)
            elif system == "Linux":
                subprocess.run(['paplay', '/usr/share/sounds/alsa/Rear_Center.wav'], 
                             check=False, capture_output=True)
                
    except Exception as e:
        logger.debug(f"Audio alert failed: {e}")

def test_audio_alerts():
    """Test audio alert functionality"""
    print("🔊 Testing audio alerts...")
    print("Playing normal signal sound...")
    play_alert_sound("signal")
    
    import time
    time.sleep(2)
    
    print("Playing urgent signal sound...")
    play_alert_sound("urgent")
    
    print("✅ Audio test complete!")
    print("If you didn't hear sounds, check AUDIO_ENABLED setting or system audio.")

@dataclass
class TradingRecommendation:
    """Trading recommendation output structure matching your order interface"""
    timestamp: datetime
    contract: str  # e.g., "ES JUN25"
    action: str   # "BUY", "SELL", "SHORT"
    quantity: int  # Number of contracts
    order_type: str  # "LIMIT", "MARKET"
    price: Optional[float]  # Limit price if applicable
    validity: str  # "DAY", "GTC", "IOC"
    confidence: float  # Bayesian confidence score (0-1)
    signal_strength: float
    bayesian_multiplier: float
    stop_loss: float
    profit_target: float
    reasoning: str  # Human-readable explanation

@dataclass
class VolumeCluster:
    """Volume cluster data structure"""
    timestamp: datetime
    volume_ratio: float
    modal_price: float
    signal_strength: float
    direction: str
    entry_price: float
    volume_rank: int

class BayesianStatsManager:
    """Manages Bayesian statistics storage and calculations"""
    
    def __init__(self, db_path: str = "/opt/v6-trading-system/data/bayesian_stats.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for Bayesian statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for storing historical performance by context
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_type TEXT NOT NULL,
                context_value INTEGER NOT NULL,
                trade_timestamp DATETIME NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                return_pct REAL NOT NULL,
                win INTEGER NOT NULL,  -- 1 for win, 0 for loss
                volume_ratio REAL,
                signal_strength REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_context_lookup 
            ON context_performance(context_type, context_value)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Bayesian stats database initialized: {self.db_path}")
    
    def get_context_stats(self, context_type: str, context_value: int, min_trades: int = 3) -> Dict:
        """Get enhanced Bayesian statistics for a specific context with rolling performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all-time stats
        cursor.execute('''
            SELECT COUNT(*) as total_trades,
                   SUM(win) as wins,
                   AVG(return_pct) as avg_return,
                   MIN(return_pct) as min_return,
                   MAX(return_pct) as max_return,
                   AVG(CASE WHEN win = 1 THEN return_pct ELSE NULL END) as avg_win_return,
                   AVG(CASE WHEN win = 0 THEN return_pct ELSE NULL END) as avg_loss_return
            FROM context_performance 
            WHERE context_type = ? AND context_value = ?
        ''', (context_type, context_value))
        
        result = cursor.fetchone()
        
        # Get recent performance (last 30 days)
        cursor.execute('''
            SELECT COUNT(*) as recent_trades,
                   SUM(win) as recent_wins,
                   AVG(return_pct) as recent_avg_return
            FROM context_performance 
            WHERE context_type = ? AND context_value = ?
            AND trade_timestamp >= datetime('now', '-30 days')
        ''', (context_type, context_value))
        
        recent_result = cursor.fetchone()
        conn.close()
        
        if result and result[0] >= min_trades:
            total_trades, wins, avg_return, min_return, max_return, avg_win_return, avg_loss_return = result
            recent_trades, recent_wins, recent_avg_return = recent_result
            
            losses = total_trades - wins
            recent_losses = recent_trades - recent_wins if recent_trades else 0
            
            # Calculate Bayesian posterior parameters
            alpha_post = config.alpha_prior + wins
            beta_post = config.beta_prior + losses
            
            # Expected win probability
            expected_p = alpha_post / (alpha_post + beta_post)
            
            # Calculate confidence interval (simplified)
            confidence_interval = 1.96 * np.sqrt(expected_p * (1 - expected_p) / total_trades) if total_trades > 0 else 0
            
            # Recent performance weight (more recent trades get higher weight)
            recent_weight = min(recent_trades / 10.0, 1.0) if recent_trades else 0
            
            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'expected_p': expected_p,
                'avg_return': avg_return or 0,
                'min_return': min_return or 0,
                'max_return': max_return or 0,
                'avg_win_return': avg_win_return or 0,
                'avg_loss_return': avg_loss_return or 0,
                'alpha_post': alpha_post,
                'beta_post': beta_post,
                'confidence_interval': confidence_interval,
                'recent_trades': recent_trades,
                'recent_wins': recent_wins,
                'recent_win_rate': recent_wins / recent_trades if recent_trades > 0 else 0,
                'recent_avg_return': recent_avg_return or 0,
                'recent_weight': recent_weight
            }
        
        return None
    
    def calculate_bayesian_multiplier(self, context_type: str, context_value: int) -> tuple[float, dict]:
        """Calculate enhanced Bayesian position sizing multiplier with recent performance weighting"""
        stats = self.get_context_stats(context_type, context_value)
        
        if stats is None:
            # No historical data - use conservative prior
            return 1.0, {
                "method": "insufficient_data", 
                "expected_p": 0.5, 
                "alpha": config.alpha_prior,
                "beta": config.beta_prior,
                "total_trades": 0,
                "confidence": "low"
            }
        
        # Use recent performance if available, otherwise use all-time
        if stats['recent_trades'] >= 3:
            # Weight recent performance more heavily
            recent_weight = stats['recent_weight']
            all_time_p = stats['expected_p']
            recent_p = stats['recent_win_rate']
            
            # Blend recent and all-time performance
            expected_p = (recent_weight * recent_p) + ((1 - recent_weight) * all_time_p)
            confidence = "high" if stats['recent_trades'] >= 10 else "medium"
        else:
            expected_p = stats['expected_p']
            confidence = "medium" if stats['total_trades'] >= 20 else "low"
        
        # V6 Bayesian parameters from config
        if expected_p > 0.5:
            raw_multiplier = 1.0 + (expected_p - 0.5) * config.bayesian_scaling_factor
            position_multiplier = min(raw_multiplier, config.bayesian_max_multiplier)
        else:
            # Conservative sizing for below-50% win rate contexts
            position_multiplier = 1.0
            raw_multiplier = 1.0
        
        # Adjust multiplier based on confidence
        if confidence == "low":
            position_multiplier = min(position_multiplier, 1.5)  # Cap at 1.5x for low confidence
        elif confidence == "medium":
            position_multiplier = min(position_multiplier, 2.0)  # Cap at 2.0x for medium confidence
        
        # Diagnostic information
        diagnostics = {
            "method": "enhanced_bayesian",
            "expected_p": expected_p,
            "all_time_p": stats['expected_p'],
            "recent_p": stats['recent_win_rate'] if stats['recent_trades'] > 0 else None,
            "recent_weight": stats['recent_weight'],
            "alpha": stats['alpha_post'],
            "beta": stats['beta_post'],
            "total_trades": stats['total_trades'],
            "recent_trades": stats['recent_trades'],
            "raw_multiplier": raw_multiplier,
            "capped_multiplier": position_multiplier,
            "confidence": confidence,
            "confidence_interval": stats['confidence_interval']
        }
        
        return position_multiplier, diagnostics
    
    def record_trade_result(self, context_type: str, context_value: int, 
                          entry_price: float, exit_price: float, 
                          volume_ratio: float, signal_strength: float):
        """Record trade result for Bayesian learning"""
        return_pct = (exit_price - entry_price) / entry_price
        win = 1 if return_pct > 0 else 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO context_performance 
            (context_type, context_value, trade_timestamp, entry_price, exit_price, 
             return_pct, win, volume_ratio, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (context_type, context_value, datetime.now(), entry_price, exit_price,
              return_pct, win, volume_ratio, signal_strength))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded trade: {context_type}[{context_value}] = {return_pct:.4f} ({'WIN' if win else 'LOSS'})")
    
    def get_bayesian_summary(self, context_type: str = "modal_bin") -> Dict:
        """Get summary of Bayesian statistics across all contexts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get summary stats for all contexts
        cursor.execute('''
            SELECT context_value,
                   COUNT(*) as total_trades,
                   SUM(win) as wins,
                   AVG(return_pct) as avg_return,
                   AVG(CASE WHEN trade_timestamp >= datetime('now', '-7 days') THEN win ELSE NULL END) as recent_win_rate
            FROM context_performance 
            WHERE context_type = ?
            GROUP BY context_value
            ORDER BY context_value
        ''', (context_type,))
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {}
        for row in results:
            context_value, total_trades, wins, avg_return, recent_win_rate = row
            losses = total_trades - wins
            
            # Calculate Bayesian parameters
            alpha_post = config.alpha_prior + wins
            beta_post = config.beta_prior + losses
            expected_p = alpha_post / (alpha_post + beta_post)
            
            summary[context_value] = {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / total_trades if total_trades > 0 else 0,
                'expected_p': expected_p,
                'avg_return': avg_return or 0,
                'recent_win_rate': recent_win_rate or 0,
                'alpha_post': alpha_post,
                'beta_post': beta_post
            }
        
        return summary
    
    def print_bayesian_summary(self, context_type: str = "modal_bin"):
        """Print a summary of Bayesian statistics for monitoring"""
        summary = self.get_bayesian_summary(context_type)
        
        if not summary:
            print("📊 No Bayesian data available yet")
            return
        
        print("\n" + "="*60)
        print("📊 V6 BAYESIAN LEARNING SUMMARY")
        print("="*60)
        print(f"{'Bin':<4} {'Trades':<7} {'Win%':<6} {'Expected P':<10} {'Avg Return':<12} {'Recent%':<8}")
        print("-"*60)
        
        for context_value in sorted(summary.keys()):
            stats = summary[context_value]
            print(f"{context_value:<4} {stats['total_trades']:<7} "
                  f"{stats['win_rate']*100:<5.1f}% {stats['expected_p']:<9.3f} "
                  f"{stats['avg_return']*100:<11.2f}% {stats['recent_win_rate']*100:<7.1f}%")
        
        print("="*60)

class RealTimeTradingSystem:
    """Main real-time trading system"""
    
    def __init__(self):
        # Load configuration
        load_config_from_file()
        
        self.bayesian_manager = BayesianStatsManager(config.database_path)
        self.databento_connector = DatabentoConnector(config.databento_api_key)
        self.current_data = pd.DataFrame()
        self.last_cluster_time = None
        self.active_recommendation = None
        self.contract_selector = None
        self.is_market_open = False
        self.current_contract = None
        
        # V6 Strategy Parameters (from config)
        self.VOLUME_THRESHOLD = config.volume_threshold
        self.MIN_SIGNAL_STRENGTH = config.min_signal_strength
        self.RETENTION_MINUTES = config.retention_minutes
        self.PROFIT_TARGET_RATIO = config.profit_target_ratio
        
        # Market hours (EST/EDT)
        self.market_open_time = config.market_open_time
        self.market_close_time = config.market_close_time
        self.est_tz = pytz.timezone('US/Eastern')
        
        logger.info("V6 Real-Time Trading System initialized")
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:30 AM - 4:00 PM EST)"""
        now_est = datetime.now(self.est_tz)
        current_time = now_est.time()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now_est.weekday() >= 5:  # Saturday or Sunday
            return False
            
        # Check if within market hours
        is_open = self.market_open_time <= current_time <= self.market_close_time
        
        if is_open != self.is_market_open:
            self.is_market_open = is_open
            status = "OPEN" if is_open else "CLOSED"
            logger.info(f"🏪 Market is now {status} (EST: {now_est.strftime('%H:%M:%S')})")
        
        return is_open
    
    async def connect_to_databento(self):
        """Connect to Databento API for real-time data"""
        logger.info("Connecting to Databento API...")
        
        # Initialize the connector
        initialized = await self.databento_connector.initialize()
        if not initialized:
            logger.error("❌ Failed to initialize Databento connector")
            return False
        
        # Set up data callback
        self.databento_connector.set_data_callback(self.on_market_data_received)
        
        # Start live data stream
        logger.info("📡 Starting live data stream...")
        try:
            # Use ES futures contract for live streaming
            contract = "ES.FUT"  # Generic ES futures - should auto-select front month
            self.databento_connector.start_live_stream(contract)
            logger.info("✅ Live data stream started")
        except Exception as e:
            logger.error(f"❌ Failed to start live stream: {e}")
            return False
        
        logger.info("✅ Connected to Databento API with live streaming")
        return True
    
    def on_market_data_received(self, data: pd.DataFrame):
        """Callback function for receiving real-time market data"""
        try:
            # Validate data
            if data.empty or len(data) == 0:
                logger.warning("⚠️  Received empty market data")
                return
            
            # Append new data to current buffer
            self.current_data = pd.concat([self.current_data, data]).tail(100)
            
            # Always save current market data for ticker display
            self.save_current_market_data(data)
            
            # Check for volume clusters only during market hours
            if self.is_market_hours():
                cluster = self.detect_volume_cluster(self.current_data)
                
                if cluster and (self.last_cluster_time is None or 
                               (cluster.timestamp - self.last_cluster_time).total_seconds() > 
                               config.cluster_detection_cooldown_minutes * 60):
                    
                    # Generate recommendation
                    recommendation = self.generate_trading_recommendation(cluster)
                    
                    # Print recommendation
                    self.print_recommendation(recommendation)
                    
                    # Save recommendation
                    self.save_recommendation(recommendation, cluster)
                    
                    # Update last cluster time
                    self.last_cluster_time = cluster.timestamp
                    self.active_recommendation = recommendation
                    
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
            # Could implement retry logic here if needed
    
    async def reconnect_with_backoff(self, max_retries: int = 5):
        """Reconnect to data feed with exponential backoff"""
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Reconnection attempt {attempt + 1}/{max_retries}")
                
                # Wait with exponential backoff
                wait_time = min(2 ** attempt, 60)  # Max 60 seconds
                await asyncio.sleep(wait_time)
                
                # Try to reconnect
                connected = await self.connect_to_databento()
                if connected:
                    logger.info("✅ Successfully reconnected to data feed")
                    return True
                    
            except Exception as e:
                logger.error(f"❌ Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("❌ Failed to reconnect after all attempts")
        return False
    
    def get_modal_bin_context(self, modal_position: float) -> int:
        """Calculate modal bin context (V6 method)"""
        return min(int(modal_position * 10), 9)
    
    def detect_volume_cluster(self, recent_data: pd.DataFrame) -> Optional[VolumeCluster]:
        """Detect volume clusters in real-time data"""
        if len(recent_data) < 10:
            return None
        
        # Calculate volume ratio
        current_volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        if volume_ratio < self.VOLUME_THRESHOLD:
            return None
        
        # Calculate modal price (simplified)
        recent_prices = recent_data['close'].tail(5)
        modal_price = recent_prices.mode().iloc[0] if len(recent_prices.mode()) > 0 else recent_prices.mean()
        
        # Calculate signal strength (simplified)
        price_range = recent_data['high'].tail(10).max() - recent_data['low'].tail(10).min()
        signal_strength = min(volume_ratio / 10.0, 1.0)  # Normalize to 0-1
        
        if signal_strength < self.MIN_SIGNAL_STRENGTH:
            return None
        
        # Determine direction based on price action
        current_price = recent_data['close'].iloc[-1]
        direction = "long" if current_price > modal_price else "short"
        
        return VolumeCluster(
            timestamp=recent_data.index[-1],
            volume_ratio=volume_ratio,
            modal_price=modal_price,
            signal_strength=signal_strength,
            direction=direction,
            entry_price=current_price,
            volume_rank=1  # Simplified
        )
    
    def generate_trading_recommendation(self, cluster: VolumeCluster) -> TradingRecommendation:
        """Generate trading recommendation based on V6 Bayesian strategy"""
        
        # Calculate modal bin context (V6 method)
        modal_position = abs(cluster.entry_price - cluster.modal_price) / cluster.entry_price
        context_value = self.get_modal_bin_context(modal_position)
        
        # Get Bayesian multiplier with diagnostics
        bayesian_multiplier, diagnostics = self.bayesian_manager.calculate_bayesian_multiplier("modal_bin", context_value)
        
        # Calculate position size using V6 logic
        base_quantity = config.base_position_size
        quantity = max(1, min(config.max_position_size, int(base_quantity * bayesian_multiplier)))
        
        # Calculate volatility for risk management (simplified - should use real calculation)
        volatility = 0.01  # 1% volatility per minute
        
        # Calculate stop loss and profit target (V6 method)
        if config.use_profit_targets:
            stop_distance = 1.0 * volatility * cluster.entry_price  # Tighter stops
            min_stop = 0.005 * cluster.entry_price
            stop_distance = max(stop_distance, min_stop)
            profit_distance = stop_distance * config.profit_target_ratio
        else:
            stop_distance = 1.5 * volatility * cluster.entry_price
            profit_distance = stop_distance * config.profit_target_ratio
        
        if cluster.direction == "long":
            stop_loss = cluster.entry_price - stop_distance
            profit_target = cluster.entry_price + profit_distance
            action = "BUY"
        else:
            stop_loss = cluster.entry_price + stop_distance
            profit_target = cluster.entry_price - profit_distance
            action = "SHORT"
        
        # Get Bayesian confidence
        confidence = diagnostics.get('expected_p', 0.5)
        
        # Determine order type based on confidence and config
        if config.use_market_orders_on_high_confidence and confidence > config.high_confidence_threshold:
            order_type = "MARKET"
        else:
            order_type = config.default_order_type
        
        # Generate detailed reasoning with V6 diagnostics
        reasoning = f"V6 Bayesian: {cluster.volume_ratio:.1f}x volume, " \
                   f"signal {cluster.signal_strength:.3f}, " \
                   f"modal_bin[{context_value}] multiplier {bayesian_multiplier:.2f}x " \
                   f"(p={confidence:.3f}, trades={diagnostics.get('total_trades', 0)})"
        
        return TradingRecommendation(
            timestamp=cluster.timestamp,
            contract=self.current_contract or "ES JUN25",
            action=action,
            quantity=quantity,
            order_type=order_type,
            price=cluster.entry_price,
            validity=config.default_validity,
            confidence=confidence,
            signal_strength=cluster.signal_strength,
            bayesian_multiplier=bayesian_multiplier,
            stop_loss=stop_loss,
            profit_target=profit_target,
            reasoning=reasoning
        )
    
    def print_recommendation(self, rec: TradingRecommendation):
        """Print trading recommendation in a clear format with audio alert"""
        
        # Play audio alert based on confidence level
        if rec.confidence >= 0.80:
            play_alert_sound("urgent")  # High confidence = urgent sound
            alert_icon = "🔊🚨"
        else:
            play_alert_sound("signal")  # Normal signal sound
            alert_icon = "🔔"
        
        print("\n" + "="*60)
        print(f"🚨 {alert_icon} V6 BAYESIAN TRADING RECOMMENDATION {alert_icon}")
        print("="*60)
        print(f"⏰ Time: {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Contract: {rec.contract}")
        print(f"📈 Action: {rec.action}")
        print(f"📦 Quantity: {rec.quantity} contracts")
        print(f"💰 Price: ${rec.price:.2f}")
        print(f"📋 Order Type: {rec.order_type}")
        print(f"⏳ Validity: {rec.validity}")
        print()
        print("📊 BAYESIAN ANALYSIS:")
        print(f"   Confidence: {rec.confidence:.1%}")
        print(f"   Signal Strength: {rec.signal_strength:.3f}")
        print(f"   Position Multiplier: {rec.bayesian_multiplier:.2f}x")
        print()
        print("🎯 RISK MANAGEMENT:")
        print(f"   Stop Loss: ${rec.stop_loss:.2f}")
        print(f"   Profit Target: ${rec.profit_target:.2f}")
        print(f"   Risk/Reward: 1:{(rec.profit_target - rec.price) / (rec.price - rec.stop_loss):.2f}")
        print()
        print(f"💡 Reasoning: {rec.reasoning}")
        print()
        print("⚡ ACTION REQUIRED: You have 30 seconds for optimal execution!")
        print("🔊 Audio alert played - check your trading platform!")
        print("="*60)
    
    async def run_real_time_strategy(self):
        """Main real-time strategy loop"""
        logger.info("🚀 Starting V6 Bayesian real-time strategy")
        
        # Connect to data feed
        connected = await self.connect_to_databento()
        if not connected:
            logger.error("Failed to connect to data feed")
            return
        
        # Select the current contract (for now, use the first available)
        self.current_contract = config.available_contracts[0]
        logger.info(f"📊 Trading contract: {self.current_contract}")
        
        # Start live data stream with error handling
        stream_task = None
        try:
            logger.info("📡 Starting live data stream...")
            stream_task = asyncio.create_task(
                self.databento_connector.start_live_stream(self.current_contract)
            )
            await stream_task
            
        except Exception as e:
            logger.error(f"❌ Failed to start live stream: {e}")
            
            # Try to reconnect
            reconnected = await self.reconnect_with_backoff()
            if not reconnected:
                logger.info("🔄 Falling back to simulation mode...")
                
                # Fallback to simulation if live stream fails
                while True:
                    try:
                        await asyncio.sleep(config.data_update_interval_seconds)
                        
                        # Check market hours
                        if not self.is_market_hours():
                            continue
                        
                        # Simulate market data for testing
                        current_time = datetime.now()
                        simulated_data = pd.DataFrame({
                            'timestamp': [current_time],
                            'open': [6010.0],
                            'high': [6012.0],
                            'low': [6008.0],
                            'close': [6011.0],
                            'volume': [15000]  # Simulate high volume for testing
                        })
                        simulated_data.set_index('timestamp', inplace=True)
                        
                        # Process the simulated data
                        self.on_market_data_received(simulated_data)
                        
                    except Exception as e:
                        logger.error(f"Error in simulation loop: {e}")
                        await asyncio.sleep(10)  # Wait before retrying
        finally:
            # Clean up stream task
            if stream_task and not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
    
    def save_recommendation(self, rec: TradingRecommendation, cluster: VolumeCluster):
        """Save recommendation to JSON file for external systems"""
        rec_dict = {
            'timestamp': rec.timestamp.isoformat(),
            'contract': rec.contract,
            'action': rec.action,
            'quantity': rec.quantity,
            'order_type': rec.order_type,
            'price': rec.price,
            'validity': rec.validity,
            'confidence': rec.confidence,
            'signal_strength': rec.signal_strength,
            'bayesian_multiplier': rec.bayesian_multiplier,
            'stop_loss': rec.stop_loss,
            'profit_target': rec.profit_target,
            'reasoning': rec.reasoning,
            'volume_ratio': cluster.volume_ratio,
            'modal_price': cluster.modal_price,
            'direction': cluster.direction,
            'entry_price': cluster.entry_price,
            'volume_rank': cluster.volume_rank
        }
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(config.latest_recommendation_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.recommendations_log_path), exist_ok=True)
        
        # Save to latest recommendation file
        with open(config.latest_recommendation_path, 'w') as f:
            json.dump(rec_dict, f, indent=2)
        
        # Append to recommendations log
        with open(config.recommendations_log_path, 'a') as f:
            f.write(json.dumps(rec_dict) + '\n')
        
        logger.info(f"💾 Recommendation saved: {rec.action} {rec.quantity} {rec.contract} @ ${rec.price}")
    
    def save_current_market_data(self, data: pd.DataFrame):
        """Save current market data for ticker display"""
        try:
            if data.empty:
                return
                
            # Get the latest data point
            latest = data.iloc[-1]
            
            # Create market data dict for ticker
            market_data = {
                'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name),
                'contract': 'ES.FUT',  # Generic ES futures
                'price': float(latest['close']),
                'volume': int(latest['volume']),
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': float(latest['close']),
                'signal_strength': 0.0,  # No signal when no cluster
                'volume_ratio': 1.0,     # No volume ratio when no cluster
                'bayesian_multiplier': 1.0,  # No multiplier when no cluster
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Live market data - no trading signal detected'
            }
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(config.latest_recommendation_path), exist_ok=True)
            
            # Save to latest recommendation file for ticker
            with open(config.latest_recommendation_path, 'w') as f:
                json.dump(market_data, f, indent=2)
                
            logger.debug(f"📊 Market data saved: ES.FUT @ ${market_data['price']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error saving market data: {e}")

async def main():
    """Main entry point"""
    # Validate configuration
    errors = config.validate()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   {error}")
        print("\n💡 Please fix configuration issues before running the system")
        return
    
    # Print configuration summary
    config.print_config()
    
    # Initialize and run the trading system
    system = RealTimeTradingSystem()
    
    try:
        await system.run_real_time_strategy()
    except KeyboardInterrupt:
        logger.info("⏹️  System stopped by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        raise
    finally:
        # Clean up connections
        if system.databento_connector:
            await system.databento_connector.stop_stream()

if __name__ == "__main__":
    print("🚀 V6 BAYESIAN REAL-TIME TRADING SYSTEM")
    print("="*50)
    print("Initializing...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"Fatal system error: {e}") 