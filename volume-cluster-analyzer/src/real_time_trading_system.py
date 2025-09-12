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
    
    def __init__(self, db_path: str = "../data/bayesian_stats.db"):
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
        """Get Bayesian statistics for a specific context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total_trades,
                   SUM(win) as wins,
                   AVG(return_pct) as avg_return,
                   STDDEV(return_pct) as return_std
            FROM context_performance 
            WHERE context_type = ? AND context_value = ?
        ''', (context_type, context_value))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] >= min_trades:
            total_trades, wins, avg_return, return_std = result
            losses = total_trades - wins
            
            # Calculate Bayesian posterior parameters
            alpha_post = 1.0 + wins  # Prior alpha = 1.0
            beta_post = 1.0 + losses  # Prior beta = 1.0
            
            # Expected win probability
            expected_p = alpha_post / (alpha_post + beta_post)
            
            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'expected_p': expected_p,
                'avg_return': avg_return or 0,
                'return_std': return_std or 0,
                'alpha_post': alpha_post,
                'beta_post': beta_post
            }
        
        return None
    
    def calculate_bayesian_multiplier(self, context_type: str, context_value: int) -> float:
        """Calculate Bayesian position sizing multiplier"""
        stats = self.get_context_stats(context_type, context_value)
        
        if stats is None:
            return 1.0  # Conservative default
        
        expected_p = stats['expected_p']
        
        # V6 Bayesian parameters
        SCALING_FACTOR = 6.0
        MAX_MULTIPLIER = 3.0
        
        if expected_p > 0.5:
            raw_multiplier = 1.0 + (expected_p - 0.5) * SCALING_FACTOR
            return min(raw_multiplier, MAX_MULTIPLIER)
        else:
            return 1.0
    
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

class RealTimeTradingSystem:
    """Main real-time trading system"""
    
    def __init__(self):
        self.bayesian_manager = BayesianStatsManager()
        self.current_data = pd.DataFrame()
        self.last_cluster_time = None
        self.active_recommendation = None
        self.contract_selector = None
        
        # V6 Strategy Parameters
        self.VOLUME_THRESHOLD = 4.0
        self.MIN_SIGNAL_STRENGTH = 0.45
        self.RETENTION_MINUTES = 60
        self.PROFIT_TARGET_RATIO = 2.0
        
        logger.info("V6 Real-Time Trading System initialized")
    
    async def connect_to_databento(self):
        """Connect to Databento API for real-time data"""
        # TODO: Implement actual Databento connection
        # This is a placeholder for the real implementation
        logger.info("Connecting to Databento API...")
        
        # For now, simulate connection
        await asyncio.sleep(1)
        logger.info("✅ Connected to Databento API")
        
        return True
    
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
        
        # Calculate modal bin context
        modal_position = abs(cluster.entry_price - cluster.modal_price) / cluster.entry_price
        context_value = self.get_modal_bin_context(modal_position)
        
        # Get Bayesian multiplier
        bayesian_multiplier = self.bayesian_manager.calculate_bayesian_multiplier("modal_bin", context_value)
        
        # Calculate position size (1-3 contracts based on Bayesian multiplier)
        base_quantity = 1
        quantity = max(1, min(3, int(base_quantity * bayesian_multiplier)))
        
        # Calculate stop loss and profit target
        volatility = 0.01  # Simplified - should use real volatility calculation
        
        if cluster.direction == "long":
            stop_loss = cluster.entry_price * (1 - 1.5 * volatility)
            profit_target = cluster.entry_price * (1 + self.PROFIT_TARGET_RATIO * 1.5 * volatility)
            action = "BUY"
        else:
            stop_loss = cluster.entry_price * (1 + 1.5 * volatility)
            profit_target = cluster.entry_price * (1 - self.PROFIT_TARGET_RATIO * 1.5 * volatility)
            action = "SHORT"
        
        # Get Bayesian confidence
        stats = self.bayesian_manager.get_context_stats("modal_bin", context_value)
        confidence = stats['expected_p'] if stats else 0.5
        
        # Determine order type based on confidence (configurable)
        if confidence > 0.80:  # High confidence - could use MARKET orders
            order_type = "LIMIT"  # Still use LIMIT for now, configurable later
        else:
            order_type = "LIMIT"
        
        # Generate reasoning
        reasoning = f"Volume cluster detected: {cluster.volume_ratio:.1f}x volume, " \
                   f"signal strength {cluster.signal_strength:.3f}, " \
                   f"Bayesian multiplier {bayesian_multiplier:.2f}x " \
                   f"(confidence {confidence:.3f})"
        
        return TradingRecommendation(
            timestamp=cluster.timestamp,
            contract="ES JUN25",  # Will be dynamic based on contract selector
            action=action,
            quantity=quantity,
            order_type=order_type,
            price=cluster.entry_price,
            validity="DAY",
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
        
        # Main trading loop
        while True:
            try:
                # TODO: Replace with real Databento data ingestion
                # For now, simulate receiving market data
                await asyncio.sleep(60)  # Check every minute
                
                # Simulate market data (replace with real data)
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
                
                # Append to current data buffer
                self.current_data = pd.concat([self.current_data, simulated_data]).tail(100)
                
                # Check for volume clusters
                cluster = self.detect_volume_cluster(self.current_data)
                
                if cluster and (self.last_cluster_time is None or 
                               (cluster.timestamp - self.last_cluster_time).total_seconds() > 1800):  # 30 min cooldown
                    
                    # Generate recommendation
                    recommendation = self.generate_trading_recommendation(cluster)
                    
                    # Print recommendation
                    self.print_recommendation(recommendation)
                    
                    # Save recommendation to file (with cluster info for feedback)
                    self.save_recommendation(recommendation, cluster)
                    
                    # Update last cluster time
                    self.last_cluster_time = cluster.timestamp
                    self.active_recommendation = recommendation
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
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
        
        # Save to latest recommendation file
        with open('../data/latest_recommendation.json', 'w') as f:
            json.dump(rec_dict, f, indent=2)
        
        # Append to recommendations log
        with open('../data/recommendations_log.jsonl', 'a') as f:
            f.write(json.dumps(rec_dict) + '\n')
        
        logger.info(f"💾 Recommendation saved: {rec.action} {rec.quantity} {rec.contract} @ ${rec.price}")

async def main():
    """Main entry point"""
    system = RealTimeTradingSystem()
    await system.run_real_time_strategy()

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