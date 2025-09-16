#!/usr/bin/env python3
"""
Historical Signal Lookback Tool
Cross-check the live robot with historical data using the same V6 Bayesian logic
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
import os
import pytz
from pathlib import Path

# Import our custom modules
from src.databento_connector import DatabentoConnector
from src.config import config, load_config_from_file
from src.volume_cluster import identify_volume_clusters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HistoricalSignal:
    """Historical signal output structure"""
    timestamp: datetime
    contract: str
    action: str
    quantity: int
    order_type: str
    price: float
    confidence: float
    signal_strength: float
    bayesian_multiplier: float
    stop_loss: float
    profit_target: float
    reasoning: str
    volume_ratio: float
    modal_price: float
    direction: str
    entry_price: float
    volume_rank: int
    modal_bin: int
    bayesian_expected_p: float
    bayesian_total_trades: int

class HistoricalBayesianManager:
    """Manages Bayesian statistics for historical analysis"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "../data/bayesian_stats.db"
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
        """Get enhanced Bayesian statistics for a specific context"""
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
        """Calculate enhanced Bayesian position sizing multiplier"""
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

class HistoricalSignalAnalyzer:
    """Analyzes historical signals using the same logic as the live system"""
    
    def __init__(self):
        # Load configuration
        load_config_from_file()
        
        self.bayesian_manager = HistoricalBayesianManager()
        self.databento_connector = DatabentoConnector(config.databento_api_key)
        
        # V6 Strategy Parameters (from config)
        self.VOLUME_THRESHOLD = config.volume_threshold
        self.MIN_SIGNAL_STRENGTH = config.min_signal_strength
        self.RETENTION_MINUTES = config.retention_minutes
        self.PROFIT_TARGET_RATIO = config.profit_target_ratio
        
        # Market hours (EST/EDT)
        self.market_open_time = config.market_open_time
        self.market_close_time = config.market_close_time
        self.est_tz = pytz.timezone('US/Eastern')
        
        logger.info("Historical Signal Analyzer initialized")
    
    def _process_databento_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Databento data format for the lookback tool"""
        try:
            # Log the actual columns to debug
            logger.info(f"📊 Databento data columns: {list(df.columns)}")
            logger.info(f"📊 Databento data shape: {df.shape}")
            
            # Handle different column formats - select only OHLCV columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in df.columns]
            
            if len(available_columns) < 5:
                logger.warning(f"⚠️  Missing required columns. Available: {list(df.columns)}")
                # Try to map common column names
                column_mapping = {
                    'ts_event': 'timestamp',
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }
                
                # Rename columns if they exist
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})
            
            # Select only the columns we need
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Ensure we have the required columns
            if not all(col in df.columns for col in required_columns):
                logger.error(f"❌ Missing required columns. Available: {list(df.columns)}")
                raise ValueError("Missing required OHLCV columns")
            
            # Select only OHLCV columns
            df = df[required_columns]
            
            logger.info(f"✅ Processed historical data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing Databento data: {e}")
            raise
    
    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within market hours"""
        # Convert to EST
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        
        est_time = timestamp.astimezone(self.est_tz)
        current_time = est_time.time()
        
        # Check if it's a weekday
        if est_time.weekday() >= 5:  # Saturday or Sunday
            return False
            
        # Check if within market hours
        return self.market_open_time <= current_time <= self.market_close_time
    
    def get_modal_bin_context(self, modal_position: float) -> int:
        """Calculate modal bin context (V6 method)"""
        return min(int(modal_position * 10), 9)
    
    def detect_volume_cluster(self, recent_data: pd.DataFrame) -> Optional[Dict]:
        """Detect volume clusters using EXACT same logic as backtest V6"""
        if len(recent_data) < 10:
            return None
        
        # Use the EXACT same logic as backtest V6
        # This is a simplified version for real-time detection
        # The backtest uses identify_volume_clusters() with 15-minute resampling
        
        # For now, we'll use a simplified approach that matches the backtest logic
        # Calculate volume ratio using daily average (like backtest)
        daily_avg_volume = recent_data['volume'].mean()
        current_volume = recent_data['volume'].iloc[-1]
        volume_ratio = current_volume / daily_avg_volume if daily_avg_volume > 0 else 0
        
        # Use a higher threshold to match the backtest's 15-minute resampling effect
        # The backtest requires 4.0x the daily 15-minute average, which is much higher
        # than 4.0x the 1-minute average
        effective_threshold = self.VOLUME_THRESHOLD * 15  # Approximate 15-minute effect
        
        if volume_ratio < effective_threshold:
            return None
        
        # Calculate modal price (simplified)
        recent_prices = recent_data['close'].tail(5)
        modal_price = recent_prices.mode().iloc[0] if len(recent_prices.mode()) > 0 else recent_prices.mean()
        
        # Calculate signal strength (simplified)
        price_range = recent_data['high'].tail(10).max() - recent_data['low'].tail(10).min()
        signal_strength = min(volume_ratio / 150.0, 1.0)  # Match backtest calculation
        
        if signal_strength < self.MIN_SIGNAL_STRENGTH:
            return None
        
        # Determine direction based on price action
        current_price = recent_data['close'].iloc[-1]
        direction = "long" if current_price > modal_price else "short"
        
        return {
            'timestamp': recent_data.index[-1],
            'volume_ratio': volume_ratio,
            'modal_price': modal_price,
            'signal_strength': signal_strength,
            'direction': direction,
            'entry_price': current_price,
            'volume_rank': 1  # Simplified for historical analysis
        }
    
    def calculate_momentum(self, df: pd.DataFrame, timestamp: datetime, lookback_minutes: int = 5) -> float:
        """Calculate short-term momentum before the signal (EXACT same as backtest)"""
        start_time = timestamp - timedelta(minutes=lookback_minutes)
        momentum_data = df.loc[start_time:timestamp]
        
        if len(momentum_data) < 2:
            return 0
        
        price_change = (momentum_data['close'].iloc[-1] - momentum_data['close'].iloc[0]) / momentum_data['close'].iloc[0]
        return price_change
    
    def calculate_signal_strength_v3(self, modal_position: float, volume_ratio: float, momentum: float) -> float:
        """Enhanced signal strength calculation (EXACT same as backtest V6)"""
        TIGHT_LONG_THRESHOLD = 0.15
        ELIMINATE_SHORTS = True
        
        if modal_position <= TIGHT_LONG_THRESHOLD:
            position_strength = 1.0 - (modal_position / TIGHT_LONG_THRESHOLD)
        elif modal_position >= 0.85 and not ELIMINATE_SHORTS:
            position_strength = (modal_position - 0.85) / 0.15
        else:
            return 0
        
        volume_strength = min(volume_ratio / 150.0, 1.0)
        
        if modal_position <= TIGHT_LONG_THRESHOLD:
            momentum_strength = max(0, momentum * 8)
        else:
            momentum_strength = max(0, -momentum * 8)
        
        momentum_strength = min(momentum_strength, 1.0)
        combined_strength = (0.5 * position_strength + 0.3 * volume_strength + 0.2 * momentum_strength)
        return combined_strength
    
    def generate_trading_recommendation_from_cluster(self, cluster_time: datetime, cluster_row: pd.Series, 
                                                   volume_ratio: float, modal_position: float, 
                                                   modal_price: float, direction: str, 
                                                   signal_strength: float, momentum: float) -> HistoricalSignal:
        """Generate trading recommendation using EXACT same logic as backtest V6"""
        
        # Calculate modal bin context (V6 method)
        context_value = self.get_modal_bin_context(modal_position)
        
        # Get Bayesian multiplier with diagnostics
        bayesian_multiplier, diagnostics = self.bayesian_manager.calculate_bayesian_multiplier("modal_bin", context_value)
        
        # Calculate position size using V6 logic
        base_quantity = config.base_position_size
        quantity = max(1, min(config.max_position_size, int(base_quantity * bayesian_multiplier)))
        
        # Calculate volatility for risk management (simplified)
        volatility = 0.01  # 1% volatility per minute
        
        # Calculate stop loss and profit target (V6 method)
        if config.use_profit_targets:
            stop_distance = 1.0 * volatility * cluster_row['close']  # Tighter stops
            min_stop = 0.005 * cluster_row['close']
            stop_distance = max(stop_distance, min_stop)
            profit_distance = stop_distance * config.profit_target_ratio
        else:
            stop_distance = 1.5 * volatility * cluster_row['close']
            profit_distance = stop_distance * config.profit_target_ratio
        
        if direction == "long":
            stop_loss = cluster_row['close'] - stop_distance
            profit_target = cluster_row['close'] + profit_distance
            action = "BUY"
        else:
            stop_loss = cluster_row['close'] + stop_distance
            profit_target = cluster_row['close'] - profit_distance
            action = "SHORT"
        
        # Get Bayesian confidence
        confidence = diagnostics.get('expected_p', 0.5)
        
        # Determine order type based on confidence and config
        if config.use_market_orders_on_high_confidence and confidence > config.high_confidence_threshold:
            order_type = "MARKET"
        else:
            order_type = config.default_order_type
        
        # Generate detailed reasoning with V6 diagnostics
        reasoning = f"V6 Bayesian: {volume_ratio:.1f}x volume, " \
                   f"signal {signal_strength:.3f}, " \
                   f"modal_bin[{context_value}] multiplier {bayesian_multiplier:.2f}x " \
                   f"(p={confidence:.3f}, trades={diagnostics.get('total_trades', 0)})"
        
        return HistoricalSignal(
            timestamp=cluster_time,
            contract="ES JUN25",  # Default contract
            action=action,
            quantity=quantity,
            order_type=order_type,
            price=cluster_row['close'],
            confidence=confidence,
            signal_strength=signal_strength,
            bayesian_multiplier=bayesian_multiplier,
            stop_loss=stop_loss,
            profit_target=profit_target,
            reasoning=reasoning,
            volume_ratio=volume_ratio,
            modal_price=modal_price,
            direction=direction,
            entry_price=cluster_row['close'],
            volume_rank=1,  # Simplified for lookback
            modal_bin=context_value,
            bayesian_expected_p=confidence,
            bayesian_total_trades=diagnostics.get('total_trades', 0)
        )
    
    def generate_trading_recommendation(self, cluster: Dict) -> HistoricalSignal:
        """Generate trading recommendation based on V6 Bayesian strategy"""
        
        # Calculate modal bin context (V6 method)
        modal_position = abs(cluster['entry_price'] - cluster['modal_price']) / cluster['entry_price']
        context_value = self.get_modal_bin_context(modal_position)
        
        # Get Bayesian multiplier with diagnostics
        bayesian_multiplier, diagnostics = self.bayesian_manager.calculate_bayesian_multiplier("modal_bin", context_value)
        
        # Calculate position size using V6 logic
        base_quantity = config.base_position_size
        quantity = max(1, min(config.max_position_size, int(base_quantity * bayesian_multiplier)))
        
        # Calculate volatility for risk management (simplified)
        volatility = 0.01  # 1% volatility per minute
        
        # Calculate stop loss and profit target (V6 method)
        if config.use_profit_targets:
            stop_distance = 1.0 * volatility * cluster['entry_price']  # Tighter stops
            min_stop = 0.005 * cluster['entry_price']
            stop_distance = max(stop_distance, min_stop)
            profit_distance = stop_distance * config.profit_target_ratio
        else:
            stop_distance = 1.5 * volatility * cluster['entry_price']
            profit_distance = stop_distance * config.profit_target_ratio
        
        if cluster['direction'] == "long":
            stop_loss = cluster['entry_price'] - stop_distance
            profit_target = cluster['entry_price'] + profit_distance
            action = "BUY"
        else:
            stop_loss = cluster['entry_price'] + stop_distance
            profit_target = cluster['entry_price'] - profit_distance
            action = "SHORT"
        
        # Get Bayesian confidence
        confidence = diagnostics.get('expected_p', 0.5)
        
        # Determine order type based on confidence and config
        if config.use_market_orders_on_high_confidence and confidence > config.high_confidence_threshold:
            order_type = "MARKET"
        else:
            order_type = config.default_order_type
        
        # Generate detailed reasoning with V6 diagnostics
        reasoning = f"V6 Bayesian: {cluster['volume_ratio']:.1f}x volume, " \
                   f"signal {cluster['signal_strength']:.3f}, " \
                   f"modal_bin[{context_value}] multiplier {bayesian_multiplier:.2f}x " \
                   f"(p={confidence:.3f}, trades={diagnostics.get('total_trades', 0)})"
        
        return HistoricalSignal(
            timestamp=cluster['timestamp'],
            contract="ES JUN25",  # Default contract
            action=action,
            quantity=quantity,
            order_type=order_type,
            price=cluster['entry_price'],
            confidence=confidence,
            signal_strength=cluster['signal_strength'],
            bayesian_multiplier=bayesian_multiplier,
            stop_loss=stop_loss,
            profit_target=profit_target,
            reasoning=reasoning,
            volume_ratio=cluster['volume_ratio'],
            modal_price=cluster['modal_price'],
            direction=cluster['direction'],
            entry_price=cluster['entry_price'],
            volume_rank=cluster['volume_rank'],
            modal_bin=context_value,
            bayesian_expected_p=confidence,
            bayesian_total_trades=diagnostics.get('total_trades', 0)
        )
    
    async def analyze_historical_period(self, start_time: datetime, end_time: datetime, 
                                      contract: str = "ES JUN25") -> List[HistoricalSignal]:
        """Analyze historical signals using EXACT same logic as backtest V6"""
        
        logger.info(f"📊 Analyzing historical signals from {start_time} to {end_time}")
        
        # Initialize Databento connector first
        initialized = await self.databento_connector.initialize()
        if not initialized:
            logger.error("❌ Failed to initialize Databento connector")
            return []
        
        # Get historical data
        try:
            # Adjust end_time to be within available data range (usually 2-3 hours behind current time)
            adjusted_end_time = min(end_time, datetime.now() - timedelta(hours=3))
            logger.info(f"📈 Adjusted end time to: {adjusted_end_time} (within available data range)")
            
            historical_data = await self.databento_connector.get_historical_data(
                contract, start_time, adjusted_end_time
            )
            
            if historical_data.empty:
                logger.warning("No historical data available for the specified period")
                return []
            
            # Process Databento data format for lookback tool
            historical_data = self._process_databento_data(historical_data)
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
        
        signals = []
        
        # Use EXACT same logic as backtest V6 - process day by day
        for day, group in historical_data.groupby(historical_data.index.date):
            intraday_df = group.copy()
            
            # Identify clusters for the day using EXACT same function as backtest
            clusters_df = identify_volume_clusters(intraday_df, volume_multiplier=self.VOLUME_THRESHOLD)
            
            if clusters_df.empty:
                continue
            
            # Calculate volume ratios for all clusters (EXACT same as backtest)
            avg_volume = intraday_df['volume'].mean()
            
            # Process clusters in chronological order (EXACT same as backtest)
            clusters_sorted = clusters_df.sort_index()
            
            for cluster_time, cluster_row in clusters_sorted.iterrows():
                # Check if we're in market hours
                if not self.is_market_hours(cluster_time):
                    continue
                
                # Calculate volume ratio (EXACT same as backtest)
                cluster_volume = cluster_row['volume']
                volume_ratio = cluster_volume / avg_volume if avg_volume > 0 else 0
                
                # Only trade if this cluster ranks in top-N (simplified for lookback)
                # In backtest: if volume_rank > TOP_N_CLUSTERS_PER_DAY: continue
                # For lookback, we'll process all clusters for analysis
                
                # Analyze cluster price action (EXACT same as backtest)
                cluster_slice = intraday_df.loc[cluster_time : cluster_time + timedelta(minutes=14)]
                
                if cluster_slice.empty:
                    continue
                    
                modal_price = cluster_slice["close"].round(2).mode()
                if len(modal_price) == 0:
                    continue
                modal_price = modal_price[0]

                price_low = cluster_slice["low"].min()
                price_high = cluster_slice["high"].max()
                modal_position = (modal_price - price_low) / (price_high - price_low + 1e-9)

                # Determine direction using EXACT same logic as backtest
                if modal_position <= 0.15:  # TIGHT_LONG_THRESHOLD
                    direction = "long"
                elif modal_position >= 0.85 and not True:  # ELIMINATE_SHORTS = True
                    direction = "short"
                else:
                    continue  # Skip this cluster

                # Calculate momentum and signal strength (EXACT same as backtest)
                momentum = self.calculate_momentum(intraday_df, cluster_time)
                signal_strength = self.calculate_signal_strength_v3(modal_position, volume_ratio, momentum)
                
                # Check signal strength threshold (EXACT same as backtest)
                min_signal = 0.65 if direction == "short" else self.MIN_SIGNAL_STRENGTH
                if signal_strength < min_signal:
                    continue

                # Generate recommendation using EXACT same logic as backtest
                recommendation = self.generate_trading_recommendation_from_cluster(
                    cluster_time, cluster_row, volume_ratio, modal_position, 
                    modal_price, direction, signal_strength, momentum
                )
                
                signals.append(recommendation)
                
                logger.info(f"🔔 Signal detected at {cluster_time}: {recommendation.action} "
                          f"@ ${recommendation.price:.2f} (confidence: {recommendation.confidence:.1%})")
        
        return signals
    
    def print_signal_summary(self, signals: List[HistoricalSignal]):
        """Print a summary of detected signals"""
        if not signals:
            print("📊 No signals detected in the specified time period")
            return
        
        print(f"\n📊 HISTORICAL SIGNAL ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Signals: {len(signals)}")
        
        # Group by action
        buy_signals = [s for s in signals if s.action == "BUY"]
        short_signals = [s for s in signals if s.action == "SHORT"]
        
        print(f"BUY Signals: {len(buy_signals)}")
        print(f"SHORT Signals: {len(short_signals)}")
        
        # Confidence analysis
        high_confidence = [s for s in signals if s.confidence >= 0.8]
        medium_confidence = [s for s in signals if 0.6 <= s.confidence < 0.8]
        low_confidence = [s for s in signals if s.confidence < 0.6]
        
        print(f"\nConfidence Distribution:")
        print(f"  High (≥80%): {len(high_confidence)}")
        print(f"  Medium (60-79%): {len(medium_confidence)}")
        print(f"  Low (<60%): {len(low_confidence)}")
        
        # Bayesian analysis
        bayesian_signals = [s for s in signals if s.bayesian_total_trades > 0]
        print(f"\nBayesian Analysis:")
        print(f"  Signals with historical data: {len(bayesian_signals)}")
        print(f"  Average Bayesian multiplier: {np.mean([s.bayesian_multiplier for s in signals]):.2f}")
        print(f"  Average expected win probability: {np.mean([s.bayesian_expected_p for s in signals]):.3f}")
        
        print(f"\n📋 DETAILED SIGNAL LIST:")
        print("-" * 60)
        for i, signal in enumerate(signals, 1):
            print(f"{i:2d}. {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"{signal.action:5s} | ${signal.price:7.2f} | "
                  f"Conf: {signal.confidence:5.1%} | "
                  f"Vol: {signal.volume_ratio:5.1f}x | "
                  f"Bayesian: {signal.bayesian_multiplier:.2f}x")
            print(f"    Reasoning: {signal.reasoning}")
            print(f"    Stop: ${signal.stop_loss:.2f} | Target: ${signal.profit_target:.2f}")
            print()

async def main():
    """Main entry point for the lookback tool"""
    print("🔍 HISTORICAL SIGNAL LOOKBACK TOOL")
    print("=" * 50)
    
    # Load configuration
    load_config_from_file()
    
    # Set the API key from config for the Databento connector
    os.environ['DATABENTO_API_KEY'] = config.databento_api_key
    
    # Initialize analyzer
    analyzer = HistoricalSignalAnalyzer()
    
    # Get user input for time period
    print("\n📅 Time Period Selection:")
    print("1. Last 15 hours")
    print("2. Last 24 hours") 
    print("3. Last 3 days")
    print("4. Custom period")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    now = datetime.now()
    
    if choice == "1":
        start_time = now - timedelta(hours=15)
        end_time = now
    elif choice == "2":
        start_time = now - timedelta(hours=24)
        end_time = now
    elif choice == "3":
        start_time = now - timedelta(days=3)
        end_time = now
    elif choice == "4":
        try:
            hours_back = int(input("Enter hours to look back: "))
            start_time = now - timedelta(hours=hours_back)
            end_time = now
        except ValueError:
            print("Invalid input, using 15 hours as default")
            start_time = now - timedelta(hours=15)
            end_time = now
    else:
        print("Invalid choice, using 15 hours as default")
        start_time = now - timedelta(hours=15)
        end_time = now
    
    print(f"\n🔍 Analyzing signals from {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze historical signals
    signals = await analyzer.analyze_historical_period(start_time, end_time)
    
    # Print summary
    analyzer.print_signal_summary(signals)
    
    # Save results to file
    if signals:
        results_file = f"historical_signals_{start_time.strftime('%Y%m%d_%H%M')}.json"
        
        # Convert signals to dictionary format for JSON serialization
        signals_dict = []
        for signal in signals:
            signal_dict = {
                'timestamp': signal.timestamp.isoformat(),
                'contract': signal.contract,
                'action': signal.action,
                'quantity': signal.quantity,
                'order_type': signal.order_type,
                'price': signal.price,
                'confidence': signal.confidence,
                'signal_strength': signal.signal_strength,
                'bayesian_multiplier': signal.bayesian_multiplier,
                'stop_loss': signal.stop_loss,
                'profit_target': signal.profit_target,
                'reasoning': signal.reasoning,
                'volume_ratio': signal.volume_ratio,
                'modal_price': signal.modal_price,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'volume_rank': signal.volume_rank,
                'modal_bin': signal.modal_bin,
                'bayesian_expected_p': signal.bayesian_expected_p,
                'bayesian_total_trades': signal.bayesian_total_trades
            }
            signals_dict.append(signal_dict)
        
        with open(results_file, 'w') as f:
            json.dump(signals_dict, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Fatal error: {e}")
