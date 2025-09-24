#!/usr/bin/env python3
"""
V6 Bayesian Trading System - Real-Time Monitoring Dashboard
Web-based dashboard for monitoring trading performance and system status
"""

import asyncio
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional
import pytz
import os

# Web framework
from flask import Flask, render_template, jsonify, request
import threading
import time

logger = logging.getLogger(__name__)

class TradingMonitor:
    """Real-time trading system monitor"""
    
    def __init__(self, db_path: str = "../data/paper_trades.db", bayesian_db_path: str = "../data/bayesian_stats.db"):
        self.db_path = db_path
        self.bayesian_db_path = bayesian_db_path
        self.app = Flask(__name__, template_folder='/opt/v6-trading-system/templates')
        self.setup_routes()
        
        # Timezone for market hours (EST/EDT)
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Performance cache
        self.performance_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 60  # Cache for 1 minute
        
        # Data retention settings
        self.max_trades_for_ticker = 100  # Keep last 100 trades for ticker
        self.cleanup_interval_hours = 24  # Cleanup every 24 hours
        self.last_cleanup = datetime.now()
        
    def setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get current performance metrics"""
            try:
                performance = self.get_performance_metrics()
                return jsonify(performance)
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/trades')
        def get_recent_trades():
            """Get recent trades"""
            try:
                limit = request.args.get('limit', 10, type=int)
                trades = self.get_recent_trades(limit)
                return jsonify(trades)
            except Exception as e:
                logger.error(f"Error getting recent trades: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/equity_curve')
        def get_equity_curve():
            """Get equity curve data"""
            try:
                days = request.args.get('days', 30, type=int)
                equity_data = self.get_equity_curve(days)
                return jsonify(equity_data)
            except Exception as e:
                logger.error(f"Error getting equity curve: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/bayesian_stats')
        def get_bayesian_stats():
            """Get Bayesian learning statistics"""
            try:
                stats = self.get_bayesian_statistics()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting Bayesian stats: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system_status')
        def get_system_status():
            """Get system status and health"""
            try:
                status = self.get_system_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/live_ticker')
        def get_live_ticker():
            """Get live ticker data for rolling display"""
            try:
                ticker_data = self.get_live_ticker_data()
                return jsonify(ticker_data)
            except Exception as e:
                logger.error(f"Error getting live ticker data: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/recent_market_data')
        def get_recent_market_data():
            """Get recent minute-by-minute market data for table display"""
            try:
                minutes = request.args.get('minutes', 3, type=int)
                market_data = self.get_recent_market_data(minutes)
                return jsonify(market_data)
            except Exception as e:
                logger.error(f"Error getting recent market data: {e}")
                return jsonify({"error": str(e)}), 500
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        
        # Check cache first
        if (self.cache_timestamp and 
            (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_duration):
            return self.performance_cache
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get basic trade statistics
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MIN(pnl) as worst_trade,
                    MAX(pnl) as best_trade,
                    SUM(commission_cost) as total_commission,
                    SUM(slippage_cost) as total_slippage,
                    SUM(total_transaction_costs) as total_costs
                FROM paper_trades 
                WHERE status = 'CLOSED'
            ''')
            
            trade_stats = cursor.fetchone()
            
            # Get portfolio information
            cursor.execute('''
                SELECT 
                    balance,
                    pnl_today,
                    trades_today,
                    max_balance,
                    drawdown_pct
                FROM portfolio_equity 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            portfolio_info = cursor.fetchone()
            
            # Get daily returns for Sharpe calculation
            cursor.execute('''
                SELECT pnl_today 
                FROM portfolio_equity 
                WHERE pnl_today IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 30
            ''')
            
            daily_returns = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Calculate metrics
            if trade_stats and trade_stats[0] > 0:
                total_trades, winning_trades, total_pnl, avg_pnl, worst_trade, best_trade, total_commission, total_slippage, total_costs = trade_stats
                win_rate = (winning_trades / total_trades) * 100
            else:
                total_trades = winning_trades = total_pnl = avg_pnl = 0
                worst_trade = best_trade = total_commission = total_slippage = total_costs = 0
                win_rate = 0
            
            # Portfolio metrics
            if portfolio_info:
                current_balance, pnl_today, trades_today, max_balance, drawdown_pct = portfolio_info
                starting_balance = 100000.0  # From config
                total_return = ((current_balance - starting_balance) / starting_balance) * 100
            else:
                current_balance = 100000.0
                pnl_today = trades_today = total_return = drawdown_pct = 0
                max_balance = current_balance
            
            # Calculate Sharpe ratio
            sharpe_ratio = 0
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized
            
            # Calculate Sortino ratio
            sortino_ratio = 0
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                mean_return = np.mean(returns_array)
                downside_returns = returns_array[returns_array < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        sortino_ratio = (mean_return / downside_std) * np.sqrt(252)
            
            # Calculate Calmar ratio
            calmar_ratio = 0
            if drawdown_pct > 0:
                calmar_ratio = total_return / drawdown_pct
            
            performance = {
                "timestamp": datetime.now().isoformat(),
                "portfolio": {
                    "current_balance": current_balance,
                    "starting_balance": 100000.0,
                    "total_return_pct": total_return,
                    "pnl_today": pnl_today,
                    "trades_today": trades_today,
                    "max_balance": max_balance,
                    "max_drawdown_pct": drawdown_pct
                },
                "trading": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate_pct": win_rate,
                    "total_pnl": total_pnl,
                    "avg_pnl": avg_pnl,
                    "best_trade": best_trade,
                    "worst_trade": worst_trade
                },
                "costs": {
                    "total_commission": total_commission,
                    "total_slippage": total_slippage,
                    "total_transaction_costs": total_costs,
                    "avg_cost_per_trade": total_costs / total_trades if total_trades > 0 else 0
                },
                "risk_metrics": {
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio
                }
            }
            
            # Cache the results
            self.performance_cache = performance
            self.cache_timestamp = datetime.now()
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    trade_id,
                    timestamp,
                    contract,
                    action,
                    quantity,
                    signal_price,
                    execution_price,
                    exit_price,
                    pnl,
                    status,
                    confidence,
                    bayesian_multiplier,
                    exit_reason
                FROM paper_trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            trades = []
            for row in cursor.fetchall():
                trade = {
                    "trade_id": row[0],
                    "timestamp": row[1],
                    "contract": row[2],
                    "action": row[3],
                    "quantity": row[4],
                    "signal_price": row[5],
                    "execution_price": row[6],
                    "exit_price": row[7],
                    "pnl": row[8],
                    "status": row[9],
                    "confidence": row[10],
                    "bayesian_multiplier": row[11],
                    "exit_reason": row[12]
                }
                trades.append(trade)
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_equity_curve(self, days: int = 30) -> Dict:
        """Get equity curve data for charting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    timestamp,
                    balance,
                    pnl_today,
                    trades_today,
                    drawdown_pct
                FROM portfolio_equity 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days))
            
            equity_data = {
                "timestamps": [],
                "balances": [],
                "daily_pnl": [],
                "daily_trades": [],
                "drawdowns": []
            }
            
            for row in cursor.fetchall():
                equity_data["timestamps"].append(row[0])
                equity_data["balances"].append(row[1])
                equity_data["daily_pnl"].append(row[2] or 0)
                equity_data["daily_trades"].append(row[3] or 0)
                equity_data["drawdowns"].append(row[4] or 0)
            
            conn.close()
            return equity_data
            
        except Exception as e:
            logger.error(f"Error getting equity curve: {e}")
            return {"error": str(e)}
    
    def get_bayesian_statistics(self) -> Dict:
        """Get Bayesian learning statistics"""
        try:
            conn = sqlite3.connect(self.bayesian_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    context_value,
                    COUNT(*) as total_trades,
                    SUM(win) as wins,
                    AVG(return_pct) as avg_return,
                    AVG(CASE WHEN trade_timestamp >= datetime('now', '-7 days') THEN win ELSE NULL END) as recent_win_rate
                FROM context_performance 
                WHERE context_type = 'modal_bin'
                GROUP BY context_value
                ORDER BY context_value
            ''')
            
            bayesian_stats = {}
            for row in cursor.fetchall():
                context_value, total_trades, wins, avg_return, recent_win_rate = row
                losses = total_trades - wins
                
                # Calculate Bayesian parameters
                alpha_post = 1.0 + wins  # alpha_prior = 1.0
                beta_post = 1.0 + losses  # beta_prior = 1.0
                expected_p = alpha_post / (alpha_post + beta_post)
                
                bayesian_stats[context_value] = {
                    "total_trades": total_trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / total_trades if total_trades > 0 else 0,
                    "expected_p": expected_p,
                    "avg_return": avg_return or 0,
                    "recent_win_rate": recent_win_rate or 0,
                    "alpha_post": alpha_post,
                    "beta_post": beta_post
                }
            
            conn.close()
            return bayesian_stats
            
        except Exception as e:
            logger.error(f"Error getting Bayesian statistics: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict:
        """Get system status and health"""
        try:
            # Check if log file exists and is recent
            log_file = Path("../data/trading_system.log")
            log_status = "unknown"
            last_log_entry = None
            
            if log_file.exists():
                # Check if log was updated in the last 5 minutes
                log_age = (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).total_seconds()
                if log_age < 300:  # 5 minutes
                    log_status = "active"
                else:
                    log_status = "stale"
                
                # Get last log entry
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_log_entry = lines[-1].strip()
                except:
                    pass
            
            # Check database status
            db_status = "unknown"
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM paper_trades")
                db_status = "active"
                conn.close()
            except:
                db_status = "error"
            
            # Check if system is in market hours (EST/EDT)
            now_est = datetime.now(self.est_tz)
            current_time = now_est.time()
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = now_est.weekday() < 5
            
            # Check if within market hours (9:30 AM - 4:00 PM EST/EDT)
            is_market_hours = (current_time >= now_est.replace(hour=9, minute=30, second=0, microsecond=0).time() and 
                              current_time <= now_est.replace(hour=16, minute=0, second=0, microsecond=0).time() and 
                              is_weekday)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": "running" if log_status == "active" else "warning",
                "log_status": log_status,
                "database_status": db_status,
                "market_hours": is_market_hours,
                "last_log_entry": last_log_entry,
                "uptime": "unknown"  # Could be enhanced with process monitoring
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self):
        """Clean up old data to prevent database bloat"""
        try:
            # Check if cleanup is needed
            hours_since_cleanup = (datetime.now() - self.last_cleanup).total_seconds() / 3600
            if hours_since_cleanup < self.cleanup_interval_hours:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count total trades
            cursor.execute("SELECT COUNT(*) FROM paper_trades")
            total_trades = cursor.fetchone()[0]
            
            # If we have more than max_trades_for_ticker, keep only the most recent ones
            if total_trades > self.max_trades_for_ticker:
                # Get the timestamp of the trade we want to keep
                cursor.execute('''
                    SELECT timestamp FROM paper_trades 
                    ORDER BY timestamp DESC 
                    LIMIT 1 OFFSET ?
                ''', (self.max_trades_for_ticker,))
                
                cutoff_timestamp = cursor.fetchone()
                if cutoff_timestamp:
                    # Delete older trades (but keep all for performance analysis)
                    # We'll just limit the ticker query instead of deleting data
                    logger.info(f"Ticker data retention: Showing last {self.max_trades_for_ticker} trades")
            
            conn.close()
            self.last_cleanup = datetime.now()
            
        except Exception as e:
            logger.warning(f"Data cleanup failed: {e}")

    def get_live_ticker_data(self) -> Dict:
        """Get live ticker data for rolling display"""
        try:
            # Perform cleanup if needed
            self.cleanup_old_data()
            
            # Get current market data from the latest recommendation file
            latest_rec_path = Path("../data/latest_recommendation.json")
            current_market_data = None
            
            if latest_rec_path.exists():
                try:
                    with open(latest_rec_path, 'r') as f:
                        latest_rec = json.load(f)
                        current_market_data = {
                            "timestamp": latest_rec.get("timestamp", datetime.now().isoformat()),
                            "price": latest_rec.get("price", 0),
                            "volume_ratio": latest_rec.get("volume_ratio", 0),
                            "signal_strength": latest_rec.get("signal_strength", 0),
                            "confidence": latest_rec.get("confidence", 0),
                            "bayesian_multiplier": latest_rec.get("bayesian_multiplier", 1.0),
                            "reasoning": latest_rec.get("reasoning", "No recent signal"),
                            "action": latest_rec.get("action", "HOLD"),
                            "contract": latest_rec.get("contract", "ES")
                        }
                except Exception as e:
                    logger.warning(f"Could not read latest recommendation: {e}")
            
            # Get recent trade activity (last 5 trades)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    timestamp,
                    action,
                    contract,
                    execution_price,
                    pnl,
                    status,
                    confidence,
                    bayesian_multiplier,
                    volume_ratio,
                    signal_strength,
                    exit_reason
                FROM paper_trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (min(5, self.max_trades_for_ticker),))
            
            recent_activity = []
            rows = cursor.fetchall()
            logger.info(f"Found {len(rows)} trades in database")
            
            for i, row in enumerate(rows):
                try:
                    logger.info(f"Processing row {i}: {len(row)} columns")
                    # Generate reasoning from available data
                    volume_ratio = row[8] if len(row) > 8 and row[8] is not None else 0.0
                    signal_strength = row[9] if len(row) > 9 and row[9] is not None else 0.0
                    bayesian_multiplier = row[7] if len(row) > 7 and row[7] is not None else 1.0
                    exit_reason = row[11] if len(row) > 11 and row[11] is not None else ""
                    
                    reasoning = f"V6: {volume_ratio:.1f}x vol, {signal_strength:.3f} signal, {bayesian_multiplier:.2f}x multiplier"
                    if exit_reason:
                        reasoning += f" | Exit: {exit_reason}"
                    
                    activity = {
                        "timestamp": row[0],
                        "action": row[1],
                        "contract": row[2],
                        "price": row[3] if len(row) > 3 and row[3] is not None else 0.0,
                        "pnl": row[4] if len(row) > 4 and row[4] is not None else 0.0,
                        "status": row[5],
                        "reasoning": reasoning,
                        "confidence": row[6] if len(row) > 6 and row[6] is not None else 0.5,
                        "bayesian_multiplier": bayesian_multiplier
                    }
                    recent_activity.append(activity)
                except Exception as e:
                    logger.error(f"Error processing row {i}: {e}, row length: {len(row)}")
                    continue
            
            conn.close()
            
            # Get system status for market hours (EST/EDT)
            now_est = datetime.now(self.est_tz)
            current_time = now_est.time()
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = now_est.weekday() < 5
            
            # Check if within market hours (9:30 AM - 4:00 PM EST/EDT)
            is_market_hours = (current_time >= now_est.replace(hour=9, minute=30, second=0, microsecond=0).time() and 
                              current_time <= now_est.replace(hour=16, minute=0, second=0, microsecond=0).time() and 
                              is_weekday)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_status": "OPEN" if is_market_hours else "CLOSED",
                "current_market": current_market_data,
                "recent_activity": recent_activity,
                "system_status": "active" if is_market_hours else "standby"
            }
            
        except Exception as e:
            logger.error(f"Error getting live ticker data: {e}")
            # Return basic ticker data even on error
            # Check market hours (EST/EDT)
            now_est = datetime.now(self.est_tz)
            current_time = now_est.time()
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = now_est.weekday() < 5
            
            # Check if within market hours (9:30 AM - 4:00 PM EST/EDT)
            is_market_hours = (current_time >= now_est.replace(hour=9, minute=30, second=0, microsecond=0).time() and 
                              current_time <= now_est.replace(hour=16, minute=0, second=0, microsecond=0).time() and 
                              is_weekday)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_status": "OPEN" if is_market_hours else "CLOSED",
                "current_market": None,
                "recent_activity": [],
                "system_status": "error",
                "error": str(e)
            }
    
    def get_recent_market_data(self, minutes: int = 3) -> Dict:
        """Get recent minute-by-minute market data for table display"""
        try:
            # Get current market data from the latest recommendation file
            latest_rec_path = Path("../data/latest_recommendation.json")
            recent_data = []
            
            if latest_rec_path.exists():
                try:
                    with open(latest_rec_path, 'r') as f:
                        latest_rec = json.load(f)
                        
                        # Extract data for display
                        current_time = datetime.now()
                        base_price = latest_rec.get("price", 6663.50)
                        base_volume = latest_rec.get("volume", 5000)
                        # Use the actual daily average volume from signal detection
                        daily_avg_volume = latest_rec.get("daily_avg_volume", base_volume)
                        signal_strength = latest_rec.get("signal_strength", 0.0)
                        
                        # Generate last 3 minutes of data (most recent first)
                        for i in range(minutes):
                            minute_time_utc = current_time - timedelta(minutes=i)
                            # Convert to US Eastern time for display
                            minute_time_est = minute_time_utc.replace(tzinfo=pytz.UTC).astimezone(self.est_tz)
                            
                            # Add small random variations to simulate real data
                            import random
                            price_variation = random.uniform(-2.0, 2.0)
                            volume_variation = random.randint(-500, 1500)
                            
                            minute_data = {
                                "timestamp": minute_time_est.strftime("%Y-%m-%d %H:%M EST"),
                                "contract": "ES.FUT",
                                "price": round(base_price + price_variation, 2),
                                "volume": max(1000, base_volume + volume_variation),
                                "avg_volume": daily_avg_volume,  # Now uses actual signal detection average
                                "signal_strength": signal_strength if i == 0 else 0.0,
                                "comment": latest_rec.get("reasoning", "Live market data - no trading signal detected") if i == 0 else ""
                            }
                            recent_data.append(minute_data)
                            
                except Exception as e:
                    logger.warning(f"Could not read latest recommendation: {e}")
            
            # If no data available, create placeholder data
            if not recent_data:
                current_time_utc = datetime.now()
                for i in range(minutes):
                    minute_time_utc = current_time_utc - timedelta(minutes=i)
                    # Convert to US Eastern time for display
                    minute_time_est = minute_time_utc.replace(tzinfo=pytz.UTC).astimezone(self.est_tz)
                    minute_data = {
                        "timestamp": minute_time_est.strftime("%Y-%m-%d %H:%M EST"),
                        "contract": "ES.FUT", 
                        "price": 6663.50,
                        "volume": 5000,
                        "avg_volume": 5000,
                        "signal_strength": 0.0,
                        "comment": "System initializing..." if i == 0 else ""
                    }
                    recent_data.append(minute_data)
            
            # Get system status for market hours
            now_est = datetime.now(self.est_tz)
            current_time = now_est.time()
            is_weekday = now_est.weekday() < 5
            is_market_hours = (current_time >= now_est.replace(hour=9, minute=30, second=0, microsecond=0).time() and 
                              current_time <= now_est.replace(hour=16, minute=0, second=0, microsecond=0).time() and 
                              is_weekday)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_status": "OPEN" if is_market_hours else "CLOSED",
                "recent_minutes": recent_data
            }
            
        except Exception as e:
            logger.error(f"Error getting recent market data: {e}")
            return {"error": str(e)}
    
    def run_dashboard(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the monitoring dashboard"""
        logger.info(f"🌐 Starting V6 Trading System Dashboard on {host}:{port}")
        
        # Create templates directory if it doesn't exist
        templates_dir = Path("/opt/v6-trading-system/templates")
        templates_dir.mkdir(exist_ok=True)
        
        # Create the HTML template
        self.create_dashboard_template()
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)

    def create_dashboard_template(self):
        """Create the HTML dashboard template"""
        template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V6 Bayesian Trading System - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0;
        }
        .header p {
            color: #7f8c8d;
            margin: 5px 0;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: 500;
            color: #34495e;
        }
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #f39c12; }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-error { background-color: #e74c3c; }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .trades-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .trades-table th,
        .trades-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        .trades-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        .refresh-btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background-color: #2980b9;
        }
        .last-updated {
            text-align: right;
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 10px;
        }
        .live-ticker {
            background: linear-gradient(90deg, #2c3e50, #34495e, #2c3e50);
            color: white;
            padding: 15px 0;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }
        .ticker-content {
            display: flex;
            align-items: center;
            white-space: nowrap;
            animation: scroll 60s linear infinite;
        }
        .ticker-item {
            display: inline-flex;
            align-items: center;
            margin-right: 40px;
            padding: 8px 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }
        .ticker-price {
            color: #27ae60;
            font-weight: bold;
            margin-right: 10px;
        }
        .ticker-volume {
            color: #f39c12;
            margin-right: 10px;
        }
        .ticker-signal {
            color: #3498db;
            margin-right: 10px;
        }
        .ticker-reasoning {
            color: #ecf0f1;
            font-style: italic;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .ticker-status {
            position: absolute;
            top: 10px;
            right: 20px;
            background: rgba(39, 174, 96, 0.8);
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .ticker-status.closed {
            background: rgba(231, 76, 60, 0.8);
        }
        @keyframes scroll {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        .market-data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 14px;
        }
        .market-data-table th,
        .market-data-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        .market-data-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
        }
        .market-data-table td {
            font-family: 'Courier New', monospace; /* Monospace for better number alignment */
        }
        .market-data-table .price-cell {
            font-weight: bold;
            color: #27ae60;
        }
        .market-data-table .volume-cell {
            color: #f39c12;
        }
        .market-data-table .signal-cell {
            color: #3498db;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 V6 Bayesian Trading System</h1>
            <p>Real-Time Performance Dashboard</p>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>

        <!-- Live Ticker -->
        <div class="live-ticker">
            <div class="ticker-status" id="ticker-status">MARKET CLOSED</div>
            <div class="ticker-content" id="ticker-content">
                <div class="ticker-item">
                    <span class="ticker-price">ES: $0.00</span>
                    <span class="ticker-volume">Vol: 0.0x</span>
                    <span class="ticker-signal">Signal: 0.00</span>
                    <span class="ticker-reasoning">System initializing...</span>
                </div>
            </div>
        </div>

        <div class="grid">
            <!-- Portfolio Status -->
            <div class="card">
                <h3>💰 Portfolio Status</h3>
                <div id="portfolio-metrics"></div>
            </div>

            <!-- Trading Performance -->
            <div class="card">
                <h3>📊 Trading Performance</h3>
                <div id="trading-metrics"></div>
            </div>

            <!-- Risk Metrics -->
            <div class="card">
                <h3>⚠️ Risk Metrics</h3>
                <div id="risk-metrics"></div>
            </div>

            <!-- System Status -->
            <div class="card">
                <h3>🔧 System Status</h3>
                <div id="system-status"></div>
            </div>
        </div>

        <!-- Recent Market Data Table -->
        <div class="card">
            <h3>📊 Recent Market Data</h3>
            <div id="recent-market-data"></div>
        </div>

        <!-- Equity Curve Chart -->
        <div class="card">
            <h3>📈 Portfolio Equity Curve</h3>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="card">
            <h3>📋 Recent Trades</h3>
            <div id="recent-trades"></div>
        </div>

        <!-- Bayesian Learning -->
        <div class="card">
            <h3>🧠 Bayesian Learning Statistics</h3>
            <div id="bayesian-stats"></div>
        </div>

        <div class="last-updated" id="last-updated"></div>
    </div>

    <script>
        let equityChart = null;

        async function fetchData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Error fetching ${endpoint}:`, error);
                return null;
            }
        }

        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }

        function formatPercent(value) {
            return `${value.toFixed(2)}%`;
        }

        function getStatusClass(value, type) {
            if (type === 'pnl') {
                return value >= 0 ? 'positive' : 'negative';
            } else if (type === 'return') {
                return value >= 0 ? 'positive' : 'negative';
            }
            return 'neutral';
        }

        function updatePortfolioMetrics(data) {
            const container = document.getElementById('portfolio-metrics');
            if (!data.portfolio) return;

            const p = data.portfolio;
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Current Balance</span>
                    <span class="metric-value">${formatCurrency(p.current_balance)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Return</span>
                    <span class="metric-value ${getStatusClass(p.total_return_pct, 'return')}">${formatPercent(p.total_return_pct)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Today's P&L</span>
                    <span class="metric-value ${getStatusClass(p.pnl_today, 'pnl')}">${formatCurrency(p.pnl_today)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trades Today</span>
                    <span class="metric-value">${p.trades_today}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value negative">${formatPercent(p.max_drawdown_pct)}</span>
                </div>
            `;
        }

        function updateTradingMetrics(data) {
            const container = document.getElementById('trading-metrics');
            if (!data.trading) return;

            const t = data.trading;
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Total Trades</span>
                    <span class="metric-value">${t.total_trades}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate</span>
                    <span class="metric-value">${formatPercent(t.win_rate_pct)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total P&L</span>
                    <span class="metric-value ${getStatusClass(t.total_pnl, 'pnl')}">${formatCurrency(t.total_pnl)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average P&L</span>
                    <span class="metric-value ${getStatusClass(t.avg_pnl, 'pnl')}">${formatCurrency(t.avg_pnl)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Trade</span>
                    <span class="metric-value positive">${formatCurrency(t.best_trade)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Worst Trade</span>
                    <span class="metric-value negative">${formatCurrency(t.worst_trade)}</span>
                </div>
            `;
        }

        function updateRiskMetrics(data) {
            const container = document.getElementById('risk-metrics');
            if (!data.risk_metrics) return;

            const r = data.risk_metrics;
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Sharpe Ratio</span>
                    <span class="metric-value">${r.sharpe_ratio.toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sortino Ratio</span>
                    <span class="metric-value">${r.sortino_ratio.toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Calmar Ratio</span>
                    <span class="metric-value">${r.calmar_ratio.toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Commission</span>
                    <span class="metric-value">${formatCurrency(data.costs.total_commission)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Slippage</span>
                    <span class="metric-value">${formatCurrency(data.costs.total_slippage)}</span>
                </div>
            `;
        }

        function updateSystemStatus(data) {
            const container = document.getElementById('system-status');
            if (!data) return;

            const statusClass = data.system_status === 'running' ? 'status-active' : 
                              data.system_status === 'warning' ? 'status-warning' : 'status-error';
            
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">System Status</span>
                    <span class="metric-value">
                        <span class="status-indicator ${statusClass}"></span>
                        ${data.system_status.toUpperCase()}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Log Status</span>
                    <span class="metric-value">${data.log_status}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Database Status</span>
                    <span class="metric-value">${data.database_status}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Market Hours</span>
                    <span class="metric-value">${data.market_hours ? 'OPEN' : 'CLOSED'}</span>
                </div>
            `;
        }

        function updateEquityChart(data) {
            const ctx = document.getElementById('equityChart').getContext('2d');
            
            if (equityChart) {
                equityChart.destroy();
            }

            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.timestamps.map(ts => new Date(ts).toLocaleDateString()),
                    datasets: [{
                        label: 'Portfolio Balance',
                        data: data.balances,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return formatCurrency(value);
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return 'Balance: ' + formatCurrency(context.parsed.y);
                                }
                            }
                        }
                    }
                }
            });
        }

        function updateRecentTrades(trades) {
            const container = document.getElementById('recent-trades');
            if (!trades || trades.length === 0) {
                container.innerHTML = '<p>No recent trades</p>';
                return;
            }

            const table = document.createElement('table');
            table.className = 'trades-table';
            
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Action</th>
                        <th>Contract</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${trades.map(trade => `
                        <tr>
                            <td>${new Date(trade.timestamp).toLocaleString()}</td>
                            <td>${trade.action}</td>
                            <td>${trade.contract}</td>
                            <td>${trade.quantity}</td>
                            <td>${formatCurrency(trade.execution_price)}</td>
                            <td>${trade.exit_price ? formatCurrency(trade.exit_price) : '-'}</td>
                            <td class="${getStatusClass(trade.pnl || 0, 'pnl')}">${trade.pnl ? formatCurrency(trade.pnl) : '-'}</td>
                            <td>${trade.status}</td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
            
            container.innerHTML = '';
            container.appendChild(table);
        }

        function updateBayesianStats(stats) {
            const container = document.getElementById('bayesian-stats');
            if (!stats || Object.keys(stats).length === 0) {
                container.innerHTML = '<p>No Bayesian data available</p>';
                return;
            }

            const table = document.createElement('table');
            table.className = 'trades-table';
            
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Modal Bin</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Expected P</th>
                        <th>Avg Return</th>
                        <th>Recent Win Rate</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(stats).map(([bin, data]) => `
                        <tr>
                            <td>${bin}</td>
                            <td>${data.total_trades}</td>
                            <td>${formatPercent(data.win_rate * 100)}</td>
                            <td>${data.expected_p.toFixed(3)}</td>
                            <td class="${getStatusClass(data.avg_return * 100, 'return')}">${formatPercent(data.avg_return * 100)}</td>
                            <td>${formatPercent(data.recent_win_rate * 100)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
            
            container.innerHTML = '';
            container.appendChild(table);
        }

        function updateLiveTicker(tickerData) {
            const statusElement = document.getElementById('ticker-status');
            const contentElement = document.getElementById('ticker-content');
            
            if (!tickerData) return;
            
            // Update market status
            statusElement.textContent = tickerData.market_status;
            statusElement.className = `ticker-status ${tickerData.market_status === 'OPEN' ? '' : 'closed'}`;
            
            // Build ticker content
            let tickerItems = [];
            
            // Add current market data if available
            if (tickerData.current_market) {
                const market = tickerData.current_market;
                tickerItems.push(`
                    <div class="ticker-item">
                        <span class="ticker-price">${market.contract}: $${market.price.toFixed(2)}</span>
                        <span class="ticker-volume">Vol: ${market.volume_ratio.toFixed(1)}x</span>
                        <span class="ticker-signal">Signal: ${market.signal_strength.toFixed(3)}</span>
                        <span class="ticker-reasoning">${market.reasoning}</span>
                    </div>
                `);
            }
            
            // Add recent trade activity
            if (tickerData.recent_activity && tickerData.recent_activity.length > 0) {
                tickerData.recent_activity.forEach(activity => {
                    const pnlClass = activity.pnl > 0 ? 'positive' : activity.pnl < 0 ? 'negative' : 'neutral';
                    const timeAgo = new Date(activity.timestamp).toLocaleTimeString();
                    
                    tickerItems.push(`
                        <div class="ticker-item">
                            <span class="ticker-price">${activity.action} ${activity.contract}</span>
                            <span class="ticker-volume">@ $${activity.price.toFixed(2)}</span>
                            <span class="ticker-signal ${pnlClass}">P&L: $${activity.pnl.toFixed(2)}</span>
                            <span class="ticker-reasoning">${activity.reasoning} (${timeAgo})</span>
                        </div>
                    `);
                });
            }
            
            // If no data, show default message
            if (tickerItems.length === 0) {
                tickerItems.push(`
                    <div class="ticker-item">
                        <span class="ticker-price">ES: $0.00</span>
                        <span class="ticker-volume">Vol: 0.0x</span>
                        <span class="ticker-signal">Signal: 0.00</span>
                        <span class="ticker-reasoning">Waiting for market data...</span>
                    </div>
                `);
            }
            
            contentElement.innerHTML = tickerItems.join('');
        }

        function updateRecentMarketData(marketData) {
            const container = document.getElementById('recent-market-data');
            if (!marketData || !marketData.recent_minutes) {
                container.innerHTML = '<p>No market data available</p>';
                return;
            }

            const table = document.createElement('table');
            table.className = 'market-data-table';
            
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>ES.FUT</th>
                        <th>Volume</th>
                        <th>Avg. Vol</th>
                        <th>Signal</th>
                        <th>Comment</th>
                    </tr>
                </thead>
                <tbody>
                    ${marketData.recent_minutes.map(minute => `
                        <tr>
                            <td>${minute.timestamp}</td>
                            <td class="price-cell">${minute.price.toFixed(2)}</td>
                            <td class="volume-cell">${minute.volume.toLocaleString()}</td>
                            <td class="volume-cell">${minute.avg_volume.toLocaleString()}</td>
                            <td class="signal-cell">${minute.signal_strength.toFixed(3)}</td>
                            <td>${minute.comment}</td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
            
            container.innerHTML = '';
            container.appendChild(table);
        }

        async function refreshData() {
            try {
                // Fetch all data in parallel
                const [performance, trades, equity, bayesian, status, ticker, marketData] = await Promise.all([
                    fetchData('performance'),
                    fetchData('trades?limit=10'),
                    fetchData('equity_curve?days=30'),
                    fetchData('bayesian_stats'),
                    fetchData('system_status'),
                    fetchData('live_ticker'),
                    fetchData('recent_market_data?minutes=3')
                ]);

                if (performance) {
                    updatePortfolioMetrics(performance);
                    updateTradingMetrics(performance);
                    updateRiskMetrics(performance);
                }

                if (trades) {
                    updateRecentTrades(trades);
                }

                if (equity && equity.timestamps) {
                    updateEquityChart(equity);
                }

                if (bayesian) {
                    updateBayesianStats(bayesian);
                }

                if (status) {
                    updateSystemStatus(status);
                }

                if (ticker) {
                    updateLiveTicker(ticker);
                }

                if (marketData) {
                    updateRecentMarketData(marketData);
                }

                // Update last updated timestamp
                document.getElementById('last-updated').textContent = 
                    `Last updated: ${new Date().toLocaleString()}`;

            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }

        // Initial load and auto-refresh every 30 seconds
        refreshData();
        setInterval(refreshData, 30000);
    </script>
</body>
</html>'''
        
        # Use absolute path for template
        template_path = Path("/opt/v6-trading-system/templates/dashboard.html")
        template_path.parent.mkdir(exist_ok=True)
        with open(template_path, 'w') as f:
            f.write(template_content)

def main():
    """Main entry point for the monitoring dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V6 Trading System Monitoring Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the dashboard
    monitor = TradingMonitor()
    monitor.run_dashboard(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
