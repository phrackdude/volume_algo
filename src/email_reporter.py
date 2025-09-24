#!/usr/bin/env python3
"""
V6 Bayesian Trading System - Email Performance Reporter
Sends daily performance reports via email with comprehensive trading metrics
"""

import smtplib
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional
import logging
import os
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EmailReporter:
    """Email performance reporter for V6 trading system"""
    
    def __init__(self, config: Dict):
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.email_address = config.get('email_address')
        self.email_password = config.get('email_password')
        self.recipients = config.get('recipients', [])
        
        self.db_path = config.get('db_path', '/opt/v6-trading-system/data/paper_trades.db')
        self.bayesian_db_path = config.get('bayesian_db_path', '/opt/v6-trading-system/data/bayesian_stats.db')
        
        # Validate configuration
        if not self.email_address or not self.email_password:
            raise ValueError("Email address and password must be configured")
        
        if not self.recipients:
            raise ValueError("At least one recipient must be configured")
    
    def get_performance_data(self, days: int = 1) -> Dict:
        """Get comprehensive performance data for the specified period"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get trades for the period
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
                    gross_pnl,
                    commission_cost,
                    slippage_cost,
                    total_transaction_costs,
                    status,
                    confidence,
                    bayesian_multiplier,
                    exit_reason,
                    portfolio_balance_before,
                    portfolio_balance_after,
                    portfolio_pct_change
                FROM paper_trades 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
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
                    "gross_pnl": row[9],
                    "commission_cost": row[10],
                    "slippage_cost": row[11],
                    "total_transaction_costs": row[12],
                    "status": row[13],
                    "confidence": row[14],
                    "bayesian_multiplier": row[15],
                    "exit_reason": row[16],
                    "portfolio_balance_before": row[17],
                    "portfolio_balance_after": row[18],
                    "portfolio_pct_change": row[19]
                }
                trades.append(trade)
            
            # Get portfolio equity data
            cursor.execute('''
                SELECT 
                    timestamp,
                    balance,
                    pnl_today,
                    trades_today,
                    max_balance,
                    drawdown_pct
                FROM portfolio_equity 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            equity_data = cursor.fetchall()
            
            # Get all-time statistics
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
            
            all_time_stats = cursor.fetchone()
            
            conn.close()
            
            # Calculate metrics
            if all_time_stats and all_time_stats[0] > 0:
                total_trades, winning_trades, total_pnl, avg_pnl, worst_trade, best_trade, total_commission, total_slippage, total_costs = all_time_stats
                win_rate = (winning_trades / total_trades) * 100
            else:
                total_trades = winning_trades = total_pnl = avg_pnl = 0
                worst_trade = best_trade = total_commission = total_slippage = total_costs = 0
                win_rate = 0
            
            # Current portfolio status
            current_balance = 100000.0  # Starting balance
            if equity_data:
                current_balance = equity_data[0][1]  # Latest balance
                pnl_today = equity_data[0][2] or 0
                trades_today = equity_data[0][3] or 0
                max_balance = equity_data[0][4] or current_balance
                drawdown_pct = equity_data[0][5] or 0
            else:
                pnl_today = trades_today = drawdown_pct = 0
                max_balance = current_balance
            
            # Calculate returns
            starting_balance = 100000.0
            total_return = ((current_balance - starting_balance) / starting_balance) * 100
            
            # Calculate period-specific metrics
            period_trades = [t for t in trades if t['status'] == 'CLOSED']
            period_pnl = sum([t['pnl'] for t in period_trades])
            period_winning_trades = len([t for t in period_trades if t['pnl'] > 0])
            period_win_rate = (period_winning_trades / len(period_trades)) * 100 if period_trades else 0
            
            # Calculate Sharpe ratio (simplified)
            daily_returns = [row[2] for row in equity_data if row[2] is not None]
            sharpe_ratio = 0
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            
            return {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": days
                },
                "portfolio": {
                    "current_balance": current_balance,
                    "starting_balance": starting_balance,
                    "total_return_pct": total_return,
                    "pnl_today": pnl_today,
                    "trades_today": trades_today,
                    "max_balance": max_balance,
                    "max_drawdown_pct": drawdown_pct
                },
                "period_trading": {
                    "trades": len(period_trades),
                    "winning_trades": period_winning_trades,
                    "win_rate_pct": period_win_rate,
                    "total_pnl": period_pnl,
                    "avg_pnl": period_pnl / len(period_trades) if period_trades else 0
                },
                "all_time_trading": {
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
                    "sharpe_ratio": sharpe_ratio
                },
                "trades": trades,
                "equity_data": equity_data
            }
            
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}
    
    def get_bayesian_summary(self) -> Dict:
        """Get Bayesian learning summary"""
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
                alpha_post = 1.0 + wins
                beta_post = 1.0 + losses
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
            logger.error(f"Error getting Bayesian summary: {e}")
            return {}
    
    def create_html_report(self, performance_data: Dict, bayesian_data: Dict) -> str:
        """Create HTML email report"""
        
        p = performance_data.get('portfolio', {})
        pt = performance_data.get('period_trading', {})
        at = performance_data.get('all_time_trading', {})
        c = performance_data.get('costs', {})
        r = performance_data.get('risk_metrics', {})
        
        # Format currency
        def fmt_currency(value):
            return f"${value:,.2f}"
        
        def fmt_percent(value):
            return f"{value:.2f}%"
        
        # Get today's date
        today = datetime.now().strftime("%A, %B %d, %Y")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>V6 Bayesian Trading System - Daily Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 5px 0;
            font-size: 16px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ecf0f1;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #f39c12; }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .trades-table th,
        .trades-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        .trades-table th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        .trades-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .bayesian-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .bayesian-table th,
        .bayesian-table td {{
            padding: 8px;
            text-align: center;
            border: 1px solid #ecf0f1;
        }}
        .bayesian-table th {{
            background-color: #2c3e50;
            color: white;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 12px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-active {{ background-color: #27ae60; }}
        .status-warning {{ background-color: #f39c12; }}
        .status-error {{ background-color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 V6 Bayesian Trading System</h1>
            <p>Daily Performance Report</p>
            <p>{today}</p>
        </div>

        <div class="section">
            <h2>💰 Portfolio Status</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Current Balance</div>
                    <div class="metric-value">{fmt_currency(p.get('current_balance', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if p.get('total_return_pct', 0) >= 0 else 'negative'}">{fmt_percent(p.get('total_return_pct', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Today's P&L</div>
                    <div class="metric-value {'positive' if p.get('pnl_today', 0) >= 0 else 'negative'}">{fmt_currency(p.get('pnl_today', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Trades Today</div>
                    <div class="metric-value">{p.get('trades_today', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{fmt_percent(p.get('max_drawdown_pct', 0))}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Trading Performance</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Total Trades (All Time)</div>
                    <div class="metric-value">{at.get('total_trades', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate (All Time)</div>
                    <div class="metric-value">{fmt_percent(at.get('win_rate_pct', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total P&L (All Time)</div>
                    <div class="metric-value {'positive' if at.get('total_pnl', 0) >= 0 else 'negative'}">{fmt_currency(at.get('total_pnl', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Average P&L</div>
                    <div class="metric-value {'positive' if at.get('avg_pnl', 0) >= 0 else 'negative'}">{fmt_currency(at.get('avg_pnl', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Trade</div>
                    <div class="metric-value positive">{fmt_currency(at.get('best_trade', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Worst Trade</div>
                    <div class="metric-value negative">{fmt_currency(at.get('worst_trade', 0))}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>💸 Transaction Costs</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Total Commission</div>
                    <div class="metric-value">{fmt_currency(c.get('total_commission', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Slippage</div>
                    <div class="metric-value">{fmt_currency(c.get('total_slippage', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Transaction Costs</div>
                    <div class="metric-value">{fmt_currency(c.get('total_transaction_costs', 0))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Cost per Trade</div>
                    <div class="metric-value">{fmt_currency(c.get('avg_cost_per_trade', 0))}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>⚠️ Risk Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{r.get('sharpe_ratio', 0):.2f}</div>
                </div>
            </div>
        </div>
"""

        # Add recent trades section
        recent_trades = performance_data.get('trades', [])[:10]  # Last 10 trades
        if recent_trades:
            html_content += f"""
        <div class="section">
            <h2>📋 Recent Trades</h2>
            <table class="trades-table">
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
"""
            for trade in recent_trades:
                pnl_class = 'positive' if (trade.get('pnl') or 0) >= 0 else 'negative'
                html_content += f"""
                    <tr>
                        <td>{datetime.fromisoformat(trade['timestamp']).strftime('%m/%d %H:%M')}</td>
                        <td>{trade['action']}</td>
                        <td>{trade['contract']}</td>
                        <td>{trade['quantity']}</td>
                        <td>{fmt_currency(trade['execution_price'])}</td>
                        <td>{fmt_currency(trade['exit_price']) if trade['exit_price'] else '-'}</td>
                        <td class="{pnl_class}">{fmt_currency(trade['pnl']) if trade['pnl'] else '-'}</td>
                        <td>{trade['status']}</td>
                    </tr>
"""
            html_content += """
                </tbody>
            </table>
        </div>
"""

        # Add Bayesian learning section
        if bayesian_data:
            html_content += """
        <div class="section">
            <h2>🧠 Bayesian Learning Statistics</h2>
            <table class="bayesian-table">
                <thead>
                    <tr>
                        <th>Modal Bin</th>
                        <th>Trades</th>
                        <th>Wins</th>
                        <th>Win Rate</th>
                        <th>Expected P</th>
                        <th>Avg Return</th>
                        <th>Recent Win Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
            for bin_id, stats in sorted(bayesian_data.items()):
                win_rate_pct = stats['win_rate'] * 100
                recent_win_rate_pct = stats['recent_win_rate'] * 100
                avg_return_pct = stats['avg_return'] * 100
                
                html_content += f"""
                    <tr>
                        <td>{bin_id}</td>
                        <td>{stats['total_trades']}</td>
                        <td>{stats['wins']}</td>
                        <td>{win_rate_pct:.1f}%</td>
                        <td>{stats['expected_p']:.3f}</td>
                        <td class="{'positive' if avg_return_pct >= 0 else 'negative'}">{avg_return_pct:.2f}%</td>
                        <td>{recent_win_rate_pct:.1f}%</td>
                    </tr>
"""
            html_content += """
                </tbody>
            </table>
        </div>
"""

        html_content += f"""
        <div class="footer">
            <p>Generated by V6 Bayesian Trading System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>This is an automated report. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def send_email_report(self, subject: str = None, days: int = 1) -> bool:
        """Send daily performance report via email"""
        try:
            # Get performance data
            performance_data = self.get_performance_data(days)
            if not performance_data:
                logger.error("No performance data available")
                return False
            
            # Get Bayesian data
            bayesian_data = self.get_bayesian_summary()
            
            # Create HTML report
            html_content = self.create_html_report(performance_data, bayesian_data)
            
            # Create email
            if not subject:
                today = datetime.now().strftime("%Y-%m-%d")
                subject = f"V6 Bayesian Trading System - Daily Report ({today})"
            
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_address
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            
            # Create text version
            text_content = f"""
V6 Bayesian Trading System - Daily Report
{datetime.now().strftime('%A, %B %d, %Y')}

Portfolio Status:
- Current Balance: ${performance_data['portfolio'].get('current_balance', 0):,.2f}
- Total Return: {performance_data['portfolio'].get('total_return_pct', 0):.2f}%
- Today's P&L: ${performance_data['portfolio'].get('pnl_today', 0):,.2f}
- Trades Today: {performance_data['portfolio'].get('trades_today', 0)}

Trading Performance:
- Total Trades: {performance_data['all_time_trading'].get('total_trades', 0)}
- Win Rate: {performance_data['all_time_trading'].get('win_rate_pct', 0):.1f}%
- Total P&L: ${performance_data['all_time_trading'].get('total_pnl', 0):,.2f}
- Average P&L: ${performance_data['all_time_trading'].get('avg_pnl', 0):,.2f}

Transaction Costs:
- Total Commission: ${performance_data['costs'].get('total_commission', 0):,.2f}
- Total Slippage: ${performance_data['costs'].get('total_slippage', 0):,.2f}
- Total Transaction Costs: ${performance_data['costs'].get('total_transaction_costs', 0):,.2f}

Risk Metrics:
- Sharpe Ratio: {performance_data['risk_metrics'].get('sharpe_ratio', 0):.2f}

Generated by V6 Bayesian Trading System
"""
            
            # Attach parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_address, self.email_password)
                
                for recipient in self.recipients:
                    server.sendmail(self.email_address, recipient, msg.as_string())
                    logger.info(f"Email report sent to {recipient}")
            
            logger.info("Daily performance report sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
            return False

def main():
    """Main entry point for email reporting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V6 Trading System Email Reporter')
    parser.add_argument('--config', required=True, help='Path to email configuration file')
    parser.add_argument('--days', type=int, default=1, help='Number of days to include in report')
    parser.add_argument('--subject', help='Custom email subject')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run email reporter
    reporter = EmailReporter(config)
    success = reporter.send_email_report(subject=args.subject, days=args.days)
    
    if success:
        print("✅ Email report sent successfully")
    else:
        print("❌ Failed to send email report")

if __name__ == "__main__":
    main()
