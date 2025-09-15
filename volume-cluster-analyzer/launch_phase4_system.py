#!/usr/bin/env python3
"""
V6 Bayesian Trading System - Phase 4 Complete System Launcher
Launches the complete trading system with monitoring and email reporting
"""

import asyncio
import threading
import time
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from automated_paper_trader import AutomatedPaperTrader
from monitoring_dashboard import TradingMonitor
from scheduled_email_reporter import ScheduledEmailReporter

logger = logging.getLogger(__name__)

class Phase4SystemLauncher:
    """Complete Phase 4 system launcher with all components"""
    
    def __init__(self):
        self.trading_system = None
        self.monitoring_dashboard = None
        self.email_reporter = None
        
        # Threading control
        self.dashboard_thread = None
        self.email_thread = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown()
    
    def start_monitoring_dashboard(self):
        """Start the monitoring dashboard in a separate thread"""
        try:
            logger.info("🌐 Starting monitoring dashboard...")
            self.monitoring_dashboard = TradingMonitor()
            
            # Run dashboard in a separate thread
            self.dashboard_thread = threading.Thread(
                target=self.monitoring_dashboard.run_dashboard,
                kwargs={'host': '0.0.0.0', 'port': 5000, 'debug': False},
                daemon=True
            )
            self.dashboard_thread.start()
            
            logger.info("✅ Monitoring dashboard started on http://localhost:5000")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring dashboard: {e}")
    
    def start_email_reporter(self):
        """Start the email reporter in a separate thread"""
        try:
            logger.info("📧 Starting email reporter...")
            self.email_reporter = ScheduledEmailReporter()
            
            # Run email reporter in a separate thread
            self.email_thread = threading.Thread(
                target=self.email_reporter.run_scheduler,
                daemon=True
            )
            self.email_thread.start()
            
            logger.info("✅ Email reporter started (daily reports at 4:30 PM EST)")
            
        except Exception as e:
            logger.error(f"❌ Failed to start email reporter: {e}")
    
    async def start_trading_system(self):
        """Start the main trading system"""
        try:
            logger.info("🤖 Starting V6 Bayesian trading system...")
            self.trading_system = AutomatedPaperTrader()
            await self.trading_system.run_automated_trading()
            
        except Exception as e:
            logger.error(f"❌ Trading system error: {e}")
            raise
    
    def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🛑 Shutting down Phase 4 system...")
        self.running = False
        
        # Stop trading system
        if self.trading_system:
            logger.info("⏹️  Stopping trading system...")
            # The trading system will stop when the main loop exits
        
        # Stop monitoring dashboard
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            logger.info("⏹️  Stopping monitoring dashboard...")
            # Dashboard will stop when the thread is killed
        
        # Stop email reporter
        if self.email_thread and self.email_thread.is_alive():
            logger.info("⏹️  Stopping email reporter...")
            # Email reporter will stop when the thread is killed
        
        logger.info("✅ Shutdown complete")
    
    async def run_system(self):
        """Run the complete Phase 4 system"""
        logger.info("🚀 V6 BAYESIAN TRADING SYSTEM - PHASE 4")
        logger.info("="*60)
        logger.info("🎯 Complete System Features:")
        logger.info("   ✅ Real-time paper trading with V6 Bayesian strategy")
        logger.info("   ✅ Web-based monitoring dashboard")
        logger.info("   ✅ Daily email performance reports")
        logger.info("   ✅ Automated deployment ready")
        logger.info("   ✅ System health monitoring")
        logger.info("="*60)
        
        self.running = True
        
        try:
            # Start monitoring dashboard
            self.start_monitoring_dashboard()
            
            # Start email reporter
            self.start_email_reporter()
            
            # Wait a moment for services to start
            await asyncio.sleep(2)
            
            # Start the main trading system
            await self.start_trading_system()
            
        except KeyboardInterrupt:
            logger.info("⏹️  System stopped by user")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            raise
        finally:
            self.shutdown()

def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../data/phase4_system.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting V6 Bayesian Trading System - Phase 4")
    logger.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create and run the system
    launcher = Phase4SystemLauncher()
    
    try:
        asyncio.run(launcher.run_system())
    except KeyboardInterrupt:
        logger.info("⏹️  System stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal system error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
