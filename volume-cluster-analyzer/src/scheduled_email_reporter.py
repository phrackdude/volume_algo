#!/usr/bin/env python3
"""
V6 Bayesian Trading System - Scheduled Email Reporter
Runs daily email reports at 4:30 PM EST
"""

import schedule
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from email_reporter import EmailReporter

logger = logging.getLogger(__name__)

class ScheduledEmailReporter:
    """Scheduled email reporter for daily performance reports"""
    
    def __init__(self, config_path: str = "../email_config.json"):
        self.config_path = config_path
        self.reporter = None
        self.load_config()
    
    def load_config(self):
        """Load email configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.reporter = EmailReporter(config)
            logger.info("Email configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading email configuration: {e}")
            raise
    
    def send_daily_report(self):
        """Send daily performance report"""
        try:
            logger.info("Sending daily performance report...")
            
            # Send report for the last trading day
            success = self.reporter.send_email_report(
                subject=f"V6 Bayesian Trading System - Daily Report ({datetime.now().strftime('%Y-%m-%d')})",
                days=1
            )
            
            if success:
                logger.info("✅ Daily report sent successfully")
            else:
                logger.error("❌ Failed to send daily report")
                
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    def run_scheduler(self):
        """Run the email scheduler"""
        logger.info("🕐 Starting V6 Trading System Email Scheduler")
        logger.info("📧 Daily reports scheduled for 4:30 PM EST")
        
        # Schedule daily report at 4:30 PM EST
        schedule.every().day.at("16:30").do(self.send_daily_report)
        
        # Also schedule a test report for immediate testing
        logger.info("🧪 Test report scheduled for 1 minute from now")
        schedule.every().minute.do(self.send_test_report).tag('test')
        
        # Remove test schedule after first run
        def remove_test_schedule():
            schedule.clear('test')
            logger.info("🧪 Test schedule removed")
        
        schedule.every().minute.do(remove_test_schedule).tag('cleanup')
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️  Email scheduler stopped by user")
        except Exception as e:
            logger.error(f"❌ Scheduler error: {e}")
    
    def send_test_report(self):
        """Send a test report (for immediate testing)"""
        try:
            logger.info("🧪 Sending test report...")
            
            success = self.reporter.send_email_report(
                subject=f"V6 Bayesian Trading System - TEST Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                days=1
            )
            
            if success:
                logger.info("✅ Test report sent successfully")
            else:
                logger.error("❌ Failed to send test report")
                
        except Exception as e:
            logger.error(f"Error sending test report: {e}")

def main():
    """Main entry point for scheduled email reporter"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V6 Trading System Scheduled Email Reporter')
    parser.add_argument('--config', default='../email_config.json', help='Path to email configuration file')
    parser.add_argument('--test', action='store_true', help='Send test report immediately')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../data/email_scheduler.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Create scheduled reporter
        scheduler = ScheduledEmailReporter(args.config)
        
        if args.test:
            # Send test report immediately
            logger.info("🧪 Sending test report...")
            scheduler.send_test_report()
        else:
            # Run the scheduler
            scheduler.run_scheduler()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
