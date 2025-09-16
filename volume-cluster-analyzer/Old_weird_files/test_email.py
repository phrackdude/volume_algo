#!/usr/bin/env python3
"""
Test email functionality for V6 Trading System
"""

import json
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

try:
    from email_reporter import EmailReporter
except ImportError:
    print("❌ Error: Could not import EmailReporter")
    print("   Make sure you're running this from the correct directory")
    sys.exit(1)

def test_email():
    """Send a test email"""
    try:
        # Check if email config exists
        config_path = Path('email_config.json')
        if not config_path.exists():
            print("❌ Error: email_config.json not found")
            print("   Run secure_setup.sh first to create the configuration")
            return False
        
        # Load configuration
        with open('email_config.json', 'r') as f:
            config = json.load(f)
        
        # Check if credentials are configured
        if config.get('email_address') == 'PLACEHOLDER_EMAIL':
            print("❌ Error: Email not configured")
            print("   Edit .env file with your actual email credentials")
            print("   Then run: ./secure_setup.sh")
            return False
        
        if config.get('email_password') == 'PLACEHOLDER_PASSWORD':
            print("❌ Error: Email password not configured")
            print("   Edit .env file with your Gmail app password")
            print("   Then run: ./secure_setup.sh")
            return False
        
        print("📧 Sending test email...")
        print(f"   From: {config['email_address']}")
        print(f"   To: {', '.join(config['recipients'])}")
        
        # Create reporter
        reporter = EmailReporter(config)
        
        # Send test report
        success = reporter.send_email_report(
            subject="V6 Trading System - TEST EMAIL",
            days=1
        )
        
        if success:
            print("✅ Test email sent successfully!")
            print("   Check your inbox for the test report")
            return True
        else:
            print("❌ Failed to send test email")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Check your email configuration and internet connection")
        return False

if __name__ == "__main__":
    print("🧪 V6 Trading System - Email Test")
    print("=" * 40)
    test_email()
