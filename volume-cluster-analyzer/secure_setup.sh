#!/bin/bash
# V6 Bayesian Trading System - Secure Setup Script
# This script sets up the system on DigitalOcean with secure configuration

set -e

echo "🔐 V6 Bayesian Trading System - Secure Setup"
echo "============================================="

# Check if we're on the droplet
if [ ! -f "/opt/v6-trading-system/setup_digitalocean.sh" ]; then
    echo "❌ Please run this script from your DigitalOcean droplet"
    echo "   SSH into your droplet first: ssh root@104.248.137.83"
    exit 1
fi

cd /opt/v6-trading-system

echo "📋 Setting up secure configuration..."

# Create .env file from template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.template .env
    echo "✅ Created .env file"
    echo "⚠️  IMPORTANT: Edit .env file with your actual API keys and passwords"
    echo "   nano .env"
else
    echo "✅ .env file already exists"
fi

# Create secure email config
if [ ! -f "email_config.json" ]; then
    echo "📧 Creating secure email configuration..."
    
    # Read email settings from .env
    if [ -f ".env" ]; then
        source .env
        cat > email_config.json << EOF
{
  "smtp_server": "${EMAIL_SMTP_SERVER:-smtp.gmail.com}",
  "smtp_port": ${EMAIL_SMTP_PORT:-587},
  "email_address": "${EMAIL_ADDRESS}",
  "email_password": "${EMAIL_PASSWORD}",
  "recipients": [
    "albert.beccu@gmail.com",
    "j.thoendl@thoendl-investments.com"
  ],
  "db_path": "../data/paper_trades.db",
  "bayesian_db_path": "../data/bayesian_stats.db"
}
EOF
        echo "✅ Created email_config.json from .env"
    else
        echo "⚠️  .env file not found, using template"
        cp email_config.json.template email_config.json
    fi
else
    echo "✅ email_config.json already exists"
fi

# Create secure trading config
if [ ! -f "trading_config.env" ]; then
    echo "⚙️  Creating secure trading configuration..."
    
    if [ -f ".env" ]; then
        source .env
        cat > trading_config.env << EOF
# V6 Bayesian Trading System Configuration
DATABENTO_API_KEY=${DATABENTO_API_KEY}

# Trading Parameters
VOLUME_THRESHOLD=4.0
MIN_SIGNAL_STRENGTH=0.45
BAYESIAN_SCALING_FACTOR=6.0
BAYESIAN_MAX_MULTIPLIER=3.0
BASE_POSITION_SIZE=1
MAX_POSITION_SIZE=5

# Market Hours (EST)
MARKET_OPEN_TIME=09:30
MARKET_CLOSE_TIME=16:00

# Risk Management
USE_PROFIT_TARGETS=true
PROFIT_TARGET_RATIO=2.0
DEFAULT_ORDER_TYPE=LIMIT
DEFAULT_VALIDITY=DAY
USE_MARKET_ORDERS_ON_HIGH_CONFIDENCE=true
HIGH_CONFIDENCE_THRESHOLD=0.8

# System Configuration
DATA_UPDATE_INTERVAL_SECONDS=60
CLUSTER_DETECTION_COOLDOWN_MINUTES=30
RETENTION_MINUTES=1440
DATABASE_PATH=../data/paper_trades.db
BAYESIAN_DATABASE_PATH=../data/bayesian_stats.db
LATEST_RECOMMENDATION_PATH=../data/latest_recommendation.json
RECOMMENDATIONS_LOG_PATH=../data/recommendations.jsonl
EOF
        echo "✅ Created trading_config.env from .env"
    else
        echo "⚠️  .env file not found, using template"
        cp trading_config.env.template trading_config.env
    fi
else
    echo "✅ trading_config.env already exists"
fi

# Set secure permissions
echo "🔐 Setting secure file permissions..."
chmod 600 .env
chmod 600 email_config.json
chmod 600 trading_config.env
chmod 600 data/*.db 2>/dev/null || true

echo ""
echo "🎉 Secure setup complete!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Edit .env file with your actual credentials:"
echo "   nano .env"
echo ""
echo "2. Start the services:"
echo "   systemctl start v6-trading-system"
echo "   systemctl start v6-email-reporter"
echo "   systemctl start v6-monitoring-dashboard"
echo ""
echo "3. Check status:"
echo "   systemctl status v6-trading-system"
echo ""
echo "4. Access dashboard:"
echo "   http://104.248.137.83:5000"
echo ""
echo "✅ All sensitive data is now secure and not in git!"
