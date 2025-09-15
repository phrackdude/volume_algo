#!/bin/bash
# V6 Bayesian Trading System - Quick Setup for DigitalOcean
# Run this directly on your droplet

set -e

echo "🚀 V6 Bayesian Trading System - Quick Setup"
echo "==========================================="

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.11 and required packages
echo "🐍 Installing Python 3.11..."
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies
echo "📚 Installing system dependencies..."
apt install -y git curl wget build-essential

# Create application directory
echo "📁 Setting up application directory..."
mkdir -p /opt/v6-trading-system
cd /opt/v6-trading-system

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3.11 -m venv venv
venv/bin/pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
venv/bin/pip install pandas numpy databento python-dotenv pytz flask schedule

# Create data directory
echo "📊 Creating data directory..."
mkdir -p data
chmod 755 data

# Create basic configuration files
echo "⚙️  Creating configuration files..."

# Create .env template
cat > .env << 'EOF'
# V6 Bayesian Trading System Configuration
DATABENTO_API_KEY=your_databento_api_key_here
EMAIL_ADDRESS=albert.beccu@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
EMAIL_RECIPIENTS=albert.beccu@gmail.com,j.thoendl@thoendl-investments.com
EOF

# Create email config
cat > email_config.json << 'EOF'
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "email_address": "PLACEHOLDER_EMAIL",
  "email_password": "PLACEHOLDER_PASSWORD",
  "recipients": [
    "albert.beccu@gmail.com",
    "j.thoendl@thoendl-investments.com"
  ],
  "db_path": "../data/paper_trades.db",
  "bayesian_db_path": "../data/bayesian_stats.db"
}
EOF

# Create trading config
cat > trading_config.env << 'EOF'
# V6 Bayesian Trading System Configuration
DATABENTO_API_KEY=your_databento_api_key_here

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

# Set secure permissions
chmod 600 .env
chmod 600 email_config.json
chmod 600 trading_config.env

echo ""
echo "🎉 Quick setup complete!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Edit .env file with your actual credentials:"
echo "   nano .env"
echo ""
echo "2. Clone your repository:"
echo "   git clone https://github.com/phrackdude/volume_algo.git repo"
echo "   cp -r repo/volume-cluster-analyzer/* ."
echo ""
echo "3. Start the services:"
echo "   systemctl start v6-trading-system"
echo ""
echo "✅ Basic setup complete!"
