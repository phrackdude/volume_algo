#!/bin/bash
# V6 Bayesian Trading System - Automated Deployment Script
# This script deploys the trading system to DigitalOcean server

set -e  # Exit on any error

# Configuration
SERVER_IP="104.248.137.83"  # Your DigitalOcean droplet IP
SERVER_USER="root"  # or your preferred user
APP_DIR="/opt/v6-trading-system"
SERVICE_NAME="v6-trading-system"

echo "🚀 V6 Bayesian Trading System Deployment"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "src/real_time_trading_system.py" ]; then
    echo "❌ Error: Please run this script from the volume-cluster-analyzer directory"
    exit 1
fi

# Check if server IP is configured
if [ "$SERVER_IP" = "YOUR_DROPLET_IP" ]; then
    echo "❌ Error: Please configure SERVER_IP in deploy.sh with your DigitalOcean droplet IP"
    exit 1
fi

echo "📡 Deploying to server: $SERVER_IP"
echo "📁 Target directory: $APP_DIR"

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf v6-trading-system.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='data/*.db' \
    --exclude='data/*.log' \
    --exclude='*.zip' \
    --exclude='GLBX-*' \
    .

# Upload to server
echo "⬆️  Uploading to server..."
scp v6-trading-system.tar.gz $SERVER_USER@$SERVER_IP:/tmp/

# Deploy on server
echo "🔧 Deploying on server..."
ssh $SERVER_USER@$SERVER_IP << 'EOF'
    set -e
    
    # Create application directory
    sudo mkdir -p /opt/v6-trading-system
    cd /opt/v6-trading-system
    
    # Extract new version
    sudo tar -xzf /tmp/v6-trading-system.tar.gz
    
    # Set permissions
    sudo chown -R root:root /opt/v6-trading-system
    sudo chmod +x /opt/v6-trading-system/*.py
    sudo chmod +x /opt/v6-trading-system/src/*.py
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "🐍 Creating Python virtual environment..."
        sudo python3 -m venv venv
        sudo venv/bin/pip install --upgrade pip
    fi
    
    # Install/update dependencies
    echo "📦 Installing dependencies..."
    sudo venv/bin/pip install -r requirements_realtime.txt
    
    # Create data directory
    sudo mkdir -p data
    sudo chmod 755 data
    
    # Copy environment file if it exists
    if [ -f "/opt/v6-trading-system/trading_config.env" ]; then
        echo "⚙️  Environment configuration found"
    else
        echo "⚠️  Warning: trading_config.env not found. Please configure manually."
    fi
    
    # Reload systemd service
    echo "🔄 Reloading systemd service..."
    sudo systemctl daemon-reload
    sudo systemctl restart v6-trading-system
    
    # Check service status
    echo "📊 Service status:"
    sudo systemctl status v6-trading-system --no-pager
    
    echo "✅ Deployment completed successfully!"
EOF

# Clean up
rm v6-trading-system.tar.gz

echo "🎉 Deployment completed!"
echo "📊 Check service status with: ssh $SERVER_USER@$SERVER_IP 'sudo systemctl status v6-trading-system'"
echo "📋 View logs with: ssh $SERVER_USER@$SERVER_IP 'sudo journalctl -u v6-trading-system -f'"
