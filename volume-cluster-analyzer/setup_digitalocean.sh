#!/bin/bash
# V6 Bayesian Trading System - DigitalOcean Setup Script
# Run this script on your DigitalOcean droplet to set up the complete system

set -e  # Exit on any error

echo "🚀 V6 Bayesian Trading System - DigitalOcean Setup"
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and required packages
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies
echo "📚 Installing system dependencies..."
sudo apt install -y git curl wget build-essential

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p /opt/v6-trading-system
cd /opt/v6-trading-system

# Clone repository (replace with your actual repository URL)
echo "📥 Cloning repository..."
if [ ! -d ".git" ]; then
    # Clone the V6 Bayesian Trading System repository
    sudo git clone https://github.com/phrackdude/volume_algo.git .
    cd volume-cluster-analyzer
else
    echo "Repository already exists, updating..."
    sudo git pull origin main
fi

# Set permissions
echo "🔐 Setting permissions..."
sudo chown -R root:root /opt/v6-trading-system
sudo chmod +x /opt/v6-trading-system/volume-cluster-analyzer/*.py
sudo chmod +x /opt/v6-trading-system/volume-cluster-analyzer/src/*.py

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
cd volume-cluster-analyzer
if [ ! -d "venv" ]; then
    sudo python3.11 -m venv venv
    sudo venv/bin/pip install --upgrade pip
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
sudo venv/bin/pip install -r requirements_realtime.txt

# Create data directory
echo "📊 Creating data directory..."
sudo mkdir -p data
sudo chmod 755 data

# Create systemd service file
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/v6-trading-system.service > /dev/null << 'EOF'
[Unit]
Description=V6 Bayesian Volume Cluster Trading System
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/v6-trading-system/volume-cluster-analyzer
Environment=PATH=/opt/v6-trading-system/volume-cluster-analyzer/venv/bin
ExecStart=/opt/v6-trading-system/volume-cluster-analyzer/venv/bin/python /opt/v6-trading-system/volume-cluster-analyzer/launch_phase4_system.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=v6-trading-system

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/v6-trading-system/volume-cluster-analyzer/data

# Resource limits
LimitNOFILE=65536
MemoryMax=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
EOF

# Create email reporter service
echo "📧 Creating email reporter service..."
sudo tee /etc/systemd/system/v6-email-reporter.service > /dev/null << 'EOF'
[Unit]
Description=V6 Bayesian Trading System Email Reporter
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/v6-trading-system/volume-cluster-analyzer
Environment=PATH=/opt/v6-trading-system/volume-cluster-analyzer/venv/bin
ExecStart=/opt/v6-trading-system/volume-cluster-analyzer/venv/bin/python /opt/v6-trading-system/volume-cluster-analyzer/src/scheduled_email_reporter.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=v6-email-reporter

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/v6-trading-system/volume-cluster-analyzer/data

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring dashboard service
echo "🌐 Creating monitoring dashboard service..."
sudo tee /etc/systemd/system/v6-monitoring-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=V6 Bayesian Trading System Monitoring Dashboard
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/v6-trading-system/volume-cluster-analyzer
Environment=PATH=/opt/v6-trading-system/volume-cluster-analyzer/venv/bin
ExecStart=/opt/v6-trading-system/volume-cluster-analyzer/venv/bin/python /opt/v6-trading-system/volume-cluster-analyzer/src/monitoring_dashboard.py --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=v6-monitoring-dashboard

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/v6-trading-system/volume-cluster-analyzer/data

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Enable services
echo "✅ Enabling services..."
sudo systemctl enable v6-trading-system
sudo systemctl enable v6-email-reporter
sudo systemctl enable v6-monitoring-dashboard

# Configure firewall (if ufw is installed)
if command -v ufw &> /dev/null; then
    echo "🔥 Configuring firewall..."
    sudo ufw allow 22/tcp    # SSH
    sudo ufw allow 5000/tcp  # Monitoring dashboard
    sudo ufw --force enable
fi

# Create configuration template
echo "⚙️  Creating configuration template..."
sudo tee /opt/v6-trading-system/trading_config.env.template > /dev/null << 'EOF'
# V6 Bayesian Trading System Configuration
# Copy this file to trading_config.env and fill in your values

# Databento API Configuration
DATABENTO_API_KEY=your_databento_api_key_here

# Email Configuration (for daily reports)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=albert.beccu@gmail.com,j.thoendl@thoendl-investments.com

# Trading Configuration
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

# Create email configuration template
echo "📧 Creating email configuration template..."
sudo tee /opt/v6-trading-system/email_config.json.template > /dev/null << 'EOF'
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "email_address": "your-email@gmail.com",
  "email_password": "your-app-password",
  "recipients": [
    "albert.beccu@gmail.com",
    "j.thoendl@thoendl-investments.com"
  ],
  "db_path": "../data/paper_trades.db",
  "bayesian_db_path": "../data/bayesian_stats.db"
}
EOF

# Create setup completion script
echo "📝 Creating setup completion script..."
sudo tee /opt/v6-trading-system/complete_setup.sh > /dev/null << 'EOF'
#!/bin/bash
# Complete the V6 Trading System setup

echo "🔧 V6 Trading System Setup Completion"
echo "====================================="

# Copy configuration templates
echo "📋 Setting up configuration files..."
if [ ! -f "trading_config.env" ]; then
    cp trading_config.env.template trading_config.env
    echo "✅ Created trading_config.env - Please edit with your API keys"
fi

if [ ! -f "email_config.json" ]; then
    cp email_config.json.template email_config.json
    echo "✅ Created email_config.json - Please edit with your email settings"
fi

echo ""
echo "📋 NEXT STEPS:"
echo "1. Edit trading_config.env with your Databento API key"
echo "2. Edit email_config.json with your email settings"
echo "3. Start the services:"
echo "   sudo systemctl start v6-trading-system"
echo "   sudo systemctl start v6-email-reporter"
echo "   sudo systemctl start v6-monitoring-dashboard"
echo ""
echo "4. Check service status:"
echo "   sudo systemctl status v6-trading-system"
echo "   sudo systemctl status v6-email-reporter"
echo "   sudo systemctl status v6-monitoring-dashboard"
echo ""
echo "5. View logs:"
echo "   sudo journalctl -u v6-trading-system -f"
echo "   sudo journalctl -u v6-email-reporter -f"
echo "   sudo journalctl -u v6-monitoring-dashboard -f"
echo ""
echo "6. Access monitoring dashboard:"
echo "   http://YOUR_DROPLET_IP:5000"
echo ""
echo "✅ Setup completion script ready!"
EOF

sudo chmod +x /opt/v6-trading-system/complete_setup.sh

# Run setup completion
echo "🔧 Running setup completion..."
cd /opt/v6-trading-system
sudo ./complete_setup.sh

echo ""
echo "🎉 V6 BAYESIAN TRADING SYSTEM SETUP COMPLETE!"
echo "============================================="
echo ""
echo "📋 CONFIGURATION REQUIRED:"
echo "1. Edit /opt/v6-trading-system/trading_config.env with your Databento API key"
echo "2. Edit /opt/v6-trading-system/email_config.json with your email settings"
echo ""
echo "🚀 STARTING SERVICES:"
echo "sudo systemctl start v6-trading-system"
echo "sudo systemctl start v6-email-reporter"
echo "sudo systemctl start v6-monitoring-dashboard"
echo ""
echo "📊 MONITORING:"
echo "• Dashboard: http://YOUR_DROPLET_IP:5000"
echo "• Logs: sudo journalctl -u v6-trading-system -f"
echo "• Status: sudo systemctl status v6-trading-system"
echo ""
echo "📧 EMAIL REPORTS:"
echo "• Daily reports sent at 4:30 PM EST"
echo "• Recipients: albert.beccu@gmail.com, j.thoendl@thoendl-investments.com"
echo ""
echo "🔄 AUTOMATIC DEPLOYMENT:"
echo "• Push to GitHub main branch to auto-deploy"
echo "• Services will restart automatically"
echo ""
echo "✅ Setup complete! Configure your API keys and start trading!"
