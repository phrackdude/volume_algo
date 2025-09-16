# 🚀 DigitalOcean Deployment Guide - V6 Bayesian Trading System

## ✅ **System Status: PRODUCTION READY**

All major improvements are complete and committed:
- ✅ **Bayesian calculations fixed** (proper modal position normalization)
- ✅ **Transaction costs corrected** ($11.875/contract, 0.2375%)
- ✅ **Volatility optimization** (98% data reduction for cost control)
- ✅ **Exit strategy enhanced** (60min limit, high/low monitoring)
- ✅ **Monitoring dashboard compatible**
- ✅ **Deployment scripts updated**

## 🎯 **Quick Deployment (Recommended)**

### **Step 1: Run Setup Script on DigitalOcean**
```bash
# SSH into your DigitalOcean droplet
ssh root@YOUR_DROPLET_IP

# Download and run the setup script
wget https://raw.githubusercontent.com/phrackdude/volume_algo/main/volume-cluster-analyzer/setup_digitalocean.sh
chmod +x setup_digitalocean.sh
./setup_digitalocean.sh
```

### **Step 2: Configure API Keys**
```bash
# Edit configuration files
cd /opt/v6-trading-system/volume-cluster-analyzer
nano trading_config.env

# Add your Databento API key:
DATABENTO_API_KEY=your_actual_api_key_here

# Configure email settings
nano email_config.json
# Add your email credentials for daily reports
```

### **Step 3: Start Services**
```bash
# Start all services
sudo systemctl start v6-trading-system
sudo systemctl start v6-email-reporter  
sudo systemctl start v6-monitoring-dashboard

# Check status
sudo systemctl status v6-trading-system
sudo systemctl status v6-monitoring-dashboard
```

### **Step 4: Access Dashboard**
- **Web Dashboard:** `http://YOUR_DROPLET_IP:5000`
- **Live Monitoring:** Real-time portfolio, trades, and Bayesian learning
- **System Logs:** `sudo journalctl -u v6-trading-system -f`

## 🔧 **Manual Deployment (Alternative)**

### **Using deploy.sh Script**
```bash
# From your local machine, in volume-cluster-analyzer directory
# Edit deploy.sh with your server IP
nano deploy.sh
# Change: SERVER_IP="YOUR_DROPLET_IP"

# Deploy
./deploy.sh
```

## 📊 **What Gets Deployed**

### **Core System:**
- **`real_time_trading_system.py`** - Main trading engine with Databento integration
- **`automated_paper_trader.py`** - Paper trading with realistic execution
- **`monitoring_dashboard.py`** - Web dashboard on port 5000
- **`launch_phase4_system.py`** - Main system launcher

### **Databases:**
- **`paper_trades.db`** - Trade execution records and portfolio tracking
- **`bayesian_stats.db`** - Learning database for position sizing
- **`latest_recommendation.json`** - Current market signals

### **Services (Auto-start):**
- **`v6-trading-system`** - Main trading system
- **`v6-email-reporter`** - Daily performance reports
- **`v6-monitoring-dashboard`** - Web dashboard

## 🎯 **Production Features**

### **Cost Optimization:**
- **Volatility calculation:** 480 bars (8 hours) vs 28,800 bars (20 days) = **98.3% cost reduction**
- **Intelligent fallbacks:** ES-specific volatility estimates (0.4-0.8%)
- **Efficient data usage:** Multi-method volatility with robust error handling

### **Risk Management:**
- **Portfolio limits:** Max 10% drawdown, 3% daily loss limit
- **Position sizing:** 1-2% risk per trade with Bayesian scaling
- **Time limits:** 60-minute maximum hold per trade
- **Stop losses:** Volatility-based with 0.5% minimum floor

### **Monitoring & Alerts:**
- **Real-time dashboard:** Live portfolio tracking and performance metrics
- **Daily email reports:** Sent to configured recipients at 4:30 PM EST
- **System health monitoring:** Automatic service restart on failure
- **Comprehensive logging:** Full audit trail in systemd journals

## 🔍 **Verification Steps**

### **1. Check System Status**
```bash
# Service status
sudo systemctl status v6-trading-system
sudo systemctl status v6-monitoring-dashboard

# View logs
sudo journalctl -u v6-trading-system -f
sudo journalctl -u v6-monitoring-dashboard -f
```

### **2. Test Dashboard**
- Navigate to `http://YOUR_DROPLET_IP:5000`
- Verify live ticker shows market status
- Check portfolio metrics display
- Confirm Bayesian learning tables appear

### **3. Verify Data Flow**
```bash
# Check databases exist
ls -la /opt/v6-trading-system/volume-cluster-analyzer/data/

# Check latest recommendation file
cat /opt/v6-trading-system/volume-cluster-analyzer/data/latest_recommendation.json
```

### **4. Monitor First Trading Session**
- Dashboard should show "MARKET OPEN" during trading hours (9:30-16:00 EST)
- Live ticker displays current ES price and volume data
- System logs show volume cluster detection attempts
- Paper trades appear in dashboard when signals trigger

## 🚨 **Troubleshooting**

### **Service Won't Start:**
```bash
# Check logs for errors
sudo journalctl -u v6-trading-system --no-pager

# Common issues:
# 1. Missing API key - edit trading_config.env
# 2. Database permissions - check /opt/v6-trading-system/volume-cluster-analyzer/data/
# 3. Python dependencies - reinstall with pip
```

### **Dashboard Not Accessible:**
```bash
# Check firewall
sudo ufw status
sudo ufw allow 5000/tcp

# Check service
sudo systemctl status v6-monitoring-dashboard
```

### **No Trading Signals:**
- Verify market hours (9:30-16:00 EST weekdays)
- Check Databento API key is valid
- Monitor logs for volume cluster detection
- Ensure ES futures data is flowing

## 📧 **Email Reports**

Daily reports automatically sent to:
- `albert.beccu@gmail.com`
- `j.thoendl@thoendl-investments.com`

Configure additional recipients in `email_config.json`.

## 🔄 **Updates & Maintenance**

### **Deploy Updates:**
```bash
# Pull latest changes
cd /opt/v6-trading-system
sudo git pull origin main
cd volume-cluster-analyzer

# Restart services
sudo systemctl restart v6-trading-system
sudo systemctl restart v6-monitoring-dashboard
```

### **Backup Data:**
```bash
# Backup databases
cp data/*.db /backup/location/
```

## ✅ **Success Checklist**

- [ ] DigitalOcean droplet created and accessible
- [ ] Setup script completed successfully
- [ ] Databento API key configured
- [ ] Email settings configured
- [ ] All services running (green status)
- [ ] Dashboard accessible on port 5000
- [ ] Live ticker showing market data
- [ ] System logs showing normal operation
- [ ] Firewall configured (ports 22, 5000 open)

## 🎉 **You're Live!**

Your V6 Bayesian Trading System is now running in production on DigitalOcean with:

- **Real-time market data** from Databento
- **Advanced volume cluster detection** with retest requirements
- **Bayesian learning** for adaptive position sizing
- **Realistic paper trading** with transaction costs
- **Professional monitoring** with web dashboard
- **Automated reporting** via email
- **Production-grade reliability** with auto-restart

**Dashboard URL:** `http://YOUR_DROPLET_IP:5000`

Monitor your first live trading session and watch the V6 Bayesian system in action! 🚀
