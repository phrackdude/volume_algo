# 🚀 V6 Trading System - DigitalOcean Deployment Guide

## Your Droplet Details
- **Name**: ubuntu-s-1vcpu-1gb-fra1-01
- **IP**: 104.248.137.83
- **OS**: Ubuntu 22.04 (LTS) x64
- **Specs**: 1 GB Memory / 25 GB Disk

## Step-by-Step Deployment

### 1. SSH into Your Droplet
```bash
ssh root@104.248.137.83
```

### 2. Clone Your Repository
```bash
cd /opt
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git v6-trading-system
cd v6-trading-system
```

### 3. Run the Setup Script
```bash
chmod +x setup_digitalocean.sh
./setup_digitalocean.sh
```

### 4. Configure Email Settings
```bash
nano email_config.json
# Update with your Gmail app password
```

### 5. Configure Trading Settings
```bash
nano trading_config.env
# Update with your Databento API key
```

### 6. Start Services
```bash
systemctl start v6-trading-system
systemctl start v6-email-reporter
systemctl start v6-monitoring-dashboard
```

### 7. Check Status
```bash
systemctl status v6-trading-system
```

## Access Points
- **Dashboard**: http://104.248.137.83:5000
- **Logs**: `journalctl -u v6-trading-system -f`

## Email Setup
1. Enable 2FA on your Gmail account
2. Generate an App Password: https://myaccount.google.com/apppasswords
3. Use the app password in email_config.json
