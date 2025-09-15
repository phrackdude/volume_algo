# 🚀 V6 Trading System - Complete Setup Guide

## Your DigitalOcean Droplet
- **IP**: 104.248.137.83
- **Name**: ubuntu-s-1vcpu-1gb-fra1-01
- **OS**: Ubuntu 22.04 (LTS) x64

---

## 🔐 **Step 1: Secure Configuration Setup**

### **1.1 Create Gmail App Password**
1. Go to https://myaccount.google.com/apppasswords
2. Sign in to your Gmail account
3. Generate an app password for "Mail"
4. **Save this password** - you'll need it for the .env file

### **1.2 Get Your Databento API Key**
- Use your existing Databento API key
- If you don't have one, get it from https://databento.com

---

## 🖥️ **Step 2: Deploy to DigitalOcean**

### **2.1 SSH into Your Droplet**
```bash
ssh root@104.248.137.83
```

### **2.2 Clone Your Repository**
```bash
cd /opt
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git v6-trading-system
cd v6-trading-system
```

### **2.3 Run Initial Setup**
```bash
chmod +x setup_digitalocean.sh
./setup_digitalocean.sh
```

### **2.4 Run Secure Configuration**
```bash
chmod +x secure_setup.sh
./secure_setup.sh
```

### **2.5 Configure Your Credentials**
```bash
nano .env
```

**Fill in your actual values:**
```bash
# Databento API Configuration
DATABENTO_API_KEY=your_actual_databento_api_key

# Email Configuration
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=albert.beccu@gmail.com
EMAIL_PASSWORD=your_gmail_app_password_here
EMAIL_RECIPIENTS=albert.beccu@gmail.com,j.thoendl@thoendl-investments.com

# DigitalOcean Configuration
DROPLET_IP=104.248.137.83
DROPLET_USER=root
```

**Save and exit** (Ctrl+X, Y, Enter)

---

## 🚀 **Step 3: Start the System**

### **3.1 Start All Services**
```bash
systemctl start v6-trading-system
systemctl start v6-email-reporter
systemctl start v6-monitoring-dashboard
```

### **3.2 Enable Auto-Start**
```bash
systemctl enable v6-trading-system
systemctl enable v6-email-reporter
systemctl enable v6-monitoring-dashboard
```

### **3.3 Check Status**
```bash
systemctl status v6-trading-system
```

---

## 🌐 **Step 4: Access the Dashboard**

Open your web browser and go to:
**http://104.248.137.83:5000**

You should see the V6 Trading System dashboard with:
- Portfolio status
- Trading performance
- Risk metrics
- Recent trades
- Bayesian learning statistics

---

## 📧 **Step 5: Test Email System**

### **5.1 Send Test Email**
```bash
cd /opt/v6-trading-system
source venv/bin/activate
python test_email.py
```

### **5.2 Check Email**
- Check your inbox (albert.beccu@gmail.com)
- Check j.thoendl@thoendl-investments.com
- You should receive a test email with performance data

---

## 🔄 **Step 6: Set Up GitHub Actions (Automatic Updates)**

### **6.1 Generate SSH Key for GitHub Actions**
On your **local machine** (not the droplet):
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions" -f ~/.ssh/github_actions_key
# Press Enter for no passphrase
```

### **6.2 Add Public Key to Droplet**
```bash
# Copy the public key
cat ~/.ssh/github_actions_key.pub

# SSH into your droplet
ssh root@104.248.137.83

# Add the public key
echo "PASTE_YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
exit
```

### **6.3 Add Private Key to GitHub Secrets**
1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. **Name**: `DROPLET_SSH_KEY`
5. **Value**: Copy the entire content of `~/.ssh/github_actions_key` (the private key)
6. Click **Add secret**

### **6.4 Test Automatic Deployment**
1. Make a small change to any file
2. Commit and push to GitHub:
```bash
git add .
git commit -m "Test automatic deployment"
git push origin main
```
3. Go to your GitHub repository → **Actions** tab
4. You should see the deployment workflow running
5. Check your droplet - the system should restart automatically

---

## 📊 **Step 7: Monitor Your System**

### **7.1 Dashboard Access**
- **URL**: http://104.248.137.83:5000
- **Features**: Real-time monitoring, performance metrics, system health

### **7.2 Log Monitoring**
```bash
# View trading system logs
journalctl -u v6-trading-system -f

# View email reporter logs
journalctl -u v6-email-reporter -f

# View dashboard logs
journalctl -u v6-monitoring-dashboard -f
```

### **7.3 Email Reports**
- **Frequency**: Daily at 4:30 PM EST
- **Recipients**: albert.beccu@gmail.com, j.thoendl@thoendl-investments.com
- **Content**: Portfolio performance, trading stats, risk metrics

---

## 🛠️ **Troubleshooting**

### **Service Won't Start**
```bash
systemctl status v6-trading-system
journalctl -u v6-trading-system -n 50
```

### **Dashboard Not Accessible**
```bash
# Check if port 5000 is open
netstat -tlnp | grep 5000

# Check firewall
ufw status
```

### **Email Not Sending**
```bash
# Check email configuration
cat email_config.json

# Test email manually
python test_email.py
```

### **Database Issues**
```bash
# Check database files
ls -la data/

# Check permissions
chmod 755 data/
chmod 644 data/*.db
```

---

## ✅ **Success Checklist**

- [ ] DigitalOcean droplet deployed
- [ ] Services running (trading, email, dashboard)
- [ ] Dashboard accessible at http://104.248.137.83:5000
- [ ] Test email sent successfully
- [ ] GitHub Actions configured
- [ ] Automatic deployment working
- [ ] Daily email reports scheduled
- [ ] All sensitive data secured (not in git)

---

## 🎉 **You're All Set!**

Your V6 Bayesian Trading System is now:
- ✅ **Deployed** on DigitalOcean
- ✅ **Monitoring** via web dashboard
- ✅ **Reporting** daily via email
- ✅ **Auto-updating** via GitHub
- ✅ **Secure** (no secrets in git)

**Dashboard**: http://104.248.137.83:5000  
**Email Reports**: Daily at 4:30 PM EST  
**Auto-Deploy**: Push to GitHub main branch

---

## 📞 **Need Help?**

If you encounter any issues:
1. Check the logs: `journalctl -u v6-trading-system -f`
2. Verify configuration: `cat .env`
3. Test email: `python test_email.py`
4. Check dashboard: http://104.248.137.83:5000
