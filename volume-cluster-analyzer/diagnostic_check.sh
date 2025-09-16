#!/bin/bash
# V6 Bayesian Trading System - Diagnostic Check Script
# Run this script on your DigitalOcean droplet to verify system status

echo "🔍 V6 BAYESIAN TRADING SYSTEM DIAGNOSTIC CHECK"
echo "=============================================="
echo "Timestamp: $(date)"
echo ""

# Check if we're in the right directory
if [ ! -d "/opt/v6-trading-system/volume-cluster-analyzer" ]; then
    echo "❌ ERROR: System not found at /opt/v6-trading-system/volume-cluster-analyzer"
    exit 1
fi

cd /opt/v6-trading-system/volume-cluster-analyzer

echo "✅ Found system directory: $(pwd)"
echo ""

# 1. Check systemd services status
echo "🔧 SYSTEMD SERVICES STATUS:"
echo "------------------------"
services=("v6-trading-system" "v6-monitoring-dashboard" "v6-email-reporter")

for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        status="🟢 RUNNING"
    else
        status="🔴 STOPPED"
    fi
    echo "$service: $status"
    
    # Show recent logs if service is not running
    if ! systemctl is-active --quiet "$service"; then
        echo "   Recent logs:"
        journalctl -u "$service" --no-pager -n 3 | sed 's/^/   /'
    fi
done
echo ""

# 2. Check SQLite databases
echo "💾 DATABASE STATUS:"
echo "------------------"
databases=("data/paper_trades.db" "data/bayesian_stats.db")

for db in "${databases[@]}"; do
    if [ -f "$db" ]; then
        size=$(du -h "$db" | cut -f1)
        echo "✅ $db exists (size: $size)"
        
        # Check if database is accessible and has tables
        if [ "$db" = "data/paper_trades.db" ]; then
            table_count=$(sqlite3 "$db" "SELECT COUNT(name) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
            trade_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM paper_trades;" 2>/dev/null || echo "0")
            echo "   Tables: $table_count, Trades: $trade_count"
        elif [ "$db" = "data/bayesian_stats.db" ]; then
            table_count=$(sqlite3 "$db" "SELECT COUNT(name) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
            context_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM context_performance;" 2>/dev/null || echo "0")
            echo "   Tables: $table_count, Context records: $context_count"
        fi
    else
        echo "❌ $db not found"
    fi
done
echo ""

# 3. Check configuration files
echo "⚙️  CONFIGURATION FILES:"
echo "----------------------"
configs=("trading_config.env" "email_config.json")

for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "✅ $config exists"
        
        # Check for API key in trading config
        if [ "$config" = "trading_config.env" ]; then
            if grep -q "DATABENTO_API_KEY=your_databento_api_key_here" "$config"; then
                echo "   ⚠️  WARNING: Default API key detected - needs configuration"
            elif grep -q "DATABENTO_API_KEY=" "$config"; then
                echo "   ✅ API key configured"
            else
                echo "   ❌ No API key found"
            fi
        fi
    else
        echo "❌ $config not found"
    fi
done
echo ""

# 4. Check data directory and recent files
echo "📊 DATA DIRECTORY STATUS:"
echo "------------------------"
if [ -d "data" ]; then
    echo "✅ Data directory exists"
    echo "Directory contents:"
    ls -la data/ | sed 's/^/   /'
    
    # Check latest recommendation file
    if [ -f "data/latest_recommendation.json" ]; then
        echo "✅ Latest recommendation file exists"
        echo "   Last modified: $(stat -c %y data/latest_recommendation.json)"
        echo "   Content preview:"
        head -n 5 data/latest_recommendation.json | sed 's/^/   /'
    else
        echo "❌ Latest recommendation file not found"
    fi
    
    # Check log file
    if [ -f "data/trading_system.log" ]; then
        echo "✅ Trading system log exists"
        echo "   Size: $(du -h data/trading_system.log | cut -f1)"
        echo "   Last 3 lines:"
        tail -n 3 data/trading_system.log | sed 's/^/   /'
    else
        echo "❌ Trading system log not found"
    fi
else
    echo "❌ Data directory not found"
fi
echo ""

# 5. Check Python environment and dependencies
echo "🐍 PYTHON ENVIRONMENT:"
echo "---------------------"
if [ -f "venv/bin/python" ]; then
    echo "✅ Virtual environment exists"
    echo "Python version: $(venv/bin/python --version)"
    
    # Check key dependencies
    key_packages=("databento" "pandas" "numpy" "flask" "sqlite3")
    echo "Key package status:"
    for package in "${key_packages[@]}"; do
        if venv/bin/python -c "import $package" 2>/dev/null; then
            echo "   ✅ $package"
        else
            echo "   ❌ $package"
        fi
    done
else
    echo "❌ Virtual environment not found"
fi
echo ""

# 6. Check network connectivity and API access
echo "🌐 NETWORK CONNECTIVITY:"
echo "-----------------------"
if ping -c 1 google.com &> /dev/null; then
    echo "✅ Internet connectivity working"
else
    echo "❌ No internet connectivity"
fi

# Test Databento API (if configured)
if [ -f "trading_config.env" ] && grep -q "DATABENTO_API_KEY=" trading_config.env && ! grep -q "your_databento_api_key_here" trading_config.env; then
    echo "🔑 Testing Databento API connectivity..."
    # This would require running a Python script, so we'll skip for now
    echo "   (API test requires running the system - check logs for connectivity)"
else
    echo "⚠️  Databento API key not configured - cannot test API connectivity"
fi
echo ""

# 7. Check firewall and port accessibility
echo "🔥 FIREWALL STATUS:"
echo "-----------------"
if command -v ufw &> /dev/null; then
    echo "UFW status:"
    ufw status | sed 's/^/   /'
else
    echo "UFW not installed"
fi

# Check if port 5000 is listening
if netstat -ln | grep -q ":5000 "; then
    echo "✅ Port 5000 is listening (dashboard should be accessible)"
else
    echo "❌ Port 5000 not listening (dashboard not accessible)"
fi
echo ""

# 8. Recent system activity
echo "📋 RECENT SYSTEM ACTIVITY:"
echo "-------------------------"
echo "System uptime: $(uptime)"
echo ""
echo "Recent systemd journal entries for trading system:"
journalctl -u v6-trading-system --no-pager -n 5 | sed 's/^/   /'
echo ""

# 9. Market hours check
echo "🕐 MARKET HOURS CHECK:"
echo "--------------------"
current_time=$(date)
echo "Current server time: $current_time"

# Convert to EST/EDT (approximate check)
est_hour=$(TZ=America/New_York date +%H)
est_day=$(TZ=America/New_York date +%u)  # 1-7, Monday-Sunday

if [ "$est_day" -le 5 ] && [ "$est_hour" -ge 9 ] && [ "$est_hour" -le 16 ]; then
    echo "✅ Currently in market hours (EST/EDT)"
else
    echo "🔴 Currently outside market hours (EST/EDT)"
fi
echo ""

# 10. Summary and recommendations
echo "📋 DIAGNOSTIC SUMMARY:"
echo "---------------------"

# Count issues
issues=0

# Check critical services
for service in "${services[@]}"; do
    if ! systemctl is-active --quiet "$service"; then
        issues=$((issues + 1))
    fi
done

# Check critical files
critical_files=("data/paper_trades.db" "data/bayesian_stats.db" "trading_config.env")
for file in "${critical_files[@]}"; do
    if [ ! -f "$file" ]; then
        issues=$((issues + 1))
    fi
done

if [ $issues -eq 0 ]; then
    echo "🎉 SYSTEM STATUS: HEALTHY"
    echo "All critical components are running properly."
    echo ""
    echo "✅ Next steps:"
    echo "   1. Verify dashboard at http://104.248.137.83:5000"
    echo "   2. Monitor logs during market hours for trading activity"
    echo "   3. Check for volume cluster detection in logs"
else
    echo "⚠️  SYSTEM STATUS: NEEDS ATTENTION"
    echo "Found $issues issues that need to be resolved."
    echo ""
    echo "🔧 Recommended actions:"
    echo "   1. Review the issues listed above"
    echo "   2. Configure missing API keys if needed"
    echo "   3. Restart failed services: sudo systemctl restart SERVICE_NAME"
    echo "   4. Check logs for detailed error messages"
fi

echo ""
echo "🔍 Diagnostic check completed at $(date)"
echo "=============================================="
