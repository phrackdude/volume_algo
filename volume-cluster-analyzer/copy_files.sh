#!/bin/bash
# Copy files from repository to application directory

echo "📁 Copying files from repository..."

# Copy all Python files
cp -r src/ /opt/v6-trading-system/
cp *.py /opt/v6-trading-system/ 2>/dev/null || true

# Copy configuration files
cp *.json /opt/v6-trading-system/ 2>/dev/null || true
cp *.env /opt/v6-trading-system/ 2>/dev/null || true
cp *.service /opt/v6-trading-system/ 2>/dev/null || true

# Copy requirements
cp requirements*.txt /opt/v6-trading-system/ 2>/dev/null || true

# Set permissions
chmod +x /opt/v6-trading-system/*.py 2>/dev/null || true
chmod +x /opt/v6-trading-system/src/*.py 2>/dev/null || true

echo "✅ Files copied successfully!"
