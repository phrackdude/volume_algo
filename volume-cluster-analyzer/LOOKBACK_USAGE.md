# Historical Signal Lookback Tool

## Overview
The `lookback.py` script allows you to cross-check your live trading robot with historical data using the exact same V6 Bayesian logic. This is perfect for verifying why you might have seen no trades or many trades during a specific time period.

## Features
- **Same Logic as Live System**: Uses identical volume cluster detection, Bayesian adaptation, and signal generation
- **Flexible Time Periods**: Analyze last 15 hours, 24 hours, 3 days, or custom periods
- **Bayesian Analysis**: Shows how the Bayesian adaptation would have worked with historical data
- **Detailed Signal Information**: Complete signal details including confidence, reasoning, and risk management
- **JSON Export**: Save results for further analysis

## Usage

### Basic Usage
```bash
python lookback.py
```

The script will prompt you to select a time period:
1. Last 15 hours
2. Last 24 hours  
3. Last 3 days
4. Custom period (enter hours to look back)

### Example Output
```
📊 HISTORICAL SIGNAL ANALYSIS SUMMARY
============================================================
Total Signals: 3

BUY Signals: 2
SHORT Signals: 1

Confidence Distribution:
  High (≥80%): 1
  Medium (60-79%): 1
  Low (<60%): 1

Bayesian Analysis:
  Signals with historical data: 2
  Average Bayesian multiplier: 1.45
  Average expected win probability: 0.67

📋 DETAILED SIGNAL LIST:
------------------------------------------------------------
 1. 2024-06-02 14:30:15 |  BUY  | $6012.50 | Conf: 85.2% | Vol:  4.2x | Bayesian: 1.8x
    Reasoning: V6 Bayesian: 4.2x volume, signal 0.456, modal_bin[2] multiplier 1.8x (p=0.852, trades=15)
    Stop: $6008.25 | Target: $6020.75

 2. 2024-06-02 15:45:30 | SHORT | $6008.75 | Conf: 72.1% | Vol:  3.8x | Bayesian: 1.3x
    Reasoning: V6 Bayesian: 3.8x volume, signal 0.423, modal_bin[7] multiplier 1.3x (p=0.721, trades=8)
    Stop: $6012.00 | Target: $6002.00
```

## Key Information Provided

### Signal Details
- **Timestamp**: Exact time when signal was generated
- **Action**: BUY or SHORT
- **Price**: Entry price
- **Confidence**: Bayesian confidence score (0-100%)
- **Volume Ratio**: How much higher than average volume
- **Bayesian Multiplier**: Position sizing multiplier based on historical performance

### Risk Management
- **Stop Loss**: Calculated stop loss price
- **Profit Target**: Calculated profit target price
- **Reasoning**: Detailed explanation of why the signal was generated

### Bayesian Analysis
- **Expected Win Probability**: Based on historical trades in the same modal bin
- **Total Historical Trades**: Number of past trades used for Bayesian calculation
- **Position Multiplier**: How much the position size was adjusted based on historical performance

## Configuration
The script uses the same configuration as your live system:
- Volume threshold: 4.0x average volume
- Minimum signal strength: 0.45
- Bayesian adaptation enabled
- Market hours: 9:30 AM - 4:00 PM EST

## Troubleshooting

### No Signals Detected
If you see "No signals detected in the specified time period", this could mean:
1. **Market was closed** during the time period
2. **No volume clusters** met the threshold requirements
3. **Signal strength** was below the minimum threshold
4. **Bayesian confidence** was too low

### Data Issues
If you get errors about data retrieval:
1. Check your Databento API key is configured
2. Ensure you have historical data access
3. Try a different time period

## Files Generated
- `historical_signals_YYYYMMDD_HHMM.json`: Complete signal data in JSON format
- Console output with detailed analysis

## Integration with Live System
This tool uses the exact same logic as your live `real_time_trading_system.py`:
- Same volume cluster detection
- Same Bayesian adaptation
- Same signal generation thresholds
- Same risk management calculations

This ensures perfect consistency between what the live system would have done and what the historical analysis shows.

## Example Scenarios

### Scenario 1: No Trades Overnight
```bash
python lookback.py
# Select option 1 (last 15 hours)
# Check if any signals were generated during market hours
```

### Scenario 2: Too Many Trades
```bash
python lookback.py  
# Select option 2 (last 24 hours)
# Analyze signal frequency and confidence levels
```

### Scenario 3: Specific Time Period
```bash
python lookback.py
# Select option 4 (custom period)
# Enter specific hours to look back
```

This tool gives you complete visibility into what your live system would have done with historical data, helping you understand and debug any discrepancies.

