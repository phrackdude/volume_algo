# 🚀 V6 Bayesian Real-Time Trading System

**Revolutionary algorithmic trading system based on your extraordinary 99% performance improvement**

- **Win Rate**: 64.7% (exceptional for futures)
- **Returns**: 0.813% per trade (position-adjusted) 
- **Bayesian Enhancement**: 98.7% utilization rate
- **Transaction Cost Resilient**: Tested across multiple cost scenarios

## 📋 Table of Contents

1. [🏗️ System Architecture](#system-architecture)
2. [⚡ Quick Start](#quick-start)
3. [🔧 Configuration](#configuration)
4. [📊 How It Works](#how-it-works)
5. [💰 Order Output Format](#order-output-format)
6. [🛡️ Risk Management](#risk-management)
7. [📈 Monitoring](#monitoring)
8. [🔍 Troubleshooting](#troubleshooting)
9. [📋 Daily Trading Workflow](docs/ORDER_WORKFLOW.md) 👈 **Start Here for Daily Routine**

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Databento     │───▶│  V6 Bayesian     │───▶│   Trading       │
│   Live Feed     │    │  Strategy Engine │    │   Recommendations│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Bayesian Stats  │
                       │    Database      │
                       └──────────────────┘
```

**Core Components:**
- **Real-Time Data Ingestion**: Databento API integration
- **Volume Cluster Detection**: Live implementation of your V6 strategy
- **Bayesian Statistics Engine**: Adaptive position sizing based on historical context performance
- **Order Recommendation System**: Outputs ready-to-trade signals
- **Risk Management**: Built-in safeguards and position limits

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Install the real-time system requirements
pip install -r requirements_realtime.txt

# Install Databento SDK (for live data)
pip install databento
```

### 2. Get Databento API Access

1. Sign up at [Databento.com](https://databento.com)
2. Subscribe to **GLBX.MDP3** dataset (ES futures data)
3. Get your API key from the dashboard

### 3. Configure the System

```bash
# Run the launcher to create configuration
python launch_trading_system.py
```

This creates `trading_config.env`. Edit it with your API key:

```env
# V6 Bayesian Trading System Configuration
DATABENTO_API_KEY=your_actual_api_key_here
```

### 4. Start Trading

```bash
# Launch the real-time system
python launch_trading_system.py
```

The system will:
- ✅ Validate your configuration
- ✅ Connect to Databento live feed  
- ✅ Start monitoring for volume clusters
- ✅ Generate trading recommendations

## 🔧 Configuration

### Core V6 Strategy Parameters (Pre-Tuned)

```python
# Volume cluster detection (from your backtesting)
volume_threshold: 4.0              # 4x average volume
min_signal_strength: 0.45          # Minimum signal quality
min_volume_ratio: 60.0             # 60x volume spikes

# Bayesian parameters (optimized)
bayesian_scaling_factor: 6.0       # Position sizing scaling
bayesian_max_multiplier: 3.0       # Maximum 3x position size
min_trades_for_bayesian: 3         # Minimum trades for statistics

# Risk management
max_risk_per_trade: 0.02           # 2% portfolio risk
profit_target_ratio: 2.0           # 2:1 reward/risk
commission_per_contract: 2.50      # Your transaction costs
slippage_ticks: 0.75               # Expected slippage
```

### Contract Selection (Dynamic)

The system automatically trades the **highest volume ES contract**:
- **Current**: ES JUN25 (1.2M+ volume)
- **Future**: Will migrate to ES SEP25 when volume shifts
- **Manual Override**: Edit `available_contracts` in config

## 📊 How It Works

### 1. **Real-Time Data Flow**
```
Databento API → 1-minute OHLCV bars → Volume cluster detection
```

### 2. **Volume Cluster Detection**
- Monitors volume ratio vs. 20-period moving average
- Detects 4x+ volume spikes with strong signal strength
- Calculates modal price (Point of Control)
- Determines trade direction (long/short)

### 3. **Bayesian Position Sizing**
```python
# For each volume cluster:
context_value = calculate_modal_bin(modal_position)
historical_stats = get_context_performance(context_value)
bayesian_multiplier = calculate_multiplier(historical_stats)
position_size = base_size * bayesian_multiplier  # 1-3 contracts
```

### 4. **Signal Generation**
- **Entry**: Limit order at cluster detection price
- **Stop Loss**: 1.5σ volatility-based
- **Profit Target**: 2:1 reward/risk ratio
- **Time Exit**: 60-minute maximum hold

## 💰 Order Output Format

When a signal is generated, you'll see:

```
🚨 V6 BAYESIAN TRADING RECOMMENDATION
============================================================
⏰ Time: 2025-01-08 14:32:15
📊 Contract: ES JUN25
📈 Action: BUY
📦 Quantity: 2 contracts
💰 Price: $6010.75
📋 Order Type: LIMIT
⏳ Validity: DAY

📊 BAYESIAN ANALYSIS:
   Confidence: 73.2%
   Signal Strength: 0.687
   Position Multiplier: 2.34x

🎯 RISK MANAGEMENT:
   Stop Loss: $5995.25
   Profit Target: $6041.25
   Risk/Reward: 1:2.00

💡 Reasoning: Volume cluster detected: 8.3x volume, 
signal strength 0.687, Bayesian multiplier 2.34x (confidence 0.732)
============================================================
```

**JSON Output** (for automated systems):
```json
{
  "timestamp": "2025-01-08T14:32:15",
  "contract": "ES JUN25",
  "action": "BUY",
  "quantity": 2,
  "order_type": "LIMIT",
  "price": 6010.75,
  "validity": "DAY",
  "stop_loss": 5995.25,
  "profit_target": 6041.25,
  "confidence": 0.732,
  "bayesian_multiplier": 2.34
}
```

## 🛡️ Risk Management

### Built-in Safeguards

1. **Position Limits**: Maximum 3 contracts per trade
2. **Daily Limits**: Max 10 trades per day
3. **Drawdown Protection**: Emergency stop at 5% portfolio loss
4. **Consecutive Loss Limit**: Pause after 3 consecutive losses
5. **Market Hours**: Only trades during 9:30 AM - 4:00 PM EST
6. **Cool-down Period**: 30 minutes between volume cluster signals

### Transaction Cost Protection

Based on your stress testing:
- ✅ **Profitable** in 83% of cost scenarios
- ⚠️ **Cost Sensitive**: Choose low-cost broker
- ✅ **Optimized** for 0.75 tick slippage + $2.50 commission

## 📈 Monitoring

### Real-Time Outputs

1. **Console**: Live system status and signals
2. **Log File**: `data/trading_system.log`
3. **Latest Signal**: `data/latest_recommendation.json`
4. **Signal History**: `data/recommendations_log.jsonl`
5. **Bayesian Stats**: `data/bayesian_stats.db` (SQLite)

### Performance Tracking

The system continuously learns and adapts:
- Records win/loss for each modal bin context
- Updates Bayesian probabilities in real-time
- Tracks performance by market regime
- Monitors position sizing effectiveness

## 🔍 Troubleshooting

### Common Issues

**❌ "Databento API key not set"**
```bash
# Solution: Edit trading_config.env
DATABENTO_API_KEY=your_actual_key_here
```

**❌ "Failed to connect to Databento"**
- Check internet connection
- Verify API key is valid
- Ensure you have GLBX.MDP3 subscription

**⚠️ "Running in simulation mode"**
- Normal if Databento not installed
- Install with: `pip install databento`
- System will simulate realistic market data

**❌ "No volume clusters detected"**
- Normal during low volatility periods
- System waits for 4x+ volume spikes
- Check market hours (9:30 AM - 4:00 PM EST)

### Performance Validation

**Expected Behavior:**
- 📊 **Trade Frequency**: 2-5 signals per week
- 📈 **Win Rate**: ~65% (based on your backtesting)
- 💰 **Average Return**: ~0.8% per trade
- 🎯 **Bayesian Utilization**: >95%

## 🎯 Next Steps

### Phase 1: Paper Trading (Current)
- ✅ Start with simulation mode
- ✅ Monitor signal generation
- ✅ Validate against your backtest expectations

### Phase 2: Live Connection (Week 2)
- 🔜 Connect to Databento live feed
- 🔜 Paper trade with real market data
- 🔜 Track performance vs. backtest

### Phase 3: Live Trading (Month 2+)
- 🔜 Start with minimum position sizes
- 🔜 Gradually scale up based on performance
- 🔜 Full deployment of V6 strategy

---

## 📞 Support

**Your V6 strategy achieved 99% performance improvement** - this real-time system is designed to replicate those exact results in live markets.

**Key Success Metrics to Monitor:**
- Win rate should stabilize around 64-65%
- Position-adjusted returns ~0.8% per trade
- Bayesian utilization rate >95%
- Transaction costs should not exceed your stress test limits

🚀 **Ready to turn on and trade!** 