# V6 Bayesian Trading System - Data Flow Documentation

## 🎯 **System Overview**

The V6 Bayesian Trading System is a sophisticated real-time trading platform that combines volume cluster analysis, Bayesian learning, and automated paper trading with comprehensive monitoring.

## 📊 **Data Flow Architecture**

```
Databento API → real_time_trading_system.py → automated_paper_trader.py → SQLite Databases → monitoring_dashboard.py
     ↓                    ↓                           ↓                      ↓                    ↓
Live OHLCV Data    Volume Clusters         Paper Trades            Performance Data      Web Dashboard
                   Signal Strength         Portfolio Tracking      Bayesian Learning     Live Monitoring
                   Bayesian Sizing         Risk Management         Transaction Costs     Real-time Alerts
```

## 🔧 **Core Components**

### **1. Real-Time Trading System** (`real_time_trading_system.py`)
**Purpose:** Core trading engine that processes live market data and generates trading signals

**Data Sources:**
- **Input:** Live 1-minute OHLCV data from Databento API
- **Buffer:** Maintains 500 data points (~8 hours) for analysis
- **Historical:** 20-day lookback for volatility calculations (optimized to 8 hours in production)

**Key Calculations:**
- **Volume Cluster Detection:** Identifies 15-minute periods with volume ≥4x average
- **Rolling Volume Ranking:** Bias-free ranking using only past clusters (2-hour window)
- **Modal Position Analysis:** Calculates normalized position within cluster price range
- **Signal Strength:** Weighted combination of position (50%), volume (30%), momentum (20%)
- **Bayesian Position Sizing:** Multiplies base size by signal strength and Bayesian confidence

**Data Outputs:**
- **JSON Files:** Latest recommendations saved to `/data/latest_recommendation.json`
- **Log Files:** System activity logged to `/data/trading_system.log`
- **Trading Recommendations:** Structured recommendations with all parameters

**Retest Logic:**
- Requires price to retest modal price within ±0.75 points before entry
- 30-minute timeout for retest confirmation
- Immediate execution if already within tolerance

### **2. Automated Paper Trader** (`automated_paper_trader.py`)
**Purpose:** Executes paper trades with realistic market conditions and portfolio management

**Data Sources:**
- **Trading Signals:** From `real_time_trading_system.py`
- **Market Data:** Real-time OHLC for exit monitoring
- **Portfolio State:** Current balance, open positions, historical performance

**Key Processes:**
- **Order Execution:** 20-second delays with realistic bid/ask spread simulation
- **Volatility Calculation:** Multi-method approach (Garman-Klass, ATR, close-to-close)
- **Exit Monitoring:** Bar-by-bar high/low checking for stops and targets
- **Risk Management:** Portfolio-based position sizing with multiple safety limits

**Database Schema:**
```sql
-- Paper Trades Table
paper_trades:
- trade_id (PRIMARY KEY)
- timestamp, signal_time, execution_time, exit_time
- contract, action, quantity
- signal_price, execution_price, exit_price
- pnl, gross_pnl, commission_cost, slippage_cost, total_transaction_costs
- status, exit_reason
- volume_ratio, signal_strength, bayesian_multiplier, confidence
- portfolio_balance_before, portfolio_balance_after, portfolio_pct_change

-- Portfolio Equity Table  
portfolio_equity:
- timestamp (PRIMARY KEY)
- balance, pnl_today, trades_today
- max_balance, drawdown_pct
```

**Transaction Costs:**
- **Commission:** $2.50 per contract per trade
- **Slippage:** 0.75 ticks × $12.50 = $9.375 per contract
- **Total:** $11.875 per contract (0.2375% of $5000 notional)

### **3. Bayesian Statistics Manager** (`real_time_trading_system.py`)
**Purpose:** Learns from historical trade performance to optimize position sizing

**Database Schema:**
```sql
context_performance:
- context_type ('modal_bin')
- context_value (0-9 bins based on modal position)
- trade_timestamp, entry_price, exit_price, return_pct
- win (1 for profit, 0 for loss)
- volume_ratio, signal_strength
```

**Bayesian Calculation:**
- **Prior:** Beta(α=1, β=1) - uninformative prior
- **Posterior:** α_post = 1 + wins, β_post = 1 + losses
- **Expected Win Probability:** E[p] = α_post / (α_post + β_post)
- **Position Multiplier:** If E[p] > 0.5: mult = 1 + (E[p] - 0.5) × 6.0, capped at 3.0x

### **4. Monitoring Dashboard** (`monitoring_dashboard.py`)
**Purpose:** Web-based real-time monitoring and performance visualization

**Data Sources:**
- **Paper Trades Database:** Performance metrics, trade history
- **Bayesian Database:** Learning statistics by modal bin
- **Latest Recommendation:** Current market data and signals
- **System Logs:** Health monitoring and status

**Web Interface:**
- **Live Ticker:** Scrolling display of current market data and recent trades
- **Portfolio Metrics:** Balance, returns, drawdown, daily P&L
- **Trading Performance:** Win rate, total P&L, best/worst trades
- **Risk Metrics:** Sharpe, Sortino, Calmar ratios
- **Equity Curve:** Interactive chart of portfolio performance
- **Bayesian Learning:** Modal bin statistics and expected probabilities

**API Endpoints:**
- `/api/performance` - Portfolio and trading metrics
- `/api/trades` - Recent trade history
- `/api/equity_curve` - Historical balance data
- `/api/bayesian_stats` - Learning statistics
- `/api/system_status` - Health monitoring
- `/api/live_ticker` - Real-time ticker data

## 🚀 **Deployment Architecture**

### **DigitalOcean Production Setup:**
```
/opt/v6-trading-system/
├── src/                          # Core Python modules
│   ├── real_time_trading_system.py     # Main trading engine
│   ├── automated_paper_trader.py       # Paper trading execution
│   ├── monitoring_dashboard.py         # Web dashboard
│   ├── databento_connector.py          # Market data API
│   └── config.py                       # System configuration
├── data/                         # Databases and logs
│   ├── paper_trades.db              # Trade execution records
│   ├── bayesian_stats.db            # Learning database
│   ├── latest_recommendation.json   # Current signals
│   └── trading_system.log           # System activity
├── templates/                    # Web dashboard templates
└── venv/                        # Python virtual environment
```

### **Systemd Service:**
- **Service:** `v6-trading-system.service`
- **Auto-start:** Boots with system
- **Process monitoring:** Automatic restart on failure
- **Resource limits:** 2GB RAM, 200% CPU quota
- **Security:** Sandboxed execution environment

## 📈 **Data Processing Flow**

### **1. Signal Generation (Every Minute):**
```
Market Data Received → Volume Analysis → Cluster Detection → Signal Strength → 
Bayesian Lookup → Position Sizing → Retest Check → Trading Recommendation
```

### **2. Trade Execution:**
```
Signal Received → Portfolio Risk Check → Position Sizing → 20s Execution Delay → 
Realistic Slippage → Database Storage → Exit Monitoring Setup
```

### **3. Exit Monitoring (Continuous):**
```
Live OHLC Data → Volatility Calculation → Stop/Target Calculation → 
High/Low Monitoring → Exit Trigger → Trade Closure → Bayesian Feedback
```

### **4. Performance Tracking:**
```
Trade Results → Portfolio Update → Equity Curve → Performance Metrics → 
Bayesian Learning → Dashboard Display
```

## 🔍 **Key Data Transformations**

### **Volume Cluster Detection:**
- Raw minute data → 15-minute resampling → Volume threshold (4x) → Cluster identification
- Modal price calculation from 14-minute window post-cluster
- Normalized modal position: (modal_price - low) / (high - low)

### **Signal Strength Calculation:**
```python
# V6 Formula (weighted average):
signal_strength = (
    0.5 × position_strength +    # Distance from edge (0.15 for longs)
    0.3 × volume_strength +      # min(volume_ratio/150, 1.0)
    0.2 × momentum_strength      # 5-minute pre-signal momentum
)
```

### **Volatility Estimation (Production-Optimized):**
```python
# Multi-method approach with fallbacks:
1. Garman-Klass: 0.5×log(H/L)² - (2×log(2)-1)×log(C/O)²  # Most accurate
2. Close-to-close: std(returns)                           # Standard
3. ATR-based: 14-period ATR / entry_price                # Gap-resistant
4. Intelligent fallback: 0.4-0.8% based on price level  # ES-specific
```

## 🛡️ **Risk Management Layers**

### **1. Signal Level:**
- Minimum signal strength: 0.45 for longs
- Volume ranking: Only trade top cluster per day
- Retest requirement: Price must revisit modal level

### **2. Position Level:**
- Portfolio risk: 1-2% per trade based on Bayesian confidence
- Maximum position: 5 contracts per trade
- Stop losses: Volatility-based with 0.5% minimum floor

### **3. Portfolio Level:**
- Maximum drawdown: 10% (system shutdown)
- Daily loss limit: 3% (trading halt)
- Consecutive losses: 5 trades (trading halt)
- Maximum open positions: 3 concurrent trades

### **4. System Level:**
- Data validation: OHLC consistency checks
- Connection monitoring: Databento API health
- Error handling: Graceful degradation and fallbacks
- Logging: Comprehensive audit trail

## 📊 **Performance Monitoring**

### **Real-Time Metrics:**
- Portfolio balance and daily P&L
- Win rate and average trade P&L
- Current drawdown and maximum drawdown
- Bayesian learning statistics by modal bin
- Transaction cost analysis
- System health and market hours status

### **Advanced Analytics:**
- Sharpe ratio (risk-adjusted returns)
- Sortino ratio (downside risk focus)
- Calmar ratio (return vs. max drawdown)
- Equity curve visualization
- Trade distribution analysis
- Exit reason breakdown

This comprehensive data flow ensures robust, monitored, and adaptive trading operations with full auditability and risk control.
