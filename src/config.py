#!/usr/bin/env python3
"""
Configuration file for V6 Bayesian Real-Time Trading System
Centralizes all settings, parameters, and API credentials
"""

import os
from dataclasses import dataclass
from typing import Dict, List
from datetime import time

@dataclass
class TradingSystemConfig:
    """Main configuration for the V6 trading system"""
    
    # =================================================================
    # API CREDENTIALS
    # =================================================================
    databento_api_key: str = os.getenv('DATABENTO_API_KEY', '')
    
    # =================================================================
    # V6 STRATEGY PARAMETERS (from your backtesting)
    # =================================================================
    
    # Volume cluster detection
    volume_threshold: float = 4.0
    min_signal_strength: float = 0.45
    min_volume_ratio: float = 60.0
    retention_minutes: int = 60
    
    # Bayesian parameters
    bayesian_scaling_factor: float = 6.0
    bayesian_max_multiplier: float = 3.0
    min_trades_for_bayesian: int = 3
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    
    # Position sizing
    base_position_size: int = 1
    max_position_size: int = 3
    max_risk_per_trade: float = 0.02  # 2% of portfolio
    
    # Risk management
    profit_target_ratio: float = 2.0
    volatility_lookback_periods: int = 20
    use_profit_targets: bool = True
    use_trailing_stops: bool = False
    
    # Order execution preferences
    default_order_type: str = "LIMIT"  # LIMIT, MARKET, STOP_LIMIT
    default_validity: str = "DAY"      # DAY, GTC, IOC, FOK
    limit_offset_ticks: float = 0.0    # Offset from signal price for limit orders
    use_market_orders_on_high_confidence: bool = False  # Use MARKET when confidence > 80%
    high_confidence_threshold: float = 0.80
    
    # Transaction costs (from your stress testing)
    commission_per_contract: float = 2.50
    slippage_ticks: float = 0.75
    tick_value: float = 12.50
    
    # =================================================================
    # MARKET CONFIGURATION
    # =================================================================
    
    # Trading hours (EST/EDT)
    market_open_time: time = time(9, 30)
    market_close_time: time = time(16, 0)
    
    # Contract specifications
    es_tick_size: float = 0.25
    es_contract_value: float = 50.0  # $50 per point
    
    # Data update intervals
    data_update_interval_seconds: int = 60  # 1 minute
    cluster_detection_cooldown_minutes: int = 30
    
    # =================================================================
    # SYSTEM CONFIGURATION
    # =================================================================
    
    # Data storage
    database_path: str = "/opt/v6-trading-system/data/bayesian_stats.db"
    log_file_path: str = "/opt/v6-trading-system/data/trading_system.log"
    recommendations_log_path: str = "/opt/v6-trading-system/data/recommendations_log.jsonl"
    latest_recommendation_path: str = "/opt/v6-trading-system/data/latest_recommendation.json"
    
    # Logging
    log_level: str = "INFO"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    
    # Safety features
    max_daily_trades: int = 10
    max_consecutive_losses: int = 3
    emergency_stop_loss_pct: float = 0.05  # 5% portfolio drawdown
    
    # =================================================================
    # CONTRACT SELECTION
    # =================================================================
    
    # Available contracts (will select highest volume automatically)
    available_contracts: List[str] = None
    
    # Contract mapping for Databento
    contract_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        # Load API key from environment
        self.databento_api_key = os.getenv('DATABENTO_API_KEY', '')
        
        if self.available_contracts is None:
            self.available_contracts = [
                'ES JUN25',
                'ES SEP25', 
                'ES DEC25',
                'ES MAR26'
            ]
        
        if self.contract_mapping is None:
            self.contract_mapping = {
                'ES JUN25': 'ESM6',  # ES Jun 2026
                'ES SEP25': 'ESU5',  # ES Sep 2025 (current front month)
                'ES DEC25': 'ESZ5',  # ES Dec 2025
                'ES MAR26': 'ESH6'   # ES Mar 2026
            }
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Check required API key
        if not self.databento_api_key:
            errors.append("⚠️  DATABENTO_API_KEY environment variable not set")
        
        # Validate strategy parameters
        if self.volume_threshold <= 0:
            errors.append("❌ volume_threshold must be positive")
            
        if not (0 <= self.min_signal_strength <= 1):
            errors.append("❌ min_signal_strength must be between 0 and 1")
            
        if self.bayesian_max_multiplier <= 1:
            errors.append("❌ bayesian_max_multiplier must be > 1")
            
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade >= 1:
            errors.append("❌ max_risk_per_trade must be between 0 and 1")
        
        # Validate file paths
        data_dir = os.path.dirname(self.database_path)
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
            except:
                errors.append(f"❌ Cannot create data directory: {data_dir}")
        
        return errors
    
    def print_config(self):
        """Print configuration summary"""
        print("⚙️  V6 TRADING SYSTEM CONFIGURATION")
        print("=" * 50)
        
        print("📊 STRATEGY PARAMETERS:")
        print(f"   Volume threshold: {self.volume_threshold}x")
        print(f"   Min signal strength: {self.min_signal_strength}")
        print(f"   Bayesian scaling: {self.bayesian_scaling_factor}")
        print(f"   Max multiplier: {self.bayesian_max_multiplier}x")
        print(f"   Profit target ratio: {self.profit_target_ratio}:1")
        
        print("\n💰 RISK MANAGEMENT:")
        print(f"   Max risk per trade: {self.max_risk_per_trade:.1%}")
        print(f"   Base position size: {self.base_position_size} contract(s)")
        print(f"   Max position size: {self.max_position_size} contract(s)")
        print(f"   Commission: ${self.commission_per_contract}/RT")
        print(f"   Expected slippage: {self.slippage_ticks} ticks")
        
        print("\n📈 MARKET SETUP:")
        print(f"   Trading hours: {self.market_open_time} - {self.market_close_time}")
        print(f"   Data interval: {self.data_update_interval_seconds}s")
        print(f"   Available contracts: {', '.join(self.available_contracts)}")
        
        print("\n💾 STORAGE:")
        print(f"   Database: {self.database_path}")
        print(f"   Log file: {self.log_file_path}")
        print(f"   Recommendations: {self.latest_recommendation_path}")
        
        # Validation
        errors = self.validate()
        if errors:
            print("\n⚠️  CONFIGURATION ISSUES:")
            for error in errors:
                print(f"   {error}")
        else:
            print("\n✅ Configuration validated successfully")

# Global configuration instance
config = TradingSystemConfig()

def load_config_from_file(file_path: str = "trading_config.env"):
    """Load configuration from environment file"""
    if os.path.exists(file_path):
        print(f"📄 Loading configuration from {file_path}")
        
        # Use python-dotenv for proper .env file loading
        try:
            from dotenv import load_dotenv
            load_dotenv(file_path)
        except ImportError:
            # Fallback to manual parsing if dotenv not available
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
        
        # Reload config with new environment variables
        global config
        config = TradingSystemConfig()
    else:
        print(f"⚠️  Config file {file_path} not found, using defaults")

def create_sample_config_file(file_path: str = "trading_config.env"):
    """Create a sample configuration file"""
    sample_config = """# V6 Bayesian Trading System Configuration
# Copy this file and customize with your settings

# =================================================================
# API CREDENTIALS
# =================================================================
DATABENTO_API_KEY=your_databento_api_key_here

# =================================================================
# OPTIONAL OVERRIDES (defaults are from your V6 backtesting)
# =================================================================

# Strategy parameters
# VOLUME_THRESHOLD=4.0
# MIN_SIGNAL_STRENGTH=0.45
# BAYESIAN_SCALING_FACTOR=6.0
# BAYESIAN_MAX_MULTIPLIER=3.0

# Risk management
# MAX_RISK_PER_TRADE=0.02
# COMMISSION_PER_CONTRACT=2.50
# SLIPPAGE_TICKS=0.75

# System settings
# LOG_LEVEL=INFO
# MAX_DAILY_TRADES=10
"""
    
    with open(file_path, 'w') as f:
        f.write(sample_config)
    
    print(f"✅ Sample configuration created: {file_path}")
    print("📝 Edit this file with your API credentials and preferences")

if __name__ == "__main__":
    print("🔧 V6 Trading System Configuration")
    print("=" * 40)
    
    # Print current configuration
    config.print_config()
    
    # Create sample config file
    if not os.path.exists("trading_config.env"):
        create_sample_config_file()
        print("\n💡 Next steps:")
        print("1. Edit trading_config.env with your Databento API key")
        print("2. Run: python src/real_time_trading_system.py") 