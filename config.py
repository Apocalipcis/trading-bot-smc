"""
Configuration settings for SMC trading bot
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class SMCConfig:
    """Smart Money Concepts configuration"""
    
    # Fractal detection parameters
    fractal_left: int = 2
    fractal_right: int = 2
    
    # Risk management
    min_risk_reward: float = 2.0
    max_risk_per_trade: float = 0.02  # 2% of account per trade
    
    # Analysis parameters
    premium_discount_lookback: int = 500  # Candles for P/D calculation
    ob_lookback: int = 10  # Candles to look back for OB detection
    
    # Signal filtering
    require_fvg_confluence: bool = False  # Require FVG confluence for signals
    premium_discount_filter: bool = True  # Use P/D filter
    
    # File paths
    ltf_data_path: Optional[str] = None
    htf_data_path: Optional[str] = None
    output_path: str = "signals.csv"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "smc_bot.log"

# Default configuration instance
DEFAULT_CONFIG = SMCConfig()

def get_config() -> SMCConfig:
    """Get the default configuration"""
    return DEFAULT_CONFIG

def update_config(**kwargs) -> SMCConfig:
    """Update configuration with new values"""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    return config
