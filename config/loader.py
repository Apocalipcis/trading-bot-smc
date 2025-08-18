"""
Configuration loader for YAML files
"""
import yaml
import logging
from pathlib import Path
from typing import Optional

from .models import AppConfig, PairConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and saves configuration from YAML files"""
    
    def __init__(self, config_path: str = "config/pairs.yaml"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> AppConfig:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.info(f"Config file not found at {self.config_path}, creating default")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning("Empty config file, using defaults")
                return self._create_default_config()
            
            # Parse pairs
            pairs = []
            for pair_data in data.get('pairs', []):
                pair = PairConfig(**pair_data)
                pairs.append(pair)
            
            # Create app config
            config = AppConfig(
                max_concurrent_pairs=data.get('max_concurrent_pairs', 10),
                pairs=pairs,
                telegram_token=data.get('telegram_token'),
                telegram_chat_id=data.get('telegram_chat_id'),
                log_level=data.get('log_level', 'INFO'),
                binance_requests_per_minute=data.get('binance_requests_per_minute', 1200),
                backtest_concurrent_limit=data.get('backtest_concurrent_limit', 2),
                data_dir=data.get('data_dir', 'data'),
                logs_dir=data.get('logs_dir', 'logs'),
                backtest_results_dir=data.get('backtest_results_dir', 'backtest_results')
            )
            
            # Validate configuration
            errors = config.validate()
            if errors:
                logger.error(f"Configuration validation errors: {errors}")
                raise ValueError(f"Configuration validation failed: {errors}")
            
            logger.info(f"Loaded configuration with {len(config.pairs)} pairs")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            return self._create_default_config()
    
    def save(self, config: AppConfig) -> bool:
        """Save configuration to YAML file"""
        try:
            # Validate before saving
            errors = config.validate()
            if errors:
                logger.error(f"Cannot save invalid configuration: {errors}")
                return False
            
            # Convert to dictionary
            data = {
                'max_concurrent_pairs': config.max_concurrent_pairs,
                'telegram_token': config.telegram_token,
                'telegram_chat_id': config.telegram_chat_id,
                'log_level': config.log_level,
                'binance_requests_per_minute': config.binance_requests_per_minute,
                'backtest_concurrent_limit': config.backtest_concurrent_limit,
                'data_dir': config.data_dir,
                'logs_dir': config.logs_dir,
                'backtest_results_dir': config.backtest_results_dir,
                'pairs': []
            }
            
            # Convert pairs to dictionaries
            for pair in config.pairs:
                pair_dict = {
                    'symbol': pair.symbol,
                    'enabled': pair.enabled,
                    'backtest_enabled': pair.backtest_enabled,
                    'timeframe': pair.timeframe,
                    'strategy': pair.strategy,
                    'min_risk_reward': pair.min_risk_reward,
                    'fractal_left': pair.fractal_left,
                    'fractal_right': pair.fractal_right,
                    'max_risk_per_trade': pair.max_risk_per_trade
                }
                data['pairs'].append(pair_dict)
            
            # Save to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            return False
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration"""
        default_pairs = [
            PairConfig(symbol="BTCUSDT", enabled=True, backtest_enabled=False),
            PairConfig(symbol="ETHUSDT", enabled=False, backtest_enabled=True),
            PairConfig(symbol="SOLUSDT", enabled=False, backtest_enabled=False),
        ]
        
        config = AppConfig(pairs=default_pairs)
        
        # Save default config
        self.save(config)
        
        return config


def load_config(config_path: str = "config/pairs.yaml") -> AppConfig:
    """Convenience function to load configuration"""
    loader = ConfigLoader(config_path)
    return loader.load()


def save_config(config: AppConfig, config_path: str = "config/pairs.yaml") -> bool:
    """Convenience function to save configuration"""
    loader = ConfigLoader(config_path)
    return loader.save(config)
