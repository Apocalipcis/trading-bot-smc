"""
Configuration models for SMC trading bot
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class PairConfig:
    """Configuration for a single trading pair"""
    symbol: str
    enabled: bool = False
    backtest_enabled: bool = False
    timeframe: str = "15m"
    strategy: str = "smc_v1"
    min_risk_reward: float = 3.0
    fractal_left: int = 2
    fractal_right: int = 2
    max_risk_per_trade: float = 0.02
    
    def __post_init__(self):
        self.symbol = self.symbol.upper()


@dataclass
class AppConfig:
    """Main application configuration"""
    max_concurrent_pairs: int = 10
    pairs: List[PairConfig] = field(default_factory=list)
    
    # Global settings
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    log_level: str = "INFO"
    
    # Rate limiting
    binance_requests_per_minute: int = 1200
    backtest_concurrent_limit: int = 2
    
    # Directories
    data_dir: str = "data"
    logs_dir: str = "logs"
    backtest_results_dir: str = "backtest_results"
    
    def get_enabled_pairs(self) -> List[PairConfig]:
        """Get list of enabled pairs"""
        return [pair for pair in self.pairs if pair.enabled]
    
    def get_backtest_enabled_pairs(self) -> List[PairConfig]:
        """Get list of pairs with backtest enabled"""
        return [pair for pair in self.pairs if pair.backtest_enabled]
    
    def get_pair_config(self, symbol: str) -> Optional[PairConfig]:
        """Get configuration for specific symbol"""
        symbol = symbol.upper()
        for pair in self.pairs:
            if pair.symbol == symbol:
                return pair
        return None
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check concurrent pairs limit
        enabled_count = len(self.get_enabled_pairs())
        if enabled_count > self.max_concurrent_pairs:
            errors.append(f"Too many enabled pairs: {enabled_count} > {self.max_concurrent_pairs}")
        
        # Check for duplicate symbols
        symbols = [pair.symbol for pair in self.pairs]
        if len(symbols) != len(set(symbols)):
            errors.append("Duplicate symbols found in configuration")
        
        # Validate individual pair configs
        for pair in self.pairs:
            if not pair.symbol or not pair.symbol.endswith('USDT'):
                errors.append(f"Invalid symbol: {pair.symbol}")
            
            if pair.min_risk_reward < 1.0:
                errors.append(f"Risk/reward too low for {pair.symbol}: {pair.min_risk_reward}")
        
        return errors


@dataclass
class Candle:
    """Unified candle data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_binance_kline(cls, kline_data: List) -> 'Candle':
        """Create Candle from Binance kline data"""
        return cls(
            timestamp=datetime.fromtimestamp(kline_data[0] / 1000),
            open=float(kline_data[1]),
            high=float(kline_data[2]),
            low=float(kline_data[3]),
            close=float(kline_data[4]),
            volume=float(kline_data[5])
        )


@dataclass
class OrderResult:
    """Unified order result structure"""
    order_id: str
    symbol: str
    side: str  # 'BUY' / 'SELL'
    qty: float
    price: float
    status: str  # 'FILLED' / 'REJECTED' / 'PENDING'
    timestamp: datetime
    is_simulation: bool = False
    error_message: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if order was successful"""
        return self.status == 'FILLED' and self.error_message is None


@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    direction: str  # 'LONG' / 'SHORT'
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    htf_bias: str
    fvg_confluence: bool
    confidence: str  # 'low', 'medium', 'high'
    strategy: str = "smc_v1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'entry': self.entry,
            'sl': self.stop_loss,
            'tp': self.take_profit,
            'rr': self.risk_reward,
            'htf_bias': self.htf_bias,
            'fvg_confluence': self.fvg_confluence,
            'confidence': self.confidence,
            'strategy': self.strategy
        }


@dataclass
class PairStatus:
    """Runtime status of a trading pair"""
    symbol: str
    enabled: bool
    backtest_enabled: bool
    status: str  # 'starting', 'running', 'error', 'stopped', 'queued'
    current_price: Optional[float] = None
    htf_bias: str = 'neutral'
    active_signals: int = 0
    last_signal_time: Optional[datetime] = None
    last_error: Optional[str] = None
    websocket_connected: bool = False
    backtest_running: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'symbol': self.symbol,
            'enabled': self.enabled,
            'backtest_enabled': self.backtest_enabled,
            'status': self.status,
            'current_price': self.current_price,
            'htf_bias': self.htf_bias,
            'active_signals': self.active_signals,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'last_error': self.last_error,
            'websocket_connected': self.websocket_connected,
            'backtest_running': self.backtest_running
        }
