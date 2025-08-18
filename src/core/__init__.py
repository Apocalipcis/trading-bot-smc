"""
Core modules for SMC trading bot
"""

from .exchange_gateway import ExchangeGateway, LiveExchangeGateway, BacktestExchangeGateway
from .strategy import Strategy, SMCStrategy, StrategyFactory
from .pair_manager import PairManager, PairWorker

__all__ = [
    'ExchangeGateway', 'LiveExchangeGateway', 'BacktestExchangeGateway',
    'Strategy', 'SMCStrategy', 'StrategyFactory',
    'PairManager', 'PairWorker'
]
