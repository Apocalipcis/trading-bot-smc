"""
Unified exchange gateway interfaces for live trading and backtesting
"""
import asyncio
import logging
import aiohttp
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from aiolimiter import AsyncLimiter

from config.models import Candle, OrderResult

logger = logging.getLogger(__name__)


class ExchangeGateway(ABC):
    """Abstract base class for exchange gateways"""
    
    @abstractmethod
    async def fetch_candles(self, symbol: str, timeframe: str, limit: int = 500) -> List[Candle]:
        """Fetch candlestick data"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, qty: float, price: Optional[float] = None) -> OrderResult:
        """Place an order (market or limit)"""
        pass
    
    @abstractmethod
    async def get_account_balance(self, asset: str = "USDT") -> float:
        """Get account balance for asset"""
        pass
    
    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Whether this is a live trading gateway"""
        pass


class LiveExchangeGateway(ExchangeGateway):
    """Live exchange gateway using Binance API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://fapi.binance.com"
        
        # Rate limiter: Binance allows 1200 requests per minute
        self.limiter = AsyncLimiter(max_rate=1200, time_period=60)
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Price cache
        self._price_cache: Dict[str, tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=5)
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Dict[str, Any]:
        """Make rate-limited HTTP request to Binance API"""
        await self._ensure_session()
        
        async with self.limiter:
            url = f"{self.base_url}{endpoint}"
            
            for attempt in range(retries):
                try:
                    async with self.session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            logger.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds")
                            await asyncio.sleep(2 ** attempt)
                        else:
                            logger.error(f"API error: {response.status} - {await response.text()}")
                            if attempt == retries - 1:
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=response.status
                                )
                except asyncio.TimeoutError:
                    logger.warning(f"Request timeout, attempt {attempt + 1}/{retries}")
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Request failed, attempt {attempt + 1}/{retries}: {e}")
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(1)
    
    async def fetch_candles(self, symbol: str, timeframe: str, limit: int = 500) -> List[Candle]:
        """Fetch candlestick data from Binance"""
        try:
            params = {
                'symbol': symbol.upper(),
                'interval': timeframe,
                'limit': min(limit, 1500)  # Binance limit
            }
            
            data = await self._make_request('/fapi/v1/klines', params)
            
            candles = []
            for kline in data:
                candle = Candle.from_binance_kline(kline)
                candles.append(candle)
            
            logger.debug(f"Fetched {len(candles)} candles for {symbol} {timeframe}")
            return candles
            
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price with caching"""
        symbol = symbol.upper()
        now = datetime.now()
        
        # Check cache
        if symbol in self._price_cache:
            price, timestamp = self._price_cache[symbol]
            if now - timestamp < self._cache_ttl:
                return price
        
        try:
            params = {'symbol': symbol}
            data = await self._make_request('/fapi/v1/ticker/price', params)
            
            price = float(data['price'])
            self._price_cache[symbol] = (price, now)
            
            return price
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            # Return cached price if available
            if symbol in self._price_cache:
                return self._price_cache[symbol][0]
            raise
    
    async def place_order(self, symbol: str, side: str, qty: float, price: Optional[float] = None) -> OrderResult:
        """Place order on Binance (requires API keys)"""
        if not self.api_key or not self.api_secret:
            return OrderResult(
                order_id="DEMO_ORDER",
                symbol=symbol,
                side=side,
                qty=qty,
                price=price or await self.get_current_price(symbol),
                status="REJECTED",
                timestamp=datetime.now(),
                is_simulation=True,
                error_message="No API credentials provided - demo mode"
            )
        
        # TODO: Implement actual order placement with proper signing
        # This would require implementing Binance API signature authentication
        
        return OrderResult(
            order_id="LIVE_ORDER_NOT_IMPLEMENTED",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price or await self.get_current_price(symbol),
            status="REJECTED",
            timestamp=datetime.now(),
            is_simulation=True,
            error_message="Live order placement not yet implemented"
        )
    
    async def get_account_balance(self, asset: str = "USDT") -> float:
        """Get account balance (requires API keys)"""
        if not self.api_key or not self.api_secret:
            logger.warning("No API credentials - returning demo balance")
            return 1000.0  # Demo balance
        
        # TODO: Implement actual balance fetching
        return 1000.0
    
    @property
    def is_live(self) -> bool:
        return True


class BacktestExchangeGateway(ExchangeGateway):
    """Backtest exchange gateway using historical data"""
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame], initial_balance: float = 10000.0):
        self.historical_data = historical_data  # symbol_timeframe -> DataFrame
        self.current_time = datetime.now()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Simulated orders
        self.orders: List[OrderResult] = []
        self.order_counter = 0
        
        # Position tracking
        self.positions: Dict[str, float] = {}  # symbol -> quantity
    
    def set_current_time(self, timestamp: datetime):
        """Set current backtest time"""
        self.current_time = timestamp
    
    async def fetch_candles(self, symbol: str, timeframe: str, limit: int = 500) -> List[Candle]:
        """Fetch candles from historical data"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.historical_data:
            logger.warning(f"No historical data for {key}")
            return []
        
        df = self.historical_data[key]
        
        # Filter data up to current time
        df_filtered = df[df.index <= self.current_time].tail(limit)
        
        candles = []
        for _, row in df_filtered.iterrows():
            candle = Candle(
                timestamp=row.name,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            candles.append(candle)
        
        return candles
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price from historical data"""
        # Try to get from 1m data first, then 15m
        for timeframe in ['1m', '15m', '1h', '4h']:
            key = f"{symbol}_{timeframe}"
            if key in self.historical_data:
                df = self.historical_data[key]
                current_data = df[df.index <= self.current_time]
                if not current_data.empty:
                    return float(current_data.iloc[-1]['close'])
        
        logger.warning(f"No price data available for {symbol} at {self.current_time}")
        return 0.0
    
    async def place_order(self, symbol: str, side: str, qty: float, price: Optional[float] = None) -> OrderResult:
        """Simulate order placement"""
        self.order_counter += 1
        order_id = f"BACKTEST_{self.order_counter:06d}"
        
        if price is None:
            price = await self.get_current_price(symbol)
        
        # Simulate order fill
        current_price = await self.get_current_price(symbol)
        
        # Simple fill logic (could be enhanced with slippage, partial fills, etc.)
        if side.upper() == 'BUY':
            cost = qty * current_price
            if cost <= self.balance:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + qty
                status = "FILLED"
                error_message = None
            else:
                status = "REJECTED"
                error_message = "Insufficient balance"
        else:  # SELL
            current_qty = self.positions.get(symbol, 0)
            if qty <= current_qty:
                self.balance += qty * current_price
                self.positions[symbol] = current_qty - qty
                status = "FILLED"
                error_message = None
            else:
                status = "REJECTED"
                error_message = "Insufficient position"
        
        order_result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            qty=qty,
            price=current_price,
            status=status,
            timestamp=self.current_time,
            is_simulation=True,
            error_message=error_message
        )
        
        self.orders.append(order_result)
        return order_result
    
    async def get_account_balance(self, asset: str = "USDT") -> float:
        """Get simulated account balance"""
        if asset.upper() == "USDT":
            return self.balance
        
        # For other assets, calculate based on positions
        if asset.upper() in self.positions:
            qty = self.positions[asset.upper()]
            if qty > 0:
                price = await self.get_current_price(f"{asset.upper()}USDT")
                return qty * price
        
        return 0.0
    
    @property
    def is_live(self) -> bool:
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get backtest performance metrics"""
        filled_orders = [o for o in self.orders if o.status == "FILLED"]
        
        total_trades = len(filled_orders)
        total_value = self.balance
        
        # Add position values
        for symbol, qty in self.positions.items():
            if qty > 0:
                # This would need current price, simplified for now
                total_value += qty * 100  # Placeholder
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': total_value,
            'total_return': (total_value - self.initial_balance) / self.initial_balance * 100,
            'total_trades': total_trades,
            'filled_orders': len(filled_orders),
            'rejected_orders': len(self.orders) - len(filled_orders)
        }
