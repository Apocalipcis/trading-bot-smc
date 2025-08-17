"""
Binance USD-M Futures WebSocket data stream for real-time price data
"""
import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional
import logging

class BinanceFuturesStream:
    """Real-time data stream for Binance USD-M Futures"""
    
    def __init__(self):
        self.base_url = "wss://fstream.binance.com/ws/"
        self.connection = None
        self.is_connected = False
        self.callbacks = {}
        self.kline_data = {}  # Store recent klines
        
    async def connect(self):
        """Connect to Binance Futures WebSocket"""
        try:
            self.connection = await websockets.connect(self.base_url)
            self.is_connected = True
            logging.info("Connected to Binance Futures WebSocket")
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            self.is_connected = False
            
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.connection:
            await self.connection.close()
        self.is_connected = False
        logging.info("Disconnected from Binance Futures WebSocket")
        
    def add_callback(self, stream_name: str, callback: Callable):
        """Add callback for specific stream"""
        self.callbacks[stream_name] = callback
        
    async def subscribe_klines(self, symbol: str, intervals: List[str]):
        """
        Subscribe to kline/candlestick streams
        
        Args:
            symbol: Trading pair (e.g., 'ethusdt')
            intervals: List of intervals ['15m', '4h']
        """
        if not self.is_connected:
            await self.connect()
            
        # Create stream names
        streams = []
        for interval in intervals:
            stream_name = f"{symbol.lower()}@kline_{interval}"
            streams.append(stream_name)
            
        # Subscribe to multiple streams
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        await self.connection.send(json.dumps(subscribe_message))
        logging.info(f"Subscribed to kline streams: {streams}")
        
    async def listen(self):
        """Listen for incoming messages"""
        if not self.is_connected:
            raise Exception("Not connected to WebSocket")
            
        try:
            async for message in self.connection:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logging.error(f"Error in listen loop: {e}")
            
    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket message"""
        if 'stream' in data and 'data' in data:
            stream = data['stream']
            kline_data = data['data']
            
            if '@kline_' in stream:
                await self._handle_kline(stream, kline_data)
                
    async def _handle_kline(self, stream: str, data: Dict):
        """Handle kline/candlestick data"""
        kline = data['k']
        symbol = kline['s']
        interval = kline['i']
        is_closed = kline['x']  # True if kline is closed
        
        # Convert to our format
        kline_info = {
            'symbol': symbol,
            'interval': interval,
            'open_time': pd.to_datetime(kline['t'], unit='ms'),
            'close_time': pd.to_datetime(kline['T'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': is_closed
        }
        
        # Store kline data
        key = f"{symbol}_{interval}"
        if key not in self.kline_data:
            self.kline_data[key] = []
            
        # Update or append kline
        if self.kline_data[key] and not is_closed:
            # Update current kline
            self.kline_data[key][-1] = kline_info
        else:
            # New closed kline
            self.kline_data[key].append(kline_info)
            # Keep only last 1000 klines
            if len(self.kline_data[key]) > 1000:
                self.kline_data[key] = self.kline_data[key][-1000:]
                
        # Call registered callbacks
        if stream in self.callbacks:
            await self.callbacks[stream](kline_info)
            
    def get_recent_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Get recent klines as DataFrame"""
        key = f"{symbol}_{interval}"
        if key not in self.kline_data:
            return pd.DataFrame()
            
        klines = self.kline_data[key][-limit:]
        if not klines:
            return pd.DataFrame()
            
        df = pd.DataFrame(klines)
        df = df.rename(columns={'open_time': 'timestamp'})
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    async def get_historical_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Get historical klines via REST API"""
        import aiohttp
        
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convert types and select columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

class LiveDataManager:
    """Manages real-time data for SMC analysis"""
    
    def __init__(self, symbol: str, intervals: List[str] = ['15m', '4h']):
        self.symbol = symbol.upper()
        self.intervals = intervals
        self.stream = BinanceFuturesStream()
        self.data_callbacks = {}
        
    async def start(self):
        """Start real-time data collection"""
        # Get historical data first
        for interval in self.intervals:
            historical = await self.stream.get_historical_klines(
                self.symbol, interval, limit=500
            )
            key = f"{self.symbol}_{interval}"
            self.stream.kline_data[key] = historical.to_dict('records')
            
        # Setup callbacks
        for interval in self.intervals:
            stream_name = f"{self.symbol.lower()}@kline_{interval}"
            self.stream.add_callback(stream_name, self._on_new_kline)
            
        # Subscribe and listen
        await self.stream.subscribe_klines(self.symbol, self.intervals)
        return self.stream.listen()
        
    async def _on_new_kline(self, kline_data: Dict):
        """Handle new kline data"""
        symbol = kline_data['symbol']
        interval = kline_data['interval']
        
        # Notify callbacks
        key = f"{symbol}_{interval}"
        if key in self.data_callbacks:
            await self.data_callbacks[key](kline_data)
            
    def add_data_callback(self, symbol: str, interval: str, callback: Callable):
        """Add callback for new data"""
        key = f"{symbol}_{interval}"
        self.data_callbacks[key] = callback
        
    def get_dataframe(self, interval: str, limit: int = 500) -> pd.DataFrame:
        """Get current data as DataFrame"""
        return self.stream.get_recent_klines(self.symbol, interval, limit)
        
    async def stop(self):
        """Stop data collection"""
        await self.stream.disconnect()
