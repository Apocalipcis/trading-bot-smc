#!/usr/bin/env python3
"""
SMC Trading Bot - Web Interface Backend
FastAPI server with WebSocket support for real-time signal monitoring
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path to import SMC modules
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.live_smc_engine import LiveSMCEngine
from src.telegram_client import TelegramClient
from src.pre_trade_backtest import run_symbol_backtest, run_multiple_backtests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SMC Trading Bot - Web Interface",
    description="Real-time Smart Money Concepts signal monitoring",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    logger.error(f"Frontend path does not exist: {frontend_path}")
    # Try alternative path
    alt_frontend_path = Path(__file__).parent.parent.parent / "web" / "frontend"
    if alt_frontend_path.exists():
        app.mount("/static", StaticFiles(directory=alt_frontend_path), name="static")
        frontend_path = alt_frontend_path
    else:
        logger.error(f"Alternative frontend path also does not exist: {alt_frontend_path}")

# Pydantic models
class BotConfig(BaseModel):
    symbols: List[str] = ["BTCUSDT"]  # Support multiple symbols
    min_risk_reward: float = 3.0
    fractal_left: int = 2
    fractal_right: int = 2
    telegram_token: Optional[str] = "7834170834:AAG1OxqOxCjxFP38oUW-TAPidA7CkfV2c3c"
    telegram_chat_id: Optional[str] = "333744879"
    status_check_interval: int = 45

class SignalResponse(BaseModel):
    timestamp: str
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    rr: float
    htf_bias: str
    fvg_confluence: bool
    confidence: str
    status: Optional[str] = None

# Global state
class BotManager:
    def __init__(self):
        self.engines: Dict[str, LiveSMCEngine] = {}  # Symbol -> Engine mapping
        self.telegram_client: Optional[TelegramClient] = None
        self.config: BotConfig = BotConfig()
        self.signals: List[Dict] = []
        self.market_data: Dict[str, Dict] = {}  # Symbol -> {price, bias, etc}
        self.is_running: bool = False
        self.websocket_connections: List[WebSocket] = []
        self.backtest_results: Dict[str, Dict] = {}  # Symbol -> backtest results
        
    async def start_engines(self, run_backtest_first: bool = False):
        """Start SMC engines for all configured symbols"""
        if self.is_running:
            await self.stop_engines()
            
        try:
            # Run backtest first if requested
            if run_backtest_first:
                logger.info("Running pre-trade backtests...")
                config_dict = {
                    'min_risk_reward': self.config.min_risk_reward,
                    'fractal_left': self.config.fractal_left,
                    'fractal_right': self.config.fractal_right,
                    'backtest_days': 30
                }
                
                backtest_results = await asyncio.create_task(
                    asyncio.to_thread(run_multiple_backtests, self.config.symbols, config_dict)
                )
                
                # Log backtest results
                for symbol, result in backtest_results.items():
                    if result.get('success', False):
                        logger.info(f"Backtest for {symbol}: {result.get('win_rate', 0):.1f}% win rate, {result.get('profit_factor', 0):.2f} profit factor")
                    else:
                        logger.warning(f"Backtest failed for {symbol}: {result.get('error', 'Unknown error')}")
                
                # Store backtest results for display
                self.backtest_results = backtest_results
            # Convert config to dict
            config_dict = {
                'min_risk_reward': self.config.min_risk_reward,
                'fractal_left': self.config.fractal_left,
                'fractal_right': self.config.fractal_right,
                'status_check_interval': self.config.status_check_interval
            }
            
            # Create Telegram client if configured
            if self.config.telegram_token and self.config.telegram_chat_id:
                try:
                    self.telegram_client = TelegramClient(
                        self.config.telegram_token,
                        self.config.telegram_chat_id
                    )
                except Exception as e:
                    logger.error(f"Failed to create Telegram client: {e}")
                    self.telegram_client = None
            
            # Create and start engines for each symbol
            for symbol in self.config.symbols:
                if symbol not in self.engines:
                    try:
                        engine = LiveSMCEngine(symbol, config_dict)
                        engine.add_signal_callback(self._on_new_signal)
                        engine.add_update_callback(lambda tf, data, sym=symbol: self._on_data_update(sym, tf, data))
                        
                        # Start engine in background
                        asyncio.create_task(engine.start())
                        self.engines[symbol] = engine
                        
                        # Initialize market data for this symbol
                        self.market_data[symbol] = {
                            'current_price': 0.0,
                            'htf_bias': 'neutral',
                            'status': 'starting'
                        }
                        logger.info(f"Started engine for {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to start engine for {symbol}: {e}")
                        continue
            
            self.is_running = True
            
            logger.info(f"SMC Engines started for symbols: {', '.join(self.config.symbols)}")
            await self._broadcast_status()
            
        except Exception as e:
            logger.error(f"Failed to start engines: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start engines: {e}")
    
    async def stop_engines(self):
        """Stop all SMC engines"""
        for symbol, engine in self.engines.items():
            if engine:
                try:
                    await engine.stop()
                    logger.info(f"Stopped engine for {symbol}")
                except Exception as e:
                    logger.error(f"Error stopping engine for {symbol}: {e}")
        
        self.engines.clear()
        self.market_data.clear()
        self.is_running = False
        logger.info("All SMC Engines stopped")
        await self._broadcast_status()
    
    async def _on_new_signal(self, signal: Dict):
        """Handle new signal from engine"""
        # Add timestamp string for JSON serialization
        signal_copy = signal.copy()
        signal_copy['timestamp_str'] = signal['timestamp'].isoformat()
        # Symbol should already be in the signal from the engine
        
        # Store signal
        self.signals.append(signal_copy)
        
        # Keep only last 50 signals
        if len(self.signals) > 50:
            self.signals = self.signals[-50:]
        
        # Send Telegram notification
        if self.telegram_client:
            try:
                await asyncio.create_task(
                    asyncio.to_thread(
                        self.telegram_client.send_signal_notification, 
                        signal_copy
                    )
                )
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")
        
        # Broadcast to all connected WebSocket clients
        await self._broadcast_signal(signal_copy)
        
    async def _on_data_update(self, symbol: str, timeframe: str, data: Dict):
        """Handle data update from engine"""
        if symbol in self.engines:
            engine = self.engines[symbol]
            status = engine.get_status()
            
            # Update market data for this symbol
            self.market_data[symbol] = {
                'current_price': status.get('current_price', 0.0) or 0.0,
                'htf_bias': status.get('htf_bias', 'neutral'),
                'status': 'running' if status.get('websocket_connected', False) else 'connecting'
            }
            
            # Broadcast updated status
            await self._broadcast_status()
    
    async def _broadcast_signal(self, signal: Dict):
        """Broadcast new signal to all connected clients"""
        message = {
            "type": "new_signal",
            "data": signal
        }
        await self._broadcast_message(message)
    
    async def _broadcast_status(self):
        """Broadcast current status to all connected clients"""
        status_data = {
            "is_running": self.is_running,
            "symbols": self.config.symbols,
            "market_data": self.market_data,
            "total_signals": len(self.signals),
            "engines_status": {
                symbol: engine.get_status() if engine else {}
                for symbol, engine in self.engines.items()
            }
        }
        
        message = {
            "type": "status_update", 
            "data": status_data
        }
        await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: Dict):
        """Broadcast message to all connected WebSocket clients"""
        if self.websocket_connections:
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.debug(f"WebSocket send failed: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                if ws in self.websocket_connections:
                    self.websocket_connections.remove(ws)

# Global bot manager instance
bot_manager = BotManager()

# API Routes
@app.get("/")
async def root():
    """Serve main page"""
    from fastapi.responses import FileResponse
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        from fastapi.responses import HTMLResponse
        return HTMLResponse("""
        <html><body>
        <h1>SMC Trading Bot - Web Interface</h1>
        <p>Error: Frontend files not found</p>
        <p>Frontend path: {}</p>
        <p>Index file: {}</p>
        <p><a href="/api/status">API Status</a></p>
        </body></html>
        """.format(frontend_path, index_path))

@app.get("/api/status")
async def get_status():
    """Get current bot status"""
    return {
        "is_running": bot_manager.is_running,
        "config": bot_manager.config.dict(),
        "signals_count": len(bot_manager.signals),
        "market_data": bot_manager.market_data,
        "engines_count": len(bot_manager.engines)
    }

@app.get("/api/signals")
async def get_signals():
    """Get all signals"""
    return {"signals": bot_manager.signals}

@app.post("/api/config")
async def update_config(config: BotConfig):
    """Update bot configuration"""
    bot_manager.config = config
    logger.info(f"Config updated: {config.dict()}")
    return {"message": "Config updated successfully"}

@app.post("/api/start")
async def start_bot():
    """Start the trading bot"""
    await bot_manager.start_engines()
    return {"message": "Bots started successfully"}

@app.post("/api/stop") 
async def stop_bot():
    """Stop the trading bot"""
    await bot_manager.stop_engines()
    return {"message": "Bots stopped successfully"}

@app.delete("/api/signals")
async def clear_signals():
    """Clear all signals"""
    bot_manager.signals.clear()
    await bot_manager._broadcast_message({
        "type": "signals_cleared",
        "data": {}
    })
    return {"message": "Signals cleared"}

@app.post("/api/symbols/add")
async def add_symbol(symbol: str):
    """Add a new trading symbol"""
    symbol = symbol.upper().strip()
    
    if symbol not in bot_manager.config.symbols:
        bot_manager.config.symbols.append(symbol)
        
        # If bot is running, start engine for new symbol
        if bot_manager.is_running:
            config_dict = {
                'min_risk_reward': bot_manager.config.min_risk_reward,
                'fractal_left': bot_manager.config.fractal_left,
                'fractal_right': bot_manager.config.fractal_right,
                'status_check_interval': bot_manager.config.status_check_interval
            }
            
            engine = LiveSMCEngine(symbol, config_dict)
            engine.add_signal_callback(bot_manager._on_new_signal)
            engine.add_update_callback(lambda tf, data, sym=symbol: bot_manager._on_data_update(sym, tf, data))
            
            # Start engine in background
            asyncio.create_task(engine.start())
            bot_manager.engines[symbol] = engine
            
            # Initialize market data
            bot_manager.market_data[symbol] = {
                'current_price': 0.0,
                'htf_bias': 'neutral',
                'status': 'starting'
            }
        
        await bot_manager._broadcast_status()
        logger.info(f"Added symbol: {symbol}")
        return {"message": f"Symbol {symbol} added successfully"}
    else:
        return {"message": f"Symbol {symbol} already exists"}

@app.delete("/api/symbols/{symbol}")
async def remove_symbol(symbol: str):
    """Remove a trading symbol"""
    symbol = symbol.upper().strip()
    
    if symbol in bot_manager.config.symbols:
        # Don't allow removing all symbols
        if len(bot_manager.config.symbols) <= 1:
            return {"message": "Cannot remove the last symbol"}
        
        bot_manager.config.symbols.remove(symbol)
        
        # Stop and remove engine if running
        if symbol in bot_manager.engines:
            engine = bot_manager.engines[symbol]
            await engine.stop()
            del bot_manager.engines[symbol]
        
        # Remove market data
        if symbol in bot_manager.market_data:
            del bot_manager.market_data[symbol]
        
        await bot_manager._broadcast_status()
        logger.info(f"Removed symbol: {symbol}")
        return {"message": f"Symbol {symbol} removed successfully"}
    else:
        return {"message": f"Symbol {symbol} not found"}

@app.post("/api/test-alert")
async def send_test_alert(signal: dict):
    """Send a test Telegram alert"""
    try:
        if bot_manager.telegram_client:
            await asyncio.create_task(
                asyncio.to_thread(
                    bot_manager.telegram_client.send_signal_notification, 
                    signal
                )
            )
            logger.info("Test alert sent successfully")
            return {"message": "Test alert sent to Telegram"}
        else:
            return {"message": "Telegram client not configured", "error": True}
    except Exception as e:
        logger.error(f"Failed to send test alert: {e}")
        return {"message": f"Failed to send test alert: {e}", "error": True}

@app.post("/api/backtest/{symbol}")
async def run_backtest(symbol: str):
    """Run backtest for a specific symbol"""
    try:
        symbol = symbol.upper().strip()
        
        # Get current config
        config_dict = {
            'min_risk_reward': bot_manager.config.min_risk_reward,
            'fractal_left': bot_manager.config.fractal_left,
            'fractal_right': bot_manager.config.fractal_right,
            'backtest_days': 30
        }
        
        logger.info(f"Running backtest for {symbol}")
        
        # Run backtest
        result = await asyncio.create_task(
            asyncio.to_thread(run_symbol_backtest, symbol, config_dict)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

@app.get("/api/backtest/{symbol}/results")
async def get_backtest_results(symbol: str):
    """Get latest backtest results for a symbol"""
    try:
        symbol = symbol.upper().strip()
        
        # Import here to avoid circular imports
        from src.pre_trade_backtest import PreTradeBacktest
        
        backtest = PreTradeBacktest(symbol, {})
        results = backtest.get_latest_results()
        
        if results:
            return results
        else:
            return {"message": f"No backtest results found for {symbol}"}
            
    except Exception as e:
        logger.error(f"Failed to get backtest results for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {e}")

@app.post("/api/start-with-backtest")
async def start_with_backtest():
    """Start trading with backtest for all symbols"""
    try:
        results = {}
        
        # Run backtests for all configured symbols
        config_dict = {
            'min_risk_reward': bot_manager.config.min_risk_reward,
            'fractal_left': bot_manager.config.fractal_left,
            'fractal_right': bot_manager.config.fractal_right,
            'backtest_days': 30
        }
        
        logger.info("Running backtests for all symbols")
        
        # Run backtests
        backtest_results = await asyncio.create_task(
            asyncio.to_thread(run_multiple_backtests, bot_manager.config.symbols, config_dict)
        )
        
        # Check if any backtests failed
        failed_symbols = [sym for sym, res in backtest_results.items() if not res.get('success', False)]
        
        if failed_symbols:
            return {
                "success": False,
                "message": f"Backtests failed for: {', '.join(failed_symbols)}",
                "results": backtest_results
            }
        
        # Start engines if all backtests passed
        await bot_manager.start_engines()
        
        return {
            "success": True,
            "message": "Trading started successfully after backtests",
            "backtest_results": backtest_results
        }
        
    except Exception as e:
        logger.error(f"Failed to start with backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start with backtest: {e}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    try:
        await websocket.accept()
        bot_manager.websocket_connections.append(websocket)
        logger.info("WebSocket client connected")
        
        # Send initial data
        try:
            initial_data = {
                "type": "initial_data",
                "data": {
                    "signals": bot_manager.signals,
                    "config": bot_manager.config.dict(),
                    "status": {
                        "is_running": bot_manager.is_running,
                        "symbols": bot_manager.config.symbols,
                        "market_data": bot_manager.market_data,
                        "engines_count": len(bot_manager.engines)
                    }
                }
            }
            await websocket.send_text(json.dumps(initial_data))
        except Exception as e:
            logger.error(f"Failed to send initial data: {e}")
        
        # Keep connection alive
        while True:
            try:
                # Wait for ping from client or send heartbeat
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back any received message as heartbeat
                await websocket.send_text(json.dumps({"type": "pong", "data": "alive"}))
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                except Exception:
                    break  # Connection is dead
            except Exception as e:
                logger.debug(f"WebSocket receive error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in bot_manager.websocket_connections:
            bot_manager.websocket_connections.remove(websocket)
            logger.info("WebSocket client removed from connections")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
