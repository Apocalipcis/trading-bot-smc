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
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Pydantic models
class BotConfig(BaseModel):
    symbols: List[str] = ["ETHUSDT"]  # Support multiple symbols
    min_risk_reward: float = 3.0
    fractal_left: int = 2
    fractal_right: int = 2
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
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
        
    async def start_engines(self):
        """Start SMC engines for all configured symbols"""
        if self.is_running:
            await self.stop_engines()
            
        try:
            # Convert config to dict
            config_dict = {
                'min_risk_reward': self.config.min_risk_reward,
                'fractal_left': self.config.fractal_left,
                'fractal_right': self.config.fractal_right,
                'status_check_interval': self.config.status_check_interval
            }
            
            # Create Telegram client if configured
            if self.config.telegram_token and self.config.telegram_chat_id:
                self.telegram_client = TelegramClient(
                    self.config.telegram_token,
                    self.config.telegram_chat_id
                )
            
            # Create and start engines for each symbol
            for symbol in self.config.symbols:
                if symbol not in self.engines:
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
                await engine.stop()
        
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
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections.remove(ws)

# Global bot manager instance
bot_manager = BotManager()

# API Routes
@app.get("/")
async def root():
    """Redirect to frontend"""
    from fastapi.responses import FileResponse
    return FileResponse(frontend_path / "index.html")

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

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    bot_manager.websocket_connections.append(websocket)
    
    try:
        # Send initial data
        await websocket.send_text(json.dumps({
            "type": "initial_data",
            "data": {
                "signals": bot_manager.signals,
                "config": bot_manager.config.dict(),
                "status": {
                    "is_running": bot_manager.is_running,
                    "current_price": bot_manager.current_price,
                    "htf_bias": bot_manager.htf_bias
                }
            }
        }))
        
        # Keep connection alive
        while True:
            # Wait for ping from client or send heartbeat
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in bot_manager.websocket_connections:
            bot_manager.websocket_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
