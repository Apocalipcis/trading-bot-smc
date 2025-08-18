#!/usr/bin/env python3
"""
New SMC Trading Bot Web Interface
FastAPI + Jinja2 + HTMX for reactive pair management
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config import load_config, save_config, AppConfig, PairConfig, PairStatus
from src.core import LiveExchangeGateway, PairManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SMC Trading Bot - Control Panel",
    description="Pair management dashboard with toggles",
    version="2.0.0"
)

# Templates and static files
templates = Jinja2Templates(directory="src/web/templates")
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global state
class BotState:
    def __init__(self):
        self.config: AppConfig = load_config()
        self.gateway: Optional[LiveExchangeGateway] = None
        self.pair_manager: Optional[PairManager] = None
        self.is_running: bool = False
        self.pair_statuses: Dict[str, PairStatus] = {}
        self.signals: List[Dict] = []

bot_state = BotState()

# Pydantic models for API
class PairToggleRequest(BaseModel):
    symbol: str
    enabled: bool

class BacktestToggleRequest(BaseModel):
    symbol: str
    backtest_enabled: bool

class AddPairRequest(BaseModel):
    symbol: str
    timeframe: str = "15m"
    strategy: str = "smc_v1"
    min_risk_reward: float = 3.0


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting SMC Trading Bot Web Interface")
    
    # Initialize exchange gateway
    bot_state.gateway = LiveExchangeGateway()
    await bot_state.gateway.__aenter__()
    
    # Initialize pair manager
    bot_state.pair_manager = PairManager(bot_state.gateway, bot_state.config)
    
    # Add callbacks
    bot_state.pair_manager.add_signal_callback(on_new_signal)
    bot_state.pair_manager.add_status_callback(on_status_update)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down SMC Trading Bot Web Interface")
    
    if bot_state.pair_manager:
        await bot_state.pair_manager.stop()
    
    if bot_state.gateway:
        await bot_state.gateway.__aexit__(None, None, None)


def on_new_signal(signal):
    """Handle new trading signal"""
    logger.info(f"New signal: {signal.symbol} {signal.direction} at {signal.entry}")
    bot_state.signals.insert(0, signal.to_dict())
    # Keep only last 50 signals
    bot_state.signals = bot_state.signals[:50]


def on_status_update(statuses: Dict[str, PairStatus]):
    """Handle status updates"""
    bot_state.pair_statuses = statuses


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "config": bot_state.config,
        "pair_statuses": bot_state.pair_statuses,
        "is_running": bot_state.is_running,
        "signals": bot_state.signals[:10],  # Last 10 signals
        "total_pairs": len(bot_state.config.pairs),
        "enabled_pairs": len([p for p in bot_state.config.pairs if p.enabled]),
        "backtest_pairs": len([p for p in bot_state.config.pairs if p.backtest_enabled])
    })


@app.get("/pairs-table", response_class=HTMLResponse)
async def pairs_table(request: Request):
    """HTMX endpoint for pairs table"""
    return templates.TemplateResponse("pairs_table.html", {
        "request": request,
        "config": bot_state.config,
        "pair_statuses": bot_state.pair_statuses,
        "is_running": bot_state.is_running
    })


@app.get("/signals-list", response_class=HTMLResponse)
async def signals_list(request: Request):
    """HTMX endpoint for signals list"""
    return templates.TemplateResponse("signals_list.html", {
        "request": request,
        "signals": bot_state.signals[:20]
    })


@app.post("/api/pairs/{symbol}/toggle-enabled")
async def toggle_pair_enabled(symbol: str, enabled: bool = Form(...)):
    """Toggle pair enabled status"""
    try:
        symbol = symbol.upper()
        
        # Find and update pair in config
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        pair_config.enabled = enabled
        
        # Save config
        save_config(bot_state.config)
        
        # Apply hot reload if bot is running
        if bot_state.pair_manager and bot_state.is_running:
            await bot_state.pair_manager.apply_config(bot_state.config)
        
        logger.info(f"Toggled {symbol} enabled: {enabled}")
        return {"success": True, "symbol": symbol, "enabled": enabled}
        
    except Exception as e:
        logger.error(f"Error toggling {symbol} enabled: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pairs/{symbol}/toggle-backtest")
async def toggle_pair_backtest(symbol: str, backtest_enabled: bool = Form(...)):
    """Toggle pair backtest status"""
    try:
        symbol = symbol.upper()
        
        # Find and update pair in config
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        pair_config.backtest_enabled = backtest_enabled
        
        # Save config
        save_config(bot_state.config)
        
        # Apply hot reload if bot is running
        if bot_state.pair_manager and bot_state.is_running:
            await bot_state.pair_manager.apply_config(bot_state.config)
        
        logger.info(f"Toggled {symbol} backtest: {backtest_enabled}")
        return {"success": True, "symbol": symbol, "backtest_enabled": backtest_enabled}
        
    except Exception as e:
        logger.error(f"Error toggling {symbol} backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pairs/add")
async def add_pair(symbol: str = Form(...), timeframe: str = Form("15m"), strategy: str = Form("smc_v1"), min_risk_reward: float = Form(3.0)):
    """Add new trading pair"""
    try:
        symbol = symbol.upper().strip()
        
        # Validate symbol
        if not symbol.endswith('USDT'):
            raise HTTPException(status_code=400, detail="Symbol must end with USDT")
        
        # Check if already exists
        if bot_state.config.get_pair_config(symbol):
            raise HTTPException(status_code=400, detail=f"Pair {symbol} already exists")
        
        # Check limit
        if len(bot_state.config.pairs) >= 20:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Maximum number of pairs reached")
        
        # Create new pair config
        new_pair = PairConfig(
            symbol=symbol,
            enabled=False,  # Start disabled
            backtest_enabled=False,
            timeframe=timeframe,
            strategy=strategy,
            min_risk_reward=min_risk_reward
        )
        
        # Add to config
        bot_state.config.pairs.append(new_pair)
        
        # Save config
        save_config(bot_state.config)
        
        # Apply hot reload if bot is running
        if bot_state.pair_manager and bot_state.is_running:
            await bot_state.pair_manager.apply_config(bot_state.config)
        
        logger.info(f"Added new pair: {symbol}")
        return {"success": True, "symbol": symbol}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding pair {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/pairs/{symbol}")
async def remove_pair(symbol: str):
    """Remove trading pair"""
    try:
        symbol = symbol.upper()
        
        # Find pair
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        # Remove from config
        bot_state.config.pairs = [p for p in bot_state.config.pairs if p.symbol != symbol]
        
        # Save config
        save_config(bot_state.config)
        
        # Apply hot reload if bot is running
        if bot_state.pair_manager and bot_state.is_running:
            await bot_state.pair_manager.apply_config(bot_state.config)
        
        logger.info(f"Removed pair: {symbol}")
        return {"success": True, "symbol": symbol}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing pair {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start")
async def start_bot(request: Request):
    """Start the trading bot"""
    try:
        if bot_state.is_running:
            return {"success": False, "message": "Bot is already running"}
        
        if not bot_state.pair_manager:
            raise HTTPException(status_code=500, detail="Pair manager not initialized")
        
        # Start pair manager
        await bot_state.pair_manager.start()
        bot_state.is_running = True
        
        logger.info("Trading bot started")
        return templates.TemplateResponse("alert.html", {
            "request": request,
            "type": "success",
            "message": "✅ Bot started successfully!"
        })
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_bot(request: Request):
    """Stop the trading bot"""
    try:
        if not bot_state.is_running:
            return {"success": False, "message": "Bot is not running"}
        
        if bot_state.pair_manager:
            await bot_state.pair_manager.stop()
        
        bot_state.is_running = False
        bot_state.pair_statuses = {}
        
        logger.info("Trading bot stopped")
        return templates.TemplateResponse("alert.html", {
            "request": request,
            "type": "info", 
            "message": "🛑 Bot stopped successfully!"
        })
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload-config")
async def reload_config():
    """Reload configuration from file"""
    try:
        # Reload config from file
        new_config = load_config()
        
        # Validate new config
        errors = new_config.validate()
        if errors:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {errors}")
        
        # Apply new config
        bot_state.config = new_config
        
        # Apply hot reload if bot is running
        if bot_state.pair_manager and bot_state.is_running:
            await bot_state.pair_manager.apply_config(bot_state.config)
        
        logger.info("Configuration reloaded successfully")
        return {"success": True, "message": "Configuration reloaded"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Get bot status"""
    return {
        "is_running": bot_state.is_running,
        "total_pairs": len(bot_state.config.pairs),
        "enabled_pairs": len([p for p in bot_state.config.pairs if p.enabled]),
        "backtest_pairs": len([p for p in bot_state.config.pairs if p.backtest_enabled]),
        "max_concurrent": bot_state.config.max_concurrent_pairs,
        "signals_count": len(bot_state.signals),
        "pair_statuses": {symbol: status.to_dict() for symbol, status in bot_state.pair_statuses.items()}
    }


@app.get("/api/backtest-results")
async def get_backtest_results():
    """Get all backtest results"""
    results = {}
    backtest_dir = Path("backtest_results")
    
    if backtest_dir.exists():
        for symbol_dir in backtest_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                latest_file = symbol_dir / "latest_results.json"
                
                if latest_file.exists():
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            import json
                            data = json.load(f)
                            
                            # Handle different JSON structures
                            if 'report' in data:
                                # New format with report wrapper
                                result_data = data['report'].copy()
                                result_data['timestamp'] = data.get('timestamp', 'Unknown')
                                results[symbol] = result_data
                                logger.info(f"Loaded backtest results for {symbol}: {result_data.get('total_trades', 0)} trades, success: {result_data.get('success', False)}")
                            else:
                                # Old format or direct data
                                results[symbol] = data
                                logger.info(f"Loaded old format results for {symbol}: {data.get('total_trades', 0)} trades")
                                
                    except Exception as e:
                        logger.error(f"Error reading backtest results for {symbol}: {e}")
                        results[symbol] = {"error": str(e)}
    
    return results


@app.get("/backtest-results", response_class=HTMLResponse)
async def backtest_results_page(request: Request):
    """Backtest results page"""
    # Get backtest results
    results = await get_backtest_results()
    
    return templates.TemplateResponse("backtest_results.html", {
        "request": request,
        "config": bot_state.config,
        "backtest_results": results,
        "is_running": bot_state.is_running
    })


@app.post("/api/pairs/{symbol}/run-backtest")
async def run_single_backtest(symbol: str, request: Request):
    """Run backtest for a specific symbol"""
    try:
        symbol = symbol.upper()
        
        # Get pair config
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        # Prepare config for backtest
        config_dict = {
            'min_risk_reward': pair_config.min_risk_reward,
            'fractal_left': pair_config.fractal_left,
            'fractal_right': pair_config.fractal_right,
            'backtest_days': 30
        }
        
        # Import and run backtest
        from src.backtest import run_backtest_async
        
        logger.info(f"Starting manual backtest for {symbol}")
        result = await run_backtest_async(symbol, config_dict)
        
        if result.get('success', False):
            message = f"✅ Backtest completed for {symbol}! Check results below."
        else:
            message = f"❌ Backtest failed for {symbol}: {result.get('error', 'Unknown error')}"
        
        return templates.TemplateResponse("alert.html", {
            "request": request,
            "type": "success" if result.get('success', False) else "error",
            "message": message
        })
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        return templates.TemplateResponse("alert.html", {
            "request": request,
            "type": "error",
            "message": f"❌ Error running backtest for {symbol}: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
