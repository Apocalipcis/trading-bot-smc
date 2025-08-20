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
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config import load_config, save_config, AppConfig, PairConfig, PairStatus
from src.core import LiveExchangeGateway, PairManager
from src.pre_trade_backtest import run_symbol_backtest
from src.data_downloader import download_crypto_data
from src.telegram_client import TelegramClient
from src.data_loader import prepare_data, validate_timeframe_alignment
from src.signal_generator import generate_signals
from src.backtester import BacktestValidator
import os
import glob
from datetime import datetime, timedelta

# Load environment variables from .env file
def _load_dotenv(path: str = ".env") -> None:
    """Lightweight .env loader (no external deps). Sets os.environ if not set.

    Supports simple lines: KEY=VALUE, ignores comments and empty lines.
    Strips surrounding single/double quotes from VALUE.
    """
    try:
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and (key not in os.environ):
                    os.environ[key] = val
    except Exception:
        # Fail silently; env loading is best-effort
        pass

# Load .env file
_load_dotenv()



# Configure logging
try:
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'app.log')
        ]
    )
except Exception as e:
    # Fallback to basic logging if file logging fails
    print(f"Warning: Could not set up file logging: {e}")
    logging.basicConfig(level=logging.INFO)
    
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SMC Trading Bot - Control Panel",
    description="Pair management dashboard with toggles",
    version="2.0.0"
)

# Templates and static files
# Get proper paths relative to project root
template_dir = project_root / "src" / "web" / "templates"
static_dir = project_root / "src" / "web" / "static"

templates = Jinja2Templates(directory=str(template_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state
class BotState:
    def __init__(self):
        self.config: AppConfig = load_config()
        self.gateway: Optional[LiveExchangeGateway] = None
        self.pair_manager: Optional[PairManager] = None
        self.is_running: bool = False
        self.pair_statuses: Dict[str, PairStatus] = {}
        self.signals: List[Dict] = []
        self.telegram_client = None
        self.telegram_enabled: bool = False

bot_state = BotState()

# Initialize Telegram if configured
def init_telegram():
    """Initialize Telegram client if configured"""
    try:
        # Use environment variables or config values
        token = os.getenv('TELEGRAM_BOT_TOKEN') or bot_state.config.telegram_token
        chat_id = os.getenv('TELEGRAM_CHAT_ID') or bot_state.config.telegram_chat_id
        
        if token and chat_id:
            bot_state.telegram_client = TelegramClient(token, chat_id)
            # Test connection
            if bot_state.telegram_client.test_connection():
                bot_state.telegram_enabled = True
                logger.info("Telegram client initialized successfully")
            else:
                bot_state.telegram_enabled = False
                logger.warning("Telegram client failed connection test")
        else:
            bot_state.telegram_enabled = False
            logger.info("Telegram not configured (missing token or chat_id)")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram: {e}")
        bot_state.telegram_enabled = False

# Initialize Telegram on startup
init_telegram()

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
        "pairs": bot_state.config.pairs,  # Add this for the template
        "pair_statuses": bot_state.pair_statuses,
        "is_running": bot_state.is_running,
        "signals": bot_state.signals[:10],  # Last 10 signals
        "total_pairs": len(bot_state.config.pairs),
        "enabled_pairs": len([p for p in bot_state.config.pairs if p.enabled]),
        "backtest_pairs": len([p for p in bot_state.config.pairs if p.backtest_enabled]),
        "telegram_enabled": bot_state.telegram_enabled
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
        
        # Return HTML fragment instead of JSON for HTMX
        return HTMLResponse(f"""
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="bi bi-check-circle"></i>
                <strong>Success!</strong> {symbol} trading {'enabled' if enabled else 'disabled'}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)
        
    except Exception as e:
        logger.error(f"Error toggling {symbol} enabled: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to toggle {symbol}: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


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
        
        # Return HTML fragment instead of JSON for HTMX
        return HTMLResponse(f"""
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="bi bi-check-circle"></i>
                <strong>Success!</strong> {symbol} backtest {'enabled' if backtest_enabled else 'disabled'}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)
        
    except Exception as e:
        logger.error(f"Error toggling {symbol} backtest: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to toggle {symbol} backtest: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.post("/api/pairs/{symbol}/test-signal")
async def generate_test_signal(symbol: str):
    """Generate a test signal for the specified pair"""
    try:
        symbol = symbol.upper()
        
        # Find pair in config
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        # Create a test signal
        from datetime import datetime
        from config.models import Signal
        
        # Generate random direction and price for testing
        import random
        direction = random.choice(['LONG', 'SHORT'])
        current_price = 100.0  # Default price for testing
        
        # Create test signal
        test_signal = Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            entry=current_price,
            stop_loss=current_price * 0.95 if direction == 'LONG' else current_price * 1.05,
            take_profit=current_price * 1.15 if direction == 'LONG' else current_price * 0.85,
            risk_reward=3.0,
            htf_bias='bull' if direction == 'LONG' else 'bear',
            fvg_confluence=True,
            confidence='high',
            strategy='test'
        )
        
        # Add to signals list
        bot_state.signals.insert(0, test_signal.to_dict())
        # Keep only last 50 signals
        bot_state.signals = bot_state.signals[:50]
        
        logger.info(f"Generated test signal for {symbol}: {direction} at {current_price}")
        
        # Return success message with signal refresh trigger
        return HTMLResponse(f"""
            <div class="alert alert-info alert-dismissible fade show" role="alert">
                <i class="bi bi-lightning"></i>
                <strong>Test Signal Generated!</strong> {symbol} {direction} at ${current_price:.2f}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
            <script>
                // Trigger signals list refresh after a short delay
                setTimeout(() => {{
                    htmx.trigger('#signals-list', 'load');
                }}, 500);
            </script>
        """)
        
    except Exception as e:
        logger.error(f"Error generating test signal for {symbol}: {e}")
        return HTMLResponse(f"""
            <div div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to generate test signal for {symbol}: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


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
        return JSONResponse({"success": True, "symbol": symbol})
        
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
        return JSONResponse({"success": True, "symbol": symbol})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing pair {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pairs/{symbol}/run-backtest")
async def run_pair_backtest(symbol: str):
    """Run backtest for the specified pair"""
    try:
        symbol = symbol.upper()
        
        # Find pair in config
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        # Check if backtest is already running
        if bot_state.pair_statuses.get(symbol) and bot_state.pair_statuses[symbol].backtest_running:
            return HTMLResponse(f"""
                <div class="alert alert-warning alert-dismissible fade show" role="alert">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Warning!</strong> Backtest for {symbol} is already running
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        
        # Import backtest function
        from src.pre_trade_backtest import run_symbol_backtest
        
        # Build config dict for backtest
        config_dict = {
            'min_risk_reward': pair_config.min_risk_reward,
            'fractal_left': pair_config.fractal_left,
            'fractal_right': pair_config.fractal_right,
            'backtest_days': 30,
            'ltf_timeframe': getattr(pair_config, 'timeframe', '15m'),
            'htf_timeframe': '4h'
        }
        
        # Run backtest in background thread and reflect running status in UI
        import asyncio
        async def _run_and_flag():
            try:
                # Mark as running in UI if we have a status entry
                status = bot_state.pair_statuses.get(symbol)
                if status:
                    status.backtest_running = True
                await asyncio.to_thread(run_symbol_backtest, symbol, config_dict)
            finally:
                status = bot_state.pair_statuses.get(symbol)
                if status:
                    status.backtest_running = False
        
        asyncio.create_task(_run_and_flag())
        
        logger.info(f"Started backtest for {symbol}")
        
        # Return success message
        return HTMLResponse(f"""
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="bi bi-check-circle"></i>
                <strong>Success!</strong> Started backtest for {symbol}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
            <script>
                // Trigger pairs table refresh after a short delay
                setTimeout(() => {{
                    htmx.trigger('#pairs-table', 'load');
                }}, 500);
            </script>
        """)
        
    except Exception as e:
        logger.error(f"Error starting backtest for {symbol}: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to start backtest for {symbol}: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.post("/api/pairs/{symbol}/toggle-backtest-status")
async def toggle_pair_backtest_status(symbol: str):
    """Toggle backtest enabled status for a pair"""
    try:
        symbol = symbol.upper()
        
        # Find and update pair in config
        pair_config = bot_state.config.get_pair_config(symbol)
        if not pair_config:
            raise HTTPException(status_code=404, detail=f"Pair {symbol} not found")
        
        # Toggle backtest status
        pair_config.backtest_enabled = not pair_config.backtest_enabled
        
        # Save config
        save_config(bot_state.config)
        
        # Apply hot reload if bot is running
        if bot_state.pair_manager and bot_state.is_running:
            await bot_state.pair_manager.apply_config(bot_state.config)
        
        status_text = "enabled" if pair_config.backtest_enabled else "disabled"
        logger.info(f"Toggled {symbol} backtest: {status_text}")
        
        # Return HTML fragment instead of JSON for HTMX
        return HTMLResponse(f"""
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="bi bi-check-circle"></i>
                <strong>Success!</strong> {symbol} backtest {status_text}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
            <script>
                // Trigger pairs table refresh after a short delay
                setTimeout(() => {{
                    htmx.trigger('#pairs-table', 'load');
                }}, 500);
            </script>
        """)
        
    except Exception as e:
        logger.error(f"Error toggling {symbol} backtest status: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to toggle {symbol} backtest: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


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


# New Backtest API endpoints
@app.post("/api/backtests/fetch")
async def fetch_backtest_data(
    request: Request,
    symbol: str = Form(...),
    timeframes: List[str] = Form([]),
    days: int = Form(30),
    until: Optional[str] = Form(None)
):
    """Fetch historical data for backtesting"""
    try:
        symbol = symbol.upper().strip()
        
        # If no timeframes selected, use default
        if not timeframes:
            timeframes = ["15m", "4h"]
        
        # Convert days to start/end dates
        if until:
            end_date = datetime.strptime(until, "%Y-%m-%d")
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {symbol} data for timeframes {timeframes} from {start_date.date()} to {end_date.date()}")
        
        # Download data for each timeframe
        downloaded_files = []
        errors = []
        
        for timeframe in timeframes:
            try:
                # Use the data downloader for each timeframe
                from src.data_downloader import download_specific_timeframe
                filename = await download_specific_timeframe(symbol, timeframe, days, end_date)
                downloaded_files.append(f"{timeframe}: {filename}")
                logger.info(f"Downloaded {symbol} {timeframe} data: {filename}")
            except Exception as e:
                error_msg = f"{timeframe}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to download {symbol} {timeframe}: {e}")
        
        # Prepare response
        if downloaded_files and not errors:
            # All successful
            files_list = "<br>".join([f"• {f}" for f in downloaded_files])
            return HTMLResponse(f"""
                <div class="alert alert-success alert-dismissible fade show">
                    <i class="bi bi-check-circle"></i>
                    <strong>Success!</strong> Downloaded {symbol} data for {len(timeframes)} timeframes ({days} days).
                    <br><small>{files_list}</small>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        elif downloaded_files and errors:
            # Partial success
            files_list = "<br>".join([f"✓ {f}" for f in downloaded_files])
            errors_list = "<br>".join([f"✗ {e}" for e in errors])
            return HTMLResponse(f"""
                <div class="alert alert-warning alert-dismissible fade show">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Partial Success!</strong> Some downloads completed, some failed.
                    <br><strong>Downloaded:</strong><br><small>{files_list}</small>
                    <br><strong>Errors:</strong><br><small>{errors_list}</small>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        else:
            # All failed
            errors_list = "<br>".join([f"• {e}" for e in errors])
            return HTMLResponse(f"""
                <div class="alert alert-danger alert-dismissible fade show">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Error!</strong> Failed to download {symbol} data for all timeframes.
                    <br><small>{errors_list}</small>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
            
    except Exception as e:
        logger.error(f"Fetch data error: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.post("/api/backtests/run")
async def run_backtest_analysis(
    symbol: str = Form(...),
    timeframe: str = Form("15m"),
    days: int = Form(30),
    min_rr: float = Form(3.0),
    require_fvg: bool = Form(False)
):
    """Run backtest analysis on cached data"""
    try:
        symbol = symbol.upper().strip()
        
        # Prepare config for backtest
        config_dict = {
            'min_risk_reward': min_rr,
            'fractal_left': 2,
            'fractal_right': 2,
            'require_fvg_confluence': require_fvg,
            'backtest_days': days
        }
        
        logger.info(f"Running backtest for {symbol} with RR={min_rr}")
        
        # Run backtest using existing module
        result = await asyncio.create_task(
            asyncio.to_thread(run_symbol_backtest, symbol, config_dict)
        )
        
        if result.get('success', False):
            total_trades = result.get('total_trades', 0)
            win_rate = result.get('win_rate', 0)
            profit_factor = result.get('profit_factor', 0)
            total_pnl = result.get('total_pnl', 0)
            
            return HTMLResponse(f"""
                <div class="alert alert-success alert-dismissible fade show">
                    <i class="bi bi-chart-line"></i>
                    <strong>Backtest Complete!</strong> {symbol} - {total_trades} trades
                    <br><small>Win Rate: {win_rate:.1f}% | P&L: ${total_pnl:.2f} | PF: {profit_factor:.2f}</small>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        else:
            error_msg = result.get('error', 'Unknown error')
            return HTMLResponse(f"""
                <div class="alert alert-warning alert-dismissible fade show">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Backtest Failed!</strong> {symbol}: {error_msg}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
            
    except Exception as e:
        logger.error(f"Backtest run error: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.delete("/api/backtests/results/{symbol}")
async def delete_backtest_results(symbol: str):
    """Delete backtest results for a symbol"""
    try:
        symbol = symbol.upper()
        backtest_dir = Path("backtest_results") / symbol
        
        if not backtest_dir.exists():
            raise HTTPException(status_code=404, detail=f"Backtest results for {symbol} not found")
        
        # Remove entire directory (this will delete both JSON and CSV results)
        import shutil
        shutil.rmtree(backtest_dir)
        
        logger.info(f"Deleted backtest results for {symbol}")
        return HTMLResponse(f"""
            <div class="alert alert-success alert-dismissible fade show">
                <i class="bi bi-check-circle"></i>
                <strong>Success!</strong> Deleted backtest results for {symbol}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)
        
    except Exception as e:
        logger.error(f"Error deleting backtest results for {symbol}: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to delete {symbol}: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.delete("/api/backtests/data/{filename}")
async def delete_cached_data(filename: str):
    """Delete cached data file"""
    try:
        data_file = Path("data") / filename
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Remove file
        data_file.unlink()
        
        logger.info(f"Deleted cached data file: {filename}")
        return HTMLResponse(f"""
            <div class="alert alert-success alert-dismissible fade show">
                <i class="bi bi-check-circle"></i>
                <strong>Success!</strong> Deleted file {filename}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)
        
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to delete {filename}: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.get("/api/backtests/results/{symbol}")
async def view_backtest_results(symbol: str, request: Request):
    """View detailed backtest results for a symbol"""
    try:
        symbol = symbol.upper()
        results_file = Path("backtest_results") / symbol / "latest_results.json"
        
        if not results_file.exists():
            raise HTTPException(status_code=404, detail=f"Results for {symbol} not found")
        
        # Read and parse results
        with open(results_file, 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
        
        # Check if this is a CSV backtest result
        if data.get('type') == 'csv_backtest':
            # Find the latest CSV file and load signals data
            symbol_dir = Path("backtest_results") / symbol
            csv_files = list(symbol_dir.glob("csv_backtest_*.csv"))
            
            signals = []
            if csv_files:
                # Get the most recent CSV file
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                try:
                    import pandas as pd
                    signals_df = pd.read_csv(latest_csv)
                    signals = signals_df.to_dict('records')
                except Exception as e:
                    logger.warning(f"Failed to load CSV data for {symbol}: {e}")
            
            # Prepare data for CSV backtest template
            results = {
                'total_signals': data.get('total_signals', 0),
                'long_signals': data.get('long_signals', 0),
                'short_signals': data.get('short_signals', 0),
                'average_rr': data.get('average_rr', 0),
                'fvg_confluence': data.get('fvg_confluence_signals', 0),
                'timestamp': data.get('timestamp', 'N/A'),
                'total_pnl': data.get('total_pnl', 0),
                'winning_trades': data.get('winning_trades', 0),
                'losing_trades': data.get('losing_trades', 0),
                'win_rate': data.get('win_rate', 0),
                'avg_duration_hours': data.get('avg_duration_hours', 0)
            }
            parameters = data.get('parameters', {})
            
            return templates.TemplateResponse("backtest_results_modal.html", {
                "request": request,
                "symbol": symbol,
                "backtest_type": "csv_backtest",
                "results": results,
                "parameters": parameters,
                "signals": signals
            })
        else:
            # Handle regular backtest results
            if 'report' in data:
                report = data['report']
            else:
                report = data
            
            # Prepare data for regular backtest template
            results = {
                'total_trades': report.get('total_trades', 0),
                'win_rate': report.get('win_rate', 0),
                'profit_factor': report.get('profit_factor', 0),
                'total_pnl': report.get('total_pnl', 0),
                'max_drawdown': report.get('max_drawdown', 0),
                'risk_level': report.get('risk_level', 'N/A'),
                'recommendation': report.get('recommendation', 'N/A'),
                'timestamp': data.get('timestamp', 'N/A')
            }
            
            return templates.TemplateResponse("backtest_results_modal.html", {
                "request": request,
                "symbol": symbol,
                "backtest_type": "regular",
                "results": results,
                "parameters": {}
            })
        
    except Exception as e:
        logger.error(f"Error viewing results for {symbol}: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-exclamation-triangle"></i>
                <strong>Error!</strong> Failed to load results for {symbol}: {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.get("/api/backtests/history")
async def get_backtest_history(request: Request):
    """Get list of cached data files and backtest results"""
    try:
        # Get data directory files
        data_dir = Path("data")
        backtest_dir = Path("backtest_results")
        
        cached_files = []
        if data_dir.exists():
            for file_path in data_dir.glob("*.csv"):
                stat = file_path.stat()
                cached_files.append({
                    'name': file_path.name,
                    'size': f"{stat.st_size / 1024:.1f} KB",
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                })
        
        # Get backtest results
        backtest_results = []
        if backtest_dir.exists():
            # Look for results in symbol directories
            for symbol_dir in backtest_dir.iterdir():
                if symbol_dir.is_dir():
                    latest_file = symbol_dir / "latest_results.json"
                    if latest_file.exists():
                        stat = latest_file.stat()
                        try:
                            # Read JSON to get more details
                            with open(latest_file, 'r', encoding='utf-8') as f:
                                import json
                                data = json.load(f)
                                backtest_type = data.get('type', 'unknown')
                                total_signals = data.get('total_signals', 0)
                        except:
                            backtest_type = 'unknown'
                            total_signals = 0
                        
                        backtest_results.append({
                            'symbol': symbol_dir.name,
                            'last_run': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                            'file_size': f"{stat.st_size / 1024:.1f} KB",
                            'type': backtest_type,
                            'total_signals': total_signals
                        })
        
        return templates.TemplateResponse("backtest_results.html", {
            "request": request,
            "cached_files": cached_files,
            "backtest_results": backtest_results
        })
        
    except Exception as e:
        logger.error(f"Backtest history error: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i>
                Error loading history: {str(e)}
            </div>
        """)


@app.post("/api/telegram/toggle")
async def toggle_telegram():
    """Toggle Telegram notifications"""
    try:
        if bot_state.telegram_enabled:
            # Disable Telegram
            bot_state.telegram_enabled = False
            bot_state.telegram_client = None
            logger.info("Telegram disabled")
            return HTMLResponse(f"""
                <div class="alert alert-warning alert-dismissible fade show">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Warning!</strong> Telegram notifications disabled
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        else:
            # Try to enable Telegram
            init_telegram()
            if bot_state.telegram_enabled:
                # Send test message
                test_sent = bot_state.telegram_client.send_status_update("✅ Telegram notifications enabled!")
                if test_sent:
                    logger.info("Telegram enabled and test message sent")
                    return HTMLResponse(f"""
                        <div class="alert alert-success alert-dismissible fade show">
                            <i class="bi bi-check-circle"></i>
                            <strong>Success!</strong> Telegram enabled and test message sent
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                    """)
                else:
                    bot_state.telegram_enabled = False
                    return HTMLResponse(f"""
                        <div class="alert alert-danger alert-dismissible fade show">
                            <i class="bi bi-x-circle"></i>
                            <strong>Error!</strong> Telegram connected but failed to send test message
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                    """)
            else:
                return HTMLResponse(f"""
                    <div class="alert alert-danger alert-dismissible fade show">
                        <i class="bi bi-x-circle"></i>
                        <strong>Error!</strong> Failed to connect to Telegram. Check token and chat_id
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                """)
                
    except Exception as e:
        logger.error(f"Telegram toggle error: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-x-circle"></i>
                <strong>Error!</strong> {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.post("/api/telegram/test")
async def test_telegram():
    """Send test message to Telegram"""
    try:
        if not bot_state.telegram_enabled or not bot_state.telegram_client:
            return HTMLResponse(f"""
                <div class="alert alert-warning alert-dismissible fade show">
                    <i class="bi bi-exclamation-triangle"></i>
                    <strong>Warning!</strong> Telegram not enabled
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        
        # Send test signal
        test_signal = {
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 43250.50,
            "sl": 42800.00,
            "tp": 44600.00,
            "rr": 3.0,
            "htf_bias": "bull",
            "fvg_confluence": True
        }
        
        success = bot_state.telegram_client.send_signal_notification(test_signal)
        
        if success:
            return HTMLResponse(f"""
                <div class="alert alert-success alert-dismissible fade show">
                    <i class="bi bi-check-circle"></i>
                    <strong>Success!</strong> Test signal sent to Telegram
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
        else:
            return HTMLResponse(f"""
                <div class="alert alert-danger alert-dismissible fade show">
                    <i class="bi bi-x-circle"></i>
                    <strong>Error!</strong> Failed to send test message
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            """)
            
    except Exception as e:
        logger.error(f"Telegram test error: {e}")
        return HTMLResponse(f"""
            <div class="alert alert-danger alert-dismissible fade show">
                <i class="bi bi-x-circle"></i>
                <strong>Error!</strong> {str(e)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        """)


@app.post("/api/backtest/csv")
async def run_csv_backtest(
    request: Request,
    ltf_file: str = Form(...),
    htf_file: str = Form(...),
    rr_min: float = Form(3.0),
    fractal_left: int = Form(2),
    fractal_right: int = Form(2),
    require_fvg: bool = Form(False)
):
    """Run backtest with CSV files (replaces main.py functionality)"""
    try:
        # Validate input files exist - look in data/ directory
        data_dir = Path("data")
        
        # Check if data directory exists
        if not data_dir.exists():
            raise HTTPException(status_code=400, detail="Data directory not found")
        
        ltf_path = data_dir / ltf_file
        htf_path = data_dir / htf_file
        
        # Security check: ensure files are within data directory
        try:
            if not ltf_path.resolve().is_relative_to(data_dir.resolve()):
                raise HTTPException(status_code=400, detail="Invalid LTF file path")
            if not htf_path.resolve().is_relative_to(data_dir.resolve()):
                raise HTTPException(status_code=400, detail="Invalid HTF file path")
        except (ValueError, RuntimeError):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        if not ltf_path.exists():
            raise HTTPException(status_code=400, detail=f"LTF file not found: {ltf_file}")
        if not htf_path.exists():
            raise HTTPException(status_code=400, detail=f"HTF file not found: {htf_file}")
        
        logger.info(f"Starting CSV backtest: LTF={ltf_file}, HTF={htf_file}")
        logger.info(f"File paths: LTF={ltf_path}, HTF={htf_path}")
        
        # Load data
        try:
            df_ltf, df_htf = prepare_data(str(ltf_path), str(htf_path))
            logger.info(f"Data loaded: LTF={len(df_ltf)} rows, HTF={len(df_htf)} rows")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")
        
        # Validate timeframe alignment
        try:
            if not validate_timeframe_alignment(df_ltf, df_htf):
                logger.warning("Timeframe alignment issues detected")
        except Exception as e:
            logger.warning(f"Timeframe validation failed: {e}")
        
        # Generate signals
        try:
            signals = generate_signals(
                df_ltf=df_ltf,
                df_htf=df_htf,
                left=fractal_left,
                right=fractal_right,
                rr_min=rr_min
            )
            logger.info(f"Signals generated: {len(signals)} total")
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate signals: {str(e)}")
        
        # Validate signals using BacktestValidator
        try:
            if len(signals) > 0:
                # Initialize validator with $100 position size
                validator = BacktestValidator(df_ltf, position_size=100.0)
                trade_results = validator.validate_all_signals(signals)
                report = validator.generate_report(trade_results)
                logger.info(f"Backtest validation completed: {len(trade_results)} trades validated")
            else:
                trade_results = []
                report = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'profit_factor': 0,
                    'avg_duration_hours': 0
                }
        except Exception as e:
            logger.error(f"Failed to validate signals: {e}")
            # Fallback to basic results if validation fails
            trade_results = []
            report = {
                'total_trades': len(signals),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'avg_duration_hours': 0
            }
        
        # Save results
        try:
            # Extract symbol from filename if possible
            symbol_from_ltf = ltf_file.replace('.csv', '').split('_')[0].upper() if '_' in ltf_file else 'UNKNOWN'
            
            # Create symbol directory in backtest_results
            backtest_dir = Path("backtest_results")
            symbol_dir = backtest_dir / symbol_from_ltf
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CSV results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = symbol_dir / f"csv_backtest_{timestamp}.csv"
            signals.to_csv(csv_file, index=False)
            
            # Also save a summary JSON file for consistency with other backtests
            summary_data = {
                "timestamp": timestamp,
                "type": "csv_backtest",
                "symbol": symbol_from_ltf,
                "total_signals": len(signals),
                "long_signals": len(signals[signals['direction'] == 'LONG']) if len(signals) > 0 else 0,
                "short_signals": len(signals[signals['direction'] == 'SHORT']) if len(signals) > 0 else 0,
                "average_rr": float(signals['rr'].mean()) if len(signals) > 0 else 0,
                "fvg_confluence_signals": int(signals['fvg_confluence'].sum()) if len(signals) > 0 else 0,
                # Add P&L statistics
                "total_pnl": float(signals['pnl'].sum()) if len(signals) > 0 else 0,
                "winning_trades": len(signals[signals['pnl'] > 0]) if len(signals) > 0 else 0,
                "losing_trades": len(signals[signals['pnl'] < 0]) if len(signals) > 0 else 0,
                "win_rate": float(len(signals[signals['pnl'] > 0]) / len(signals) * 100) if len(signals) > 0 else 0,
                "avg_duration_hours": float(signals['duration_minutes'].mean() / 60) if len(signals) > 0 else 0,
                "parameters": {
                    "ltf_file": ltf_file,
                    "htf_file": htf_file,
                    "rr_min": rr_min,
                    "fractal_left": fractal_left,
                    "fractal_right": fractal_right,
                    "require_fvg": require_fvg
                }
            }
            
            # Save summary JSON
            import json
            summary_file = symbol_dir / "latest_results.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {symbol_dir}")
            output_file = csv_file  # Use CSV file as main output for backward compatibility
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")
        
        # Prepare response
        result = {
            "success": True,
            "total_signals": len(signals),
            "long_signals": len(signals[signals['direction'] == 'LONG']) if len(signals) > 0 else 0,
            "short_signals": len(signals[signals['direction'] == 'SHORT']) if len(signals) > 0 else 0,
            "average_rr": float(signals['rr'].mean()) if len(signals) > 0 else 0,
            "fvg_confluence_signals": int(signals['fvg_confluence'].sum()) if len(signals) > 0 else 0,
            # Add P&L statistics
            "total_pnl": float(signals['pnl'].sum()) if len(signals) > 0 else 0,
            "winning_trades": len(signals[signals['pnl'] > 0]) if len(signals) > 0 else 0,
            "losing_trades": len(signals[signals['pnl'] < 0]) if len(signals) > 0 else 0,
            "win_rate": float(len(signals[signals['pnl'] > 0]) / len(signals) * 100) if len(signals) > 0 else 0,
            "avg_duration_hours": float(signals['duration_minutes'].mean() / 60) if len(signals) > 0 else 0,
            "output_file": str(output_file),
            "parameters": {
                "ltf_file": ltf_file,
                "htf_file": htf_file,
                "rr_min": rr_min,
                "fractal_left": fractal_left,
                "fractal_right": fractal_right,
                "require_fvg": require_fvg
            }
        }
        
        logger.info(f"CSV backtest completed: {result['total_signals']} signals generated")
        
        # Return HTML results using template
        try:
            return templates.TemplateResponse("backtest_csv_results.html", {
                "request": request,
                "signals": signals.to_dict('records'),
                "trade_results": trade_results,
                "report": report
            })
        except Exception as e:
            logger.error(f"Failed to render template: {e}")
            # Fallback to JSON if template rendering fails
            return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"CSV backtest error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/backtest/csv/files")
async def get_csv_files():
    """Get list of available CSV files in data/ directory"""
    try:
        data_dir = Path("data")
        csv_files = []
        
        if data_dir.exists():
            for file_path in data_dir.glob("*.csv"):
                csv_files.append(file_path.name)
        
        # Sort files alphabetically
        csv_files.sort()
        
        return JSONResponse(content=csv_files)
        
    except Exception as e:
        logger.error(f"Error getting CSV files: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/backtest/csv/form")
async def get_csv_backtest_form():
    """Get HTML form for CSV backtest"""
    return HTMLResponse("""
        <div class="card">
            <div class="card-header">
                <h5><i class="bi bi-file-earmark-csv"></i> CSV Backtest</h5>
            </div>
            <div class="card-body">
                <form hx-post="/api/backtest/csv" hx-target="#csv-backtest-result">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="ltf_file" class="form-label">Lower Timeframe CSV</label>
                                <input type="text" class="form-control" id="ltf_file" name="ltf_file" 
                                       placeholder="data/btc_15m.csv" required>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="htf_file" class="form-label">Higher Timeframe CSV</label>
                                <input type="text" class="form-control" id="htf_file" name="htf_file" 
                                       placeholder="data/btc_4h.csv" required>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="mb-3">
                                <label for="rr_min" class="form-label">Min Risk/Reward</label>
                                <input type="number" class="form-control" id="rr_min" name="rr_min" 
                                       value="3.0" step="0.1" min="1.0">
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="mb-3">
                                <label for="fractal_left" class="form-label">Fractal Left</label>
                                <input type="number" class="form-control" id="fractal_left" name="fractal_left" 
                                       value="2" min="1">
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="mb-3">
                                <label for="fractal_right" class="form-label">Fractal Right</label>
                                <input type="number" class="form-control" id="fractal_right" name="fractal_right" 
                                       value="2" min="1">
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="mb-3">
                                <label for="require_fvg" class="form-label">Require FVG</label>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="require_fvg" name="require_fvg">
                                    <label class="form-check-label" for="require_fvg">
                                        Require FVG confluence
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-primary">
                            <i class="bi bi-play-circle"></i> Run CSV Backtest
                        </button>
                    </div>
                </form>
                <div id="csv-backtest-result" class="mt-3"></div>
            </div>
        </div>
    """)


# Функція render_backtest_results_html видалена - тепер використовуються Jinja2 шаблони


# Цей endpoint більше не потрібен - видалено


@app.get("/api/backtests/csv/download/{symbol}")
async def download_csv_file(symbol: str):
    """Download CSV file for a symbol"""
    try:
        symbol = symbol.upper()
        symbol_dir = Path("backtest_results") / symbol
        
        if not symbol_dir.exists():
            raise HTTPException(status_code=404, detail=f"Results directory for {symbol} not found")
        
        # Find the latest CSV file
        csv_files = list(symbol_dir.glob("csv_backtest_*.csv"))
        
        if not csv_files:
            raise HTTPException(status_code=404, detail=f"CSV file not found for {symbol}")
        
        # Get the most recent CSV file
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Return file as response
        from fastapi.responses import FileResponse
        return FileResponse(
            path=str(latest_csv),
            filename=f"{symbol}_backtest_results.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error downloading CSV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
