#!/usr/bin/env python3
"""
NEW SMC Trading Bot Web Panel - 2 Tabs (Dashboard + Backtests)
"""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SMC Trading Bot - NEW Web Panel",
    description="2-Tab Interface: Dashboard + Backtests",
    version="3.0.0"
)

# Templates and static files
template_dir = project_root / "src" / "web" / "templates"
static_dir = project_root / "src" / "web" / "static"

templates = Jinja2Templates(directory=str(template_dir))

# Mock data for testing
mock_pairs = [
    {"symbol": "BTCUSDT", "enabled": True, "backtest_enabled": False, "status": "running"},
    {"symbol": "ETHUSDT", "enabled": False, "backtest_enabled": True, "status": "stopped"},
    {"symbol": "SOLUSDT", "enabled": False, "backtest_enabled": False, "status": "stopped"},
]

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page with 2 tabs"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "pairs": mock_pairs,
        "total_pairs": len(mock_pairs),
        "enabled_pairs": len([p for p in mock_pairs if p["enabled"]]),
        "backtest_pairs": len([p for p in mock_pairs if p["backtest_enabled"]]),
        "signals": []
    })

@app.get("/api/status")
async def get_status():
    """Get API status"""
    return {
        "status": "running",
        "version": "3.0.0",
        "message": "NEW Web Panel is working!"
    }

@app.post("/api/pairs/{symbol}/toggle-trade")
async def toggle_trade(symbol: str):
    """Toggle trading for a pair"""
    for pair in mock_pairs:
        if pair["symbol"] == symbol:
            pair["enabled"] = not pair["enabled"]
            pair["status"] = "running" if pair["enabled"] else "stopped"
            return {"success": True, "enabled": pair["enabled"]}
    raise HTTPException(status_code=404, detail="Pair not found")

@app.post("/api/pairs/{symbol}/toggle-backtest")
async def toggle_backtest(symbol: str):
    """Toggle backtest for a pair"""
    for pair in mock_pairs:
        if pair["symbol"] == symbol:
            pair["backtest_enabled"] = not pair["backtest_enabled"]
            return {"success": True, "backtest_enabled": pair["backtest_enabled"]}
    raise HTTPException(status_code=404, detail="Pair not found")

@app.post("/api/backtests/fetch")
async def fetch_data(
    symbol: str = Form(...),
    timeframe: str = Form("15m"),
    days: int = Form(30)
):
    """Mock fetch historical data"""
    return HTMLResponse(f"""
        <div class="alert alert-success alert-dismissible fade show">
            <i class="bi bi-check-circle"></i>
            <strong>Success!</strong> Downloaded {symbol} {timeframe} data for {days} days.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    """)

@app.post("/api/backtests/run")
async def run_backtest(
    symbol: str = Form(...),
    timeframe: str = Form("15m"),
    days: int = Form(30),
    min_rr: float = Form(3.0)
):
    """Mock run backtest"""
    return HTMLResponse(f"""
        <div class="alert alert-success alert-dismissible fade show">
            <i class="bi bi-chart-line"></i>
            <strong>Backtest Complete!</strong> {symbol} - 25 trades
            <br><small>Win Rate: 68.0% | P&L: $1250.50 | PF: 2.15</small>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    """)

@app.get("/api/backtests/history")
async def get_history(request: Request):
    """Get backtest history"""
    return templates.TemplateResponse("backtest_results.html", {
        "request": request,
        "cached_files": [
            {"name": "BTCUSDT_15m.csv", "size": "2.5 MB", "modified": "2024-01-15 14:30"},
            {"name": "ETHUSDT_4h.csv", "size": "1.8 MB", "modified": "2024-01-15 12:15"}
        ],
        "backtest_results": [
            {"symbol": "BTCUSDT", "last_run": "2024-01-15 14:30", "file_size": "45.2 KB"},
            {"symbol": "ETHUSDT", "last_run": "2024-01-15 12:15", "file_size": "38.7 KB"}
        ]
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NEW SMC Web Panel on port 8001...")
    print("🌐 Open: http://localhost:8001")
    print("🔧 Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
