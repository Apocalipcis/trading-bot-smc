@echo off
echo 🐳 SMC Trading Bot - Docker Quick Start
echo ======================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running! Please start Docker Desktop first.
    pause
    exit /b 1
)

echo ✅ Docker is running

REM Build and start services
echo.
echo 🚀 Building and starting services...
docker-compose up --build -d

if errorlevel 1 (
    echo ❌ Failed to start services
    pause
    exit /b 1
)

echo.
echo ✅ Services started successfully!
echo.
echo 🌐 Web Interface: http://localhost:8000
echo 📊 WebSocket: ws://localhost:8001
echo.
echo 📋 Available commands:
echo   docker-compose logs -f smc-trading-bot    - View logs
echo   docker-compose down                       - Stop services
echo   python docker-run.py                      - Interactive menu
echo.
echo 🎯 Quick start:
echo   python docker-run.py --backtest ETHUSDT  - Run backtest
echo   python docker-run.py --live ETHUSDT      - Start live trading
echo.
pause
