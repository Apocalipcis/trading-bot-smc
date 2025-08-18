@echo off
echo 🐳 SMC Trading Bot - Docker Quick Start
echo ======================================

REM Check if Docker is running
echo 🔍 Checking Docker status...
docker version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running! Please start Docker Desktop first.
    echo 💡 Make sure Docker Desktop is installed and running.
    pause
    exit /b 1
)

echo ✅ Docker is available

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found!
    echo 📝 Creating .env from template...
    if exist "dot.env.example" (
        copy "dot.env.example" ".env" >nul
        echo ✅ .env file created from template
        echo 🔧 Please edit .env file with your credentials if needed
    ) else (
        echo ❌ dot.env.example not found! Cannot create .env file.
        pause
        exit /b 1
    )
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down >nul 2>&1

REM Build and start services
echo.
echo 🚀 Building and starting SMC Trading Bot Web Panel...
docker-compose up --build -d

if errorlevel 1 (
    echo ❌ Failed to start services
    echo 📋 Showing recent logs:
    docker-compose logs --tail=20 smc-web
    pause
    exit /b 1
)

REM Wait for service to be ready
echo 🕐 Waiting for web panel to start...
timeout /t 5 /nobreak >nul

echo.
echo ✅ SMC Trading Bot Web Panel started successfully!
echo.
echo 🌐 Web Interface: http://localhost:8000
echo 📊 Live Trading Panel: http://localhost:8000
echo.
echo 📋 Available commands:
echo   docker-compose logs -f smc-web           - View web panel logs
echo   docker-compose down                      - Stop services
echo   docker-compose restart smc-web          - Restart web panel
echo.
echo 🔍 Check status:
echo   docker-compose ps                        - View containers
echo   docker-compose logs smc-web              - View all logs
echo.
echo 🎯 Opening web panel in browser...
start http://localhost:8000
echo.
pause
