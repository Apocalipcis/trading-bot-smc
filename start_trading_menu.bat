@echo off
setlocal
chcp 65001 >nul
color 0A
title SMC Live Trading Monitor

rem Resolve script root (assumes this BAT sits in project root)
set "ROOT=%~dp0"
set "PY=%ROOT%venv\Scripts\python.exe"
set "SCRIPT=%ROOT%live_trading.py"

:MAIN_MENU
cls
echo.
echo ===============================
echo       SMC Live Trading Monitor
echo ===============================
echo.
echo Choose a trading pair (with backtest):
echo.
echo  1) BTCUSDT  - Bitcoin
echo  2) ETHUSDT  - Ethereum
echo  3) BNBUSDT  - Binance Coin
echo  4) SOLUSDT  - Solana
echo  5) DOGEUSDT - Dogecoin
echo  6) Custom pair
echo.
echo Advanced options:
echo  7) Settings
echo  8) Help
echo  9) Exit
echo.
set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" set "SYMBOL=BTCUSDT"& goto BACKTEST_TRADING
if "%choice%"=="2" set "SYMBOL=ETHUSDT" & goto BACKTEST_TRADING
if "%choice%"=="3" set "SYMBOL=BNBUSDT" & goto BACKTEST_TRADING
if "%choice%"=="4" set "SYMBOL=SOLUSDT" & goto BACKTEST_TRADING
if "%choice%"=="5" set "SYMBOL=DOGEUSDT" & goto BACKTEST_TRADING
if "%choice%"=="6" goto CUSTOM_PAIR_BACKTEST
if "%choice%"=="7" goto SETTINGS
if "%choice%"=="8" goto HELP
if "%choice%"=="9" goto EXIT

echo Invalid choice! Press any key to try again...
pause >nul
goto MAIN_MENU

rem RUN_WITH_BACKTEST section removed - all pairs now go directly to backtest

:CUSTOM_PAIR_BACKTEST
cls
echo.
echo ===============================
echo    Custom Trading Pair (with backtest)
echo ===============================
echo.
echo Enter a trading pair (e.g., ATOMUSDT, LINKUSDT):
set /p SYMBOL="Symbol: "

if "%SYMBOL%"=="" (
    echo Symbol cannot be empty!
    pause
    goto MAIN_MENU
)

echo.
echo You entered: %SYMBOL%
set /p confirm="Is this correct? (Y/N): "
if /i "%confirm%"=="Y" (
    rem Defaults if not set
    if not defined RR set "RR=3.0"
    if not defined INTERVAL set "INTERVAL=45"
    if not defined ALERTS set "ALERTS=--desktop-alerts"
    
    if exist "%PY%" (
        echo Starting bot with backtest...
        "%PY%" "%SCRIPT%" --symbol "%SYMBOL%" --rr %RR% --status-check-interval %INTERVAL% %ALERTS% --run-backtest --backtest-days 30
    ) else (
        echo Error: Virtual environment not found!
        echo Please create it and install dependencies:
        echo   python -m venv venv
        echo   venv\Scripts\pip install -r requirements.txt
    )
    
    echo.
    echo Bot stopped. Press any key to return to the menu...
    pause >nul
    goto MAIN_MENU
)
goto MAIN_MENU

:BACKTEST_TRADING
cls
echo.
echo ===============================
echo    Starting Trading Bot with Backtest
echo ===============================
echo.
echo Symbol: %SYMBOL%
echo Mode: Backtest + Live Trading
echo.
echo The bot will:
echo  1. Run a 30-day backtest first
echo  2. Show you the results
echo  3. Ask if you want to continue (Y/N)
echo  4. Start live trading if you choose Y
echo.
pause

rem Defaults if not set
if not defined RR set "RR=3.0"
if not defined INTERVAL set "INTERVAL=45"
if not defined ALERTS set "ALERTS=--desktop-alerts"

if exist "%PY%" (
    echo Starting bot with backtest...
    "%PY%" "%SCRIPT%" --symbol "%SYMBOL%" --rr %RR% --status-check-interval %INTERVAL% %ALERTS% --run-backtest --backtest-days 30
) else (
    echo Error: Virtual environment not found!
    echo Please create it and install dependencies:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
)

echo.
echo Bot stopped. Press any key to return to the menu...
pause >nul
goto MAIN_MENU

rem CUSTOM_PAIR section removed - now handled by CUSTOM_PAIR_BACKTEST

:SETTINGS
cls
echo.
echo ===============================
echo             Settings
echo ===============================
echo.
echo Choose Risk/Reward ratio:
echo  1) RR 2.0 (Conservative)
echo  2) RR 3.0 (Default)
echo  3) RR 4.0 (Aggressive)
echo.
set /p rr_choice="Enter choice (1-3): "

if "%rr_choice%"=="1" set "RR=2.0"
if "%rr_choice%"=="2" set "RR=3.0"
if "%rr_choice%"=="3" set "RR=4.0"
if not defined RR set "RR=3.0"

echo.
echo Status-check interval:
echo  1) 30 seconds (Fast)
echo  2) 45 seconds (Default)
echo  3) 60 seconds (Slow)
echo.
set /p interval_choice="Enter choice (1-3): "

if "%interval_choice%"=="1" set "INTERVAL=30"
if "%interval_choice%"=="2" set "INTERVAL=45"
if "%interval_choice%"=="3" set "INTERVAL=60"
if not defined INTERVAL set "INTERVAL=45"

echo.
set /p alerts="Desktop alerts? (Y/N): "
if /i "%alerts%"=="Y" (
    set "ALERTS=--desktop-alerts"
) else (
    set "ALERTS="
)

echo.
echo Settings applied:
echo  - Risk/Reward: %RR%
echo  - Check interval: %INTERVAL%s
echo  - Desktop alerts: %alerts%
echo.
pause
goto MAIN_MENU

rem START_TRADING section removed - all pairs now go through backtest

:HELP
cls
echo.
echo ===============================
echo               Help
echo ===============================
echo.
echo SMC Live Trading Monitor допомагає відстежувати сигнали
echo на основі Smart Money Concepts.
echo.
echo СТАТУСИ:
echo  🟡 NEW       - Новий сигнал (перші 5 хв)
echo  📈 VALID     - Ціна між SL та Entry
echo  ⚡ TRIGGERED - Досягнуто Entry
echo  ✅ HIT TP    - Досягнуто Take Profit
echo  ❌ HIT SL    - Досягнуто Stop Loss
echo  ⏭️ MISSED    - Пройшли повз Entry
echo  ⏰ EXPIRED   - Старше 2 годин
echo.
echo ГАРЯЧІ КЛАВІШІ:
echo  [C] Entry   [S] SL   [V] TP
echo  [A] All signal info  [P] Price
echo  [Del] Clear   [Q] Quit
echo.
echo Налаштування:
echo  - Risk/Reward: мінімальний RR для відбору
echo  - Check interval: частота оновлення
echo  - Desktop alerts: системні сповіщення
echo.
pause
goto MAIN_MENU

:EXIT
cls
echo.
echo Thanks for using SMC Live Trading Monitor!
echo.
timeout /t 2 >nul
endlocal
exit /b

:CLEANUP
set "SYMBOL="
set "RR="
set "INTERVAL="
set "ALERTS="
rem All pairs now use backtest by default
