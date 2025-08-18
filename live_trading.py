#!/usr/bin/env python3
"""
Live SMC Trading Monitor
Real-time Smart Money Concepts signal detection for Binance USD-M Futures
"""
import asyncio
import argparse
import sys
import logging
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.live_monitor import run_live_monitor
from src.telegram_client import TelegramClient

def setup_logging(level: str = "INFO", quiet_mode: bool = False):
    """Setup logging configuration"""
    handlers = [logging.FileHandler('live_trading.log')]
    
    # In quiet mode, don't log to console during TUI operation
    if not quiet_mode:
        handlers.append(logging.StreamHandler(sys.stdout))
        
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Suppress noisy loggers
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

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

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Live SMC Signal Monitor for Binance USD-M Futures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_trading.py --symbol ETHUSDT
  python live_trading.py --symbol BTCUSDT --rr 2.5
  python live_trading.py --symbol ADAUSDT --log-level DEBUG
  python live_trading.py --symbol ETHUSDT --quiet --desktop-alerts
  
Controls (while running):
  [T] - Test signal
  [C] - Clear signals  
  [P] - Pause/Resume
  [Q] - Quit
  [H] - Help
        """
    )
    
    # Arguments
    parser.add_argument('--symbol', default='ETHUSDT',
                       help='Trading pair symbol (default: ETHUSDT)')
    parser.add_argument('--rr', type=float, default=3.0,
                       help='Minimum Risk/Reward ratio (default: 3.0)')
    parser.add_argument('--fractal-left', type=int, default=2,
                       help='Fractal left bars (default: 2)')
    parser.add_argument('--fractal-right', type=int, default=2,
                       help='Fractal right bars (default: 2)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--desktop-alerts', action='store_true',
                       help='Enable desktop notifications')
    parser.add_argument('--sound-alerts', action='store_true',
                       help='Enable sound alerts')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode - log only to file, not console')
    parser.add_argument('--status-check-interval', type=int, default=45,
                       help='Signal status check interval in seconds (default: 45)')
    parser.add_argument('--telegram-token', default=None,
                       help='Telegram bot token for notifications')
    parser.add_argument('--telegram-chat-id', default=None,
                       help='Telegram chat ID for notifications')
    parser.add_argument('--run-backtest', action='store_true',
                       help='Run backtest before starting live trading')
    parser.add_argument('--backtest-days', type=int, default=30,
                       help='Number of days for backtest (default: 30)')
    parser.add_argument('--skip-backtest-prompt', action='store_true',
                       help='Skip confirmation prompt after backtest')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, quiet_mode=args.quiet)
    _load_dotenv()
    
    # Configuration
    token = args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
    config = {
        'min_risk_reward': args.rr,
        'fractal_left': args.fractal_left,
        'fractal_right': args.fractal_right,
        'desktop_alerts': args.desktop_alerts,
        'sound_alerts': args.sound_alerts,
        'status_check_interval': args.status_check_interval,
        'telegram_token': token,
        'telegram_chat_id': chat_id,
        'backtest_days': args.backtest_days
    }
    
    # Validate symbol
    symbol = args.symbol.upper()
    if not symbol.endswith('USDT'):
        print(f"Warning: Symbol {symbol} doesn't end with USDT")
        
    if not args.quiet:
        print(f"🚀 Starting Live SMC Monitor for {symbol}")
        print(f"📊 Min RR: {args.rr}")
        print(f"⚙️  Fractal params: {args.fractal_left}/{args.fractal_right}")
        print(f"🔔 Alerts: Desktop={args.desktop_alerts}, Sound={args.sound_alerts}")
        print(f"⏰ Status check interval: {args.status_check_interval}s")
        if token and chat_id:

            print("📨 Telegram notifications enabled")
        print(f"📝 Log level: {args.log_level}")
        if args.run_backtest:
            print(f"🔍 Backtest: {args.backtest_days} days")
        print("=" * 60)
        print("Press [Ctrl+C] or [Q] to quit")
        print("=" * 60)
    
    try:
        # Run backtest if requested
        if args.run_backtest:
            await run_pre_trade_backtest(symbol, config, args.quiet, args.skip_backtest_prompt)
        # Create Telegram client if configured
        telegram_client = None
        if token and chat_id:
            telegram_client = TelegramClient(token, chat_id)
            # Test connection
            if telegram_client.test_connection():
                if not args.quiet:
                    print("✅ Telegram bot connected successfully")
                # Send startup message
                telegram_client.send_status_update(f"🚀 SMC Bot started for {symbol}")
            else:
                print("❌ Failed to connect to Telegram bot")
                telegram_client = None
        
        # Run live monitor
        await run_live_monitor(symbol, config, telegram_client)
        
    except KeyboardInterrupt:
        print("\n👋 Live monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Main error: {e}", exc_info=True)
        return 1
        
    return 0

async def run_pre_trade_backtest(symbol: str, config: dict, quiet: bool = False, skip_prompt: bool = False):
    """Run pre-trade backtest and show results"""
    try:
        from src.pre_trade_backtest import run_symbol_backtest
        
        if not quiet:
            print(f"\n🔍 Running backtest for {symbol}...")
            print("This may take a few minutes...")
        
        # Run backtest
        result = run_symbol_backtest(symbol, config)
        
        if not result.get('success', False):
            if not quiet:
                print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
            return False
        
        if not quiet:
            print("\n" + "=" * 60)
            print(f"📊 BACKTEST RESULTS FOR {symbol}")
            print("=" * 60)
            
            # Display key metrics
            total_trades = result.get('total_trades', 0)
            if total_trades == 0:
                print(f"⚠️  {result.get('message', 'No signals generated')}")
                print(f"💡 {result.get('recommendation', 'Consider adjusting parameters')}")
                return False
            
            win_rate = result.get('win_rate', 0)
            profit_factor = result.get('profit_factor', 0)
            total_pnl = result.get('total_pnl', 0)
            risk_level = result.get('risk_level', 'UNKNOWN')
            recommendation = result.get('recommendation', 'No recommendation')
            
            print(f"📈 Total Trades: {total_trades}")
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"💰 Total P&L: ${total_pnl:.2f}")
            print(f"📊 Profit Factor: {profit_factor:.2f}")
            print(f"⚠️  Risk Level: {risk_level}")
            print(f"💡 Recommendation: {recommendation}")
            
            # Additional details
            sl_rate = result.get('sl_rate', 0)
            tp_rate = result.get('tp_rate', 0)
            avg_duration = result.get('avg_duration_hours', 0)
            
            print(f"\n📊 Exit Analysis:")
            print(f"   TP Hits: {result.get('tp_hits', 0)} ({tp_rate:.1f}%)")
            print(f"   SL Hits: {result.get('sl_hits', 0)} ({sl_rate:.1f}%)")
            print(f"   No Exit: {result.get('no_exit', 0)}")
            
            print(f"\n⏱️  Timing:")
            print(f"   Average Duration: {avg_duration:.1f} hours")
            print(f"   Backtest Period: {result.get('backtest_period_days', 0)} days")
            
            # Risk assessment
            if risk_level == "HIGH":
                print(f"\n🚨 WARNING: High risk level detected!")
                print(f"   Consider adjusting parameters before trading")
            elif risk_level == "MEDIUM":
                print(f"\n⚠️  CAUTION: Medium risk level")
                print(f"   Monitor performance closely")
            else:
                print(f"\n✅ Good risk level - safe to proceed")
            
            print("=" * 60)
        
        # ALWAYS ask for confirmation - IGNORE ALL FLAGS!
        while True:
            response = input("\n❓ Continue with live trading? (Y/N): ").strip().upper()
            if response in ['Y', 'YES']:
                print("✅ Proceeding with live trading...")
                return True
            elif response in ['N', 'NO']:
                print("❌ Trading cancelled by user")
                sys.exit(0)
            else:
                print("Please enter Y or N")
        
    except Exception as e:
        if not quiet:
            print(f"❌ Backtest error: {e}")
        logging.error(f"Backtest error: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
