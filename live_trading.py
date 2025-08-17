#!/usr/bin/env python3
"""
Live SMC Trading Monitor
Real-time Smart Money Concepts signal detection for Binance USD-M Futures
"""
import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.live_monitor import run_live_monitor

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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, quiet_mode=args.quiet)
    
    # Configuration
    config = {
        'min_risk_reward': args.rr,
        'fractal_left': args.fractal_left,
        'fractal_right': args.fractal_right,
        'desktop_alerts': args.desktop_alerts,
        'sound_alerts': args.sound_alerts,
        'status_check_interval': args.status_check_interval
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
        print(f"📝 Log level: {args.log_level}")
        print("=" * 60)
        print("Press [Ctrl+C] or [Q] to quit")
        print("=" * 60)
    
    try:
        # Run live monitor
        await run_live_monitor(symbol, config)
        
    except KeyboardInterrupt:
        print("\n👋 Live monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Main error: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
