#!/usr/bin/env python3
"""
Test script for the new SMC trading bot architecture
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import load_config, save_config, AppConfig, PairConfig
from src.core import LiveExchangeGateway, PairManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_config_system():
    """Test configuration loading and saving"""
    print("=" * 50)
    print("TESTING CONFIGURATION SYSTEM")
    print("=" * 50)
    
    # Load config
    config = load_config()
    print(f"Loaded config with {len(config.pairs)} pairs")
    
    # Validate config
    errors = config.validate()
    if errors:
        print(f"Config validation errors: {errors}")
    else:
        print("✅ Configuration is valid")
    
    # Test hot config updates
    enabled_pairs = config.get_enabled_pairs()
    backtest_pairs = config.get_backtest_enabled_pairs()
    
    print(f"Enabled pairs: {[p.symbol for p in enabled_pairs]}")
    print(f"Backtest enabled pairs: {[p.symbol for p in backtest_pairs]}")
    
    return config


async def test_exchange_gateway():
    """Test exchange gateway"""
    print("\n" + "=" * 50)
    print("TESTING EXCHANGE GATEWAY")
    print("=" * 50)
    
    async with LiveExchangeGateway() as gateway:
        # Test price fetching
        try:
            price = await gateway.get_current_price("BTCUSDT")
            print(f"✅ Current BTC price: ${price:,.2f}")
        except Exception as e:
            print(f"❌ Failed to get price: {e}")
        
        # Test candle fetching
        try:
            candles = await gateway.fetch_candles("BTCUSDT", "15m", 10)
            print(f"✅ Fetched {len(candles)} candles for BTCUSDT 15m")
            if candles:
                latest = candles[-1]
                print(f"   Latest candle: O:{latest.open} H:{latest.high} L:{latest.low} C:{latest.close}")
        except Exception as e:
            print(f"❌ Failed to fetch candles: {e}")


async def test_pair_manager(config: AppConfig):
    """Test pair manager"""
    print("\n" + "=" * 50)
    print("TESTING PAIR MANAGER")
    print("=" * 50)
    
    async with LiveExchangeGateway() as gateway:
        # Create pair manager
        pair_manager = PairManager(gateway, config)
        
        # Add callbacks
        def on_signal(signal):
            print(f"📈 Signal: {signal.symbol} {signal.direction} at {signal.entry}")
        
        def on_status(statuses):
            print(f"📊 Status update: {len(statuses)} pairs")
            for symbol, status in statuses.items():
                print(f"   {symbol}: {status.status} (price: ${status.current_price or 0:.2f})")
        
        pair_manager.add_signal_callback(on_signal)
        pair_manager.add_status_callback(on_status)
        
        try:
            # Start pair manager
            await pair_manager.start()
            print("✅ Pair manager started")
            
            # Let it run for a short time
            print("Running for 30 seconds...")
            await asyncio.sleep(30)
            
            # Test hot config reload
            print("\n🔄 Testing hot config reload...")
            
            # Modify config - enable ETHUSDT
            new_config = AppConfig(
                max_concurrent_pairs=config.max_concurrent_pairs,
                pairs=[
                    PairConfig(symbol="BTCUSDT", enabled=True, backtest_enabled=False),
                    PairConfig(symbol="ETHUSDT", enabled=True, backtest_enabled=True),
                    PairConfig(symbol="SOLUSDT", enabled=False, backtest_enabled=False),
                ],
                telegram_token=config.telegram_token,
                telegram_chat_id=config.telegram_chat_id
            )
            
            await pair_manager.apply_config(new_config)
            print("✅ Hot config reload completed")
            
            # Run a bit more
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"❌ Error in pair manager test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop pair manager
            await pair_manager.stop()
            print("✅ Pair manager stopped")


async def test_backtest_system():
    """Test backtest system"""
    print("\n" + "=" * 50)
    print("TESTING BACKTEST SYSTEM")
    print("=" * 50)
    
    try:
        from src.backtest import run_backtest_async
        
        config = {
            'min_risk_reward': 3.0,
            'fractal_left': 2,
            'fractal_right': 2,
            'backtest_days': 7  # Short backtest for testing
        }
        
        print("Starting async backtest for BTCUSDT...")
        result = await run_backtest_async("BTCUSDT", config)
        
        if result.get('success', False):
            print("✅ Backtest completed successfully")
            print(f"   Total trades: {result.get('total_trades', 0)}")
            print(f"   Win rate: {result.get('win_rate', 0):.1f}%")
            print(f"   Profit factor: {result.get('profit_factor', 0):.2f}")
        else:
            print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error in backtest test: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function"""
    print("🚀 TESTING NEW SMC TRADING BOT ARCHITECTURE")
    print("=" * 60)
    
    try:
        # Test configuration system
        config = await test_config_system()
        
        # Test exchange gateway
        await test_exchange_gateway()
        
        # Test backtest system  
        await test_backtest_system()
        
        # Test pair manager (most comprehensive test)
        await test_pair_manager(config)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
