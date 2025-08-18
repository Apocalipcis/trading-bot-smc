#!/usr/bin/env python3
"""
Test script for pre-trade backtest functionality
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_backtest():
    """Test the backtest functionality"""
    try:
        from src.pre_trade_backtest import run_symbol_backtest
        
        print("🧪 Testing Pre-Trade Backtest Module")
        print("=" * 50)
        
        # Test configuration
        config = {
            'min_risk_reward': 3.0,
            'fractal_left': 2,
            'fractal_right': 2,
            'backtest_days': 30
        }
        
        # Test with SOLUSDT (should have data)
        print("Testing with SOLUSDT...")
        result = run_symbol_backtest("SOLUSDT", config)
        
        if result.get('success', False):
            print("✅ Backtest completed successfully!")
            print(f"📊 Total trades: {result.get('total_trades', 0)}")
            print(f"🎯 Win rate: {result.get('win_rate', 0):.1f}%")
            print(f"💰 Total P&L: ${result.get('total_pnl', 0):.2f}")
            print(f"📈 Profit factor: {result.get('profit_factor', 0):.2f}")
            print(f"⚠️  Risk level: {result.get('risk_level', 'UNKNOWN')}")
            print(f"💡 Recommendation: {result.get('recommendation', 'N/A')}")
        else:
            print("❌ Backtest failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest()
