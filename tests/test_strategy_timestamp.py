#!/usr/bin/env python3
"""
Test strategy timestamp handling
"""
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_strategy_timestamp_handling():
    """Test strategy timestamp handling"""
    print("🧪 Testing Strategy timestamp handling...")
    
    try:
        # Test the _candles_to_dataframe function
        from core.strategy import SMCStrategy
        
        # Create mock candles data
        candles_data = []
        for i in range(100):
            candles_data.append({
                'timestamp': f'2025-08-18T10:{i:02d}:00',
                'open': 100.0 + i * 0.01,
                'high': 100.5 + i * 0.01,
                'low': 99.5 + i * 0.01,
                'close': 100.1 + i * 0.01,
                'volume': 1000 + i * 10
            })
        
        # Create mock Candle objects
        class MockCandle:
            def __init__(self, data):
                self.timestamp = data['timestamp']
                self.open = data['open']
                self.high = data['high']
                self.low = data['low']
                self.close = data['close']
                self.volume = data['volume']
        
        mock_candles = [MockCandle(data) for data in candles_data]
        
        # Test DataFrame conversion
        strategy = SMCStrategy.__new__(SMCStrategy)  # Create instance without calling __init__
        df = strategy._candles_to_dataframe(mock_candles)
        
        # Check that timestamp index is properly formatted
        assert df.index.dtype == 'datetime64[ns]', f"Expected datetime64 index, got {df.index.dtype}"
        assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
        
        print("✅ Strategy DataFrame conversion works correctly")
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Index type: {df.index.dtype}")
        print(f"   First timestamp: {df.index[0]}")
        print(f"   Last timestamp: {df.index[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy timestamp test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("🚀 Testing Strategy Timestamp Handling")
    print("=" * 50)
    
    if test_strategy_timestamp_handling():
        print("\n🎉 Strategy timestamp test passed!")
        return 0
    else:
        print("\n❌ Strategy timestamp test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
