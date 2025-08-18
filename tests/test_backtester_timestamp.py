#!/usr/bin/env python3
"""
Test backtester with real timestamp data
"""
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_backtester_timestamp_handling():
    """Test backtester with timestamp data"""
    print("🧪 Testing Backtester with timestamp data...")
    
    try:
        from backtester import BacktestValidator
        
        # Create sample price data with timestamps
        dates = pd.date_range('2025-08-18 10:00:00', periods=100, freq='1min')
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0 + i * 0.01 for i in range(100)],
            'high': [100.5 + i * 0.01 for i in range(100)],
            'low': [99.5 + i * 0.01 for i in range(100)],
            'close': [100.1 + i * 0.01 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
        
        # Create sample signal
        signal = {
            'timestamp': '2025-08-18 10:30:00',
            'entry': 100.3,
            'sl': 99.8,
            'tp': 101.0,
            'direction': 'LONG'
        }
        
        # Initialize validator
        validator = BacktestValidator(price_data)
        
        # Validate signal
        result = validator.validate_signal(signal)
        
        # Check that result has proper timestamp handling
        assert isinstance(result.timestamp, str), f"Expected string timestamp, got {type(result.timestamp)}"
        assert isinstance(result.exit_time, str), f"Expected string exit_time, got {type(result.exit_time)}"
        assert isinstance(result.duration_minutes, int), f"Expected int duration, got {type(result.duration_minutes)}"
        
        print("✅ Backtester timestamp handling works correctly")
        print(f"   Signal timestamp: {result.timestamp}")
        print(f"   Exit time: {result.exit_time}")
        print(f"   Duration: {result.duration_minutes} minutes")
        print(f"   Exit reason: {result.exit_reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtester timestamp test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("🚀 Testing Backtester Timestamp Handling")
    print("=" * 50)
    
    if test_backtester_timestamp_handling():
        print("\n🎉 Backtester timestamp test passed!")
        return 0
    else:
        print("\n❌ Backtester timestamp test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
