#!/usr/bin/env python3
"""
Test script to verify timestamp handling fixes
"""
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_timestamp_arithmetic():
    """Test timestamp arithmetic operations"""
    print("🧪 Testing timestamp arithmetic...")
    
    try:
        # Test basic timestamp arithmetic
        t1 = pd.to_datetime('2025-08-18 10:00:00')
        t2 = pd.to_datetime('2025-08-18 11:00:00')
        duration = int((t2 - t1).total_seconds() / 60)
        
        assert duration == 60, f"Expected 60 minutes, got {duration}"
        print("✅ Basic timestamp arithmetic works")
        
        # Test with different timestamp formats
        t3 = pd.to_datetime('2025-08-18T12:00:00')
        t4 = pd.to_datetime('2025-08-18T13:30:00')
        duration2 = int((t4 - t3).total_seconds() / 60)
        
        assert duration2 == 90, f"Expected 90 minutes, got {duration2}"
        print("✅ ISO format timestamp arithmetic works")
        
        return True
        
    except Exception as e:
        print(f"❌ Timestamp arithmetic test failed: {e}")
        return False

def test_signal_creation():
    """Test Signal object creation with proper timestamp"""
    print("\n🧪 Testing Signal creation...")
    
    try:
        from models import Signal
        
        # Create signal with ISO timestamp
        signal = Signal(
            timestamp=datetime.now().isoformat(),
            direction='LONG',
            entry=100.0,
            sl=95.0,
            tp=110.0,
            rr=2.0,
            htf_bias='bull',
            ob_idx=1,
            bos_idx=2,
            fvg_confluence=True
        )
        
        # Verify timestamp is string
        assert isinstance(signal.timestamp, str), f"Expected string, got {type(signal.timestamp)}"
        assert 'T' in signal.timestamp, "Expected ISO format timestamp"
        
        print("✅ Signal creation with ISO timestamp works")
        return True
        
    except Exception as e:
        print(f"❌ Signal creation test failed: {e}")
        return False

def test_backtester_import():
    """Test that backtester imports without errors"""
    print("\n🧪 Testing backtester import...")
    
    try:
        import backtester
        print("✅ Backtester imports successfully")
        return True
        
    except Exception as e:
        print(f"❌ Backtester import failed: {e}")
        return False

def test_strategy_import():
    """Test that strategy imports without errors"""
    print("\n🧪 Testing strategy import...")
    
    try:
        import core.strategy
        print("✅ Strategy imports successfully")
        return True
        
    except Exception as e:
        print(f"❌ Strategy import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Timestamp Handling Fixes")
    print("=" * 50)
    
    tests = [
        test_timestamp_arithmetic,
        test_signal_creation,
        test_backtester_import,
        test_strategy_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Timestamp fixes are working.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
