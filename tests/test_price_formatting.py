#!/usr/bin/env python3
"""
Test price formatting for different coin values
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.live_monitor import format_price

def test_price_formatting():
    """Test price formatting for various coin types"""
    
    print("🧪 Testing Price Formatting")
    print("=" * 40)
    
    # Test cases: (price, expected_decimals)
    test_cases = [
        # High value coins
        (123.45, 2),      # BTC-like
        (45.67, 2),       # ETH-like
        
        # Medium value coins  
        (12.3456, 4),     # ADA-like
        (1.2345, 4),      # USDT-like
        (0.9876, 4),      # Stablecoins
        
        # Low value coins
        (0.123456, 6),    # Small altcoins
        (0.012345, 6),    # Medium altcoins
        
        # Very low value (memecoins)
        (0.000023456, 8), # BONK-like
        (0.000000123, 8), # Very small coins
        (0.00000001, 8),  # Minimal values
    ]
    
    for price, expected_decimals in test_cases:
        formatted = format_price(price)
        actual_decimals = len(formatted.split('.')[1]) if '.' in formatted else 0
        
        print(f"Price: {price:>12} → {formatted:>12} (decimals: {actual_decimals})")
        
        # Verify it shows enough precision for trading
        if price < 0.01:
            assert actual_decimals >= 6, f"Low value coins need more precision: {formatted}"
        
    print("\n✅ All price formatting tests passed!")
    
    # Test with BONK-like prices
    print("\n🪙 BONK Price Examples:")
    bonk_prices = [0.000023456, 0.000034567, 0.000012345]
    for price in bonk_prices:
        print(f"BONK: ${format_price(price)}")

if __name__ == '__main__':
    test_price_formatting()
