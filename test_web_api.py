#!/usr/bin/env python3
"""
Test script for web API endpoints
"""
import requests
import json

def test_backtest_api():
    """Test the backtest API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Web API Endpoints")
    print("=" * 50)
    
    try:
        # Test 1: Run backtest for SOLUSDT
        print("1. Testing backtest endpoint...")
        response = requests.post(f"{base_url}/api/backtest/SOLUSDT")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backtest API working!")
            print(f"   Total trades: {result.get('total_trades', 0)}")
            print(f"   Win rate: {result.get('win_rate', 0):.1f}%")
            print(f"   Success: {result.get('success', False)}")
        else:
            print(f"❌ Backtest API failed: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # Test 2: Get backtest results
        print("\n2. Testing get results endpoint...")
        response = requests.get(f"{base_url}/api/backtest/SOLUSDT/results")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Get results API working!")
            if 'report' in result:
                print(f"   Symbol: {result['report'].get('symbol', 'N/A')}")
                print(f"   Total trades: {result['report'].get('total_trades', 0)}")
            else:
                print(f"   Message: {result.get('message', 'N/A')}")
        else:
            print(f"❌ Get results API failed: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # Test 3: Get status
        print("\n3. Testing status endpoint...")
        response = requests.get(f"{base_url}/api/status")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Status API working!")
            print(f"   Running: {result.get('is_running', False)}")
            print(f"   Symbols: {result.get('symbols', [])}")
        else:
            print(f"❌ Status API failed: {response.status_code}")
        
        print("\n" + "=" * 50)
        print("API testing completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web server")
        print("   Make sure web server is running: python web/start-web.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_backtest_api()
