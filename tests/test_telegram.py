#!/usr/bin/env python3
"""
Test Telegram functionality with live credentials
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.telegram_client import TelegramClient

def _load_dotenv(path: str = ".env") -> None:
    """Simple .env loader"""
    env_path = Path(__file__).resolve().parents[1] / path
    try:
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Strip quotes
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        if key not in os.environ:
                            os.environ[key] = value
    except Exception as e:
        print(f"Warning: Failed to load .env: {e}")

def test_telegram_integration():
    """Test live Telegram integration"""
    print("🧪 Testing Telegram Bot Integration")
    print("=" * 50)
    
    # Load environment variables
    _load_dotenv()
    
    # Get credentials
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        print("🔧 Set credentials in .env file to run this test")
        return False
    
    print(f"🔑 Bot Token: {token[:20]}...")
    print(f"💬 Chat ID: {chat_id}")
    print()
    
    # Create client
    client = TelegramClient(token, chat_id)
    
    # Test 1: Connection test
    print("🔌 Testing connection...")
    if client.test_connection():
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed!")
        return False
    
    # Test 2: Signal notification
    print("\n📈 Sending test signal...")
    test_signal = {
        'symbol': 'ETHUSDT',
        'direction': 'LONG',
        'entry': 2450.50,
        'sl': 2430.20,
        'tp': 2510.80,
        'rr': 3.0,
        'htf_bias': 'bullish',
        'fvg_confluence': True
    }
    
    if client.send_signal_notification(test_signal):
        print("✅ Signal notification sent!")
    else:
        print("❌ Failed to send signal notification!")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == '__main__':
    success = test_telegram_integration()
    sys.exit(0 if success else 1)
