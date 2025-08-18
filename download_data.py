#!/usr/bin/env python3
"""
Simple script to download crypto data and run SMC bot
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_downloader import download_crypto_data

def main():
    print("🚀 SMC Bot Data Downloader")
    print("=" * 40)
    
    # Get user input
    symbol = input("Enter trading pair (default: BTCUSDT): ").strip().upper() or "BTCUSDT"
    days_input = input("Enter days to download (default: 30): ").strip()
    days = int(days_input) if days_input else 30
    
    print(f"\n📊 Downloading {symbol} data for {days} days...")
    
    try:
        # Download data
        ltf_file, htf_file = download_crypto_data(symbol, days)
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Files created:")
        print(f"   LTF (15m): {ltf_file}")
        print(f"   HTF (4h):  {htf_file}")
        
        # Ask if user wants to run the bot
        run_bot = input(f"\n🤖 Run SMC bot now? (y/n): ").strip().lower()
        
        if run_bot in ['y', 'yes']:
            import subprocess
            cmd = f"venv\\Scripts\\python.exe main.py --ltf {ltf_file} --htf {htf_file}"
            print(f"\n🚀 Running: {cmd}")
            subprocess.run(cmd, shell=True)
        else:
            print(f"\n📝 To run bot manually:")
            print(f"   venv\\Scripts\\python.exe main.py --ltf {ltf_file} --htf {htf_file}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main())
