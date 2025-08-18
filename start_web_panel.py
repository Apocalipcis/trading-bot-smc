#!/usr/bin/env python3
"""
Launch script for the new SMC Trading Bot Web Panel
"""
import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("🚀 Starting SMC Trading Bot Web Panel...")
    print("=" * 50)
    print("📊 Features:")
    print("  ✅ Individual pair toggles")
    print("  ✅ Backtest toggles per pair")
    print("  ✅ Live trading controls")
    print("  ✅ Real-time signals")
    print("  ✅ Hot config reload")
    print("=" * 50)
    print("🌐 Opening at: http://localhost:8000")
    print("🔧 Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "src.web.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Web panel stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting web panel: {e}")
        sys.exit(1)
