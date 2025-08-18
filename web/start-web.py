#!/usr/bin/env python3
"""
SMC Trading Bot - Web Interface Launcher
Simple launcher for development and testing
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Load environment variables from .env if exists
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('\'"')
                os.environ.setdefault(key, value)

# Start the web interface
if __name__ == "__main__":
    import uvicorn
    from web.backend.main import app
    
    print("🚀 Starting SMC Trading Bot - Web Interface")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("⚡ Press Ctrl+C to stop\n")
    
    # Run with auto-reload for development
    uvicorn.run(
        "web.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "web"), str(project_root / "src")]
    )
