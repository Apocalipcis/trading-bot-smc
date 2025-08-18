#!/usr/bin/env python3
"""
Docker Runner for SMC Trading Bot
Interactive menu to run the bot in Docker container
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        return False

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not found or not running")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run("docker-compose --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker Compose not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker Compose: {e}")
        return False

def build_image():
    """Build Docker image"""
    return run_command("docker build -t smc-trading-bot .", "Building Docker image")

def start_services():
    """Start all services with docker-compose"""
    return run_command("docker-compose up -d", "Starting services")

def stop_services():
    """Stop all services"""
    return run_command("docker-compose down", "Stopping services")

def view_logs():
    """View container logs"""
    return run_command("docker-compose logs -f smc-trading-bot", "Viewing logs")

def run_backtest(symbol, days=30):
    """Run backtest for specific symbol in container"""
    cmd = f'docker exec -it smc-trading-bot python live_trading.py --symbol {symbol} --run-backtest --backtest-days {days}'
    return run_command(cmd, f"Running backtest for {symbol}")

def run_live_trading(symbol, rr=3.0, interval=45):
    """Run live trading for specific symbol in container"""
    cmd = f'docker exec -it smc-trading-bot python live_trading.py --symbol {symbol} --rr {rr} --status-check-interval {interval} --desktop-alerts'
    return run_command(cmd, f"Starting live trading for {symbol}")

def show_status():
    """Show container status"""
    return run_command("docker-compose ps", "Container status")

def interactive_menu():
    """Interactive menu for Docker operations"""
    while True:
        print("\n" + "=" * 60)
        print("🐳 SMC Trading Bot - Docker Manager")
        print("=" * 60)
        print("1. Build Docker image")
        print("2. Start services")
        print("3. Stop services")
        print("4. View logs")
        print("5. Show status")
        print("6. Run backtest (ETHUSDT)")
        print("7. Run live trading (ETHUSDT)")
        print("8. Custom backtest")
        print("9. Custom live trading")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (0-9): ").strip()
        
        if choice == "1":
            build_image()
        elif choice == "2":
            start_services()
        elif choice == "3":
            stop_services()
        elif choice == "4":
            view_logs()
        elif choice == "5":
            show_status()
        elif choice == "6":
            run_backtest("ETHUSDT")
        elif choice == "7":
            run_live_trading("ETHUSDT")
        elif choice == "8":
            symbol = input("Enter symbol (e.g., BTCUSDT): ").strip().upper()
            days = input("Enter backtest days (default 30): ").strip()
            days = int(days) if days.isdigit() else 30
            run_backtest(symbol, days)
        elif choice == "9":
            symbol = input("Enter symbol (e.g., BTCUSDT): ").strip().upper()
            rr = input("Enter Risk/Reward (default 3.0): ").strip()
            rr = float(rr) if rr.replace('.', '').isdigit() else 3.0
            interval = input("Enter check interval seconds (default 45): ").strip()
            interval = int(interval) if interval.isdigit() else 45
            run_live_trading(symbol, rr, interval)
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Docker Runner for SMC Trading Bot')
    parser.add_argument('--build', action='store_true', help='Build Docker image')
    parser.add_argument('--start', action='store_true', help='Start services')
    parser.add_argument('--stop', action='store_true', help='Stop services')
    parser.add_argument('--logs', action='store_true', help='View logs')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--backtest', help='Run backtest for symbol')
    parser.add_argument('--live', help='Run live trading for symbol')
    parser.add_argument('--interactive', action='store_true', help='Interactive menu')
    
    args = parser.parse_args()
    
    # Check Docker availability
    if not check_docker():
        print("❌ Please install Docker first!")
        sys.exit(1)
    
    if not check_docker_compose():
        print("❌ Please install Docker Compose first!")
        sys.exit(1)
    
    # Handle command line arguments
    if args.build:
        build_image()
    elif args.start:
        start_services()
    elif args.stop:
        stop_services()
    elif args.logs:
        view_logs()
    elif args.status:
        show_status()
    elif args.backtest:
        run_backtest(args.backtest)
    elif args.live:
        run_live_trading(args.live)
    elif args.interactive:
        interactive_menu()
    else:
        # Default to interactive mode
        interactive_menu()

if __name__ == '__main__':
    main()
