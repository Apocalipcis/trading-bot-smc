"""
Main entry point for SMC trading bot
"""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import prepare_data, validate_timeframe_alignment
from src.signal_generator import generate_signals
from config import SMCConfig, update_config

def run_backtest(config: SMCConfig) -> None:
    """
    Run SMC backtest with given configuration
    
    Args:
        config: SMC configuration object
    """
    print("=== SMC Trading Bot - Backtest Mode ===")
    print(f"LTF Data: {config.ltf_data_path}")
    print(f"HTF Data: {config.htf_data_path}")
    print(f"Min RR: {config.min_risk_reward}")
    print("=" * 40)
    
    try:
        # Load data
        df_ltf, df_htf = prepare_data(config.ltf_data_path, config.htf_data_path)
        
        # Validate timeframe alignment
        if not validate_timeframe_alignment(df_ltf, df_htf):
            print("Warning: Timeframe alignment issues detected")
        
        # Generate signals
        print("Generating trading signals...")
        signals = generate_signals(
            df_ltf=df_ltf,
            df_htf=df_htf,
            left=config.fractal_left,
            right=config.fractal_right,
            rr_min=config.min_risk_reward
        )
        
        # Save results
        signals.to_csv(config.output_path, index=False)
        
        # Print summary
        print(f"\n=== RESULTS ===")
        print(f"Generated {len(signals)} signals")
        print(f"Results saved to: {config.output_path}")
        
        if len(signals) > 0:
            long_signals = signals[signals['direction'] == 'LONG']
            short_signals = signals[signals['direction'] == 'SHORT']
            
            print(f"Long signals: {len(long_signals)}")
            print(f"Short signals: {len(short_signals)}")
            print(f"Average RR: {signals['rr'].mean():.2f}")
            print(f"FVG confluence signals: {signals['fvg_confluence'].sum()}")
            
            # Show first few signals
            print(f"\nFirst 3 signals:")
            print(signals[['timestamp', 'direction', 'entry', 'sl', 'tp', 'rr']].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        sys.exit(1)

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='SMC Trading Bot - Smart Money Concepts Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv
  python main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv --rr 3.0 --out my_signals.csv
        """
    )
    
    # Required arguments
    parser.add_argument('--ltf', required=True, 
                       help='Lower timeframe CSV file (e.g., 15m data)')
    parser.add_argument('--htf', required=True,
                       help='Higher timeframe CSV file (e.g., 4h data)')
    
    # Optional arguments
    parser.add_argument('--rr', type=float, default=3.0,
                       help='Minimum Risk/Reward ratio (default: 2.0)')
    parser.add_argument('--left', type=int, default=2,
                       help='Fractal left bars (default: 2)')
    parser.add_argument('--right', type=int, default=2,
                       help='Fractal right bars (default: 2)')
    parser.add_argument('--out', default='signals.csv',
                       help='Output CSV file (default: signals.csv)')
    parser.add_argument('--require-fvg', action='store_true',
                       help='Require FVG confluence for signals')
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.ltf).exists():
        print(f"Error: LTF file not found: {args.ltf}")
        sys.exit(1)
        
    if not Path(args.htf).exists():
        print(f"Error: HTF file not found: {args.htf}")
        sys.exit(1)
    
    # Create configuration
    config = update_config(
        ltf_data_path=args.ltf,
        htf_data_path=args.htf,
        min_risk_reward=args.rr,
        fractal_left=args.left,
        fractal_right=args.right,
        output_path=args.out,
        require_fvg_confluence=args.require_fvg
    )
    
    # Run backtest
    run_backtest(config)

if __name__ == '__main__':
    main()
