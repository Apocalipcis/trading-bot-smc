"""
Backtest validator for SMC trading signals
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeResult:
    """Result of a single trade validation"""
    timestamp: str
    direction: str
    entry: float
    sl: float
    tp: float
    exit_price: float
    exit_reason: str  # 'SL', 'TP', 'NO_EXIT'
    exit_time: str
    pnl: float
    duration_minutes: int
    max_favorable: float  # Best price reached
    max_adverse: float    # Worst price reached

class BacktestValidator:
    """Validates trading signals against actual price movements"""
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize with price data
        
        Args:
            price_data: DataFrame with OHLC data
        """
        self.price_data = price_data.copy()
        self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp']).dt.tz_localize(None)
        self.price_data = self.price_data.sort_values('timestamp').reset_index(drop=True)
    
    def validate_signal(self, signal: Dict) -> TradeResult:
        """
        Validate a single trading signal
        
        Args:
            signal: Dictionary with signal data
            
        Returns:
            TradeResult with validation outcome
        """
        entry_time = pd.to_datetime(signal['timestamp'])
        if entry_time.tz is not None:
            entry_time = entry_time.tz_convert('UTC').tz_localize(None)
        entry_price = float(signal['entry'])
        sl_price = float(signal['sl'])
        tp_price = float(signal['tp'])
        direction = signal['direction']
        
        # Find entry candle
        entry_idx = None
        for i, row in self.price_data.iterrows():
            if row['timestamp'] >= entry_time:
                entry_idx = i
                break
        
        if entry_idx is None:
            return TradeResult(
                timestamp=signal['timestamp'],
                direction=direction,
                entry=entry_price,
                sl=sl_price,
                tp=tp_price,
                exit_price=entry_price,
                exit_reason='NO_DATA',
                exit_time=signal['timestamp'],
                pnl=0.0,
                duration_minutes=0,
                max_favorable=entry_price,
                max_adverse=entry_price
            )
        
        # Check subsequent candles for SL/TP hits
        max_favorable = entry_price
        max_adverse = entry_price
        
        for i in range(entry_idx, len(self.price_data)):
            candle = self.price_data.iloc[i]
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            
            if direction == 'LONG':
                # Update extremes
                max_favorable = max(max_favorable, high)
                max_adverse = min(max_adverse, low)
                
                # Check SL first (more conservative)
                if low <= sl_price:
                    duration = int((candle['timestamp'] - entry_time).total_seconds() / 60)
                    pnl = sl_price - entry_price
                    return TradeResult(
                        timestamp=signal['timestamp'],
                        direction=direction,
                        entry=entry_price,
                        sl=sl_price,
                        tp=tp_price,
                        exit_price=sl_price,
                        exit_reason='SL',
                        exit_time=str(candle['timestamp']),
                        pnl=pnl,
                        duration_minutes=duration,
                        max_favorable=max_favorable,
                        max_adverse=max_adverse
                    )
                
                # Check TP
                if high >= tp_price:
                    duration = int((candle['timestamp'] - entry_time).total_seconds() / 60)
                    pnl = tp_price - entry_price
                    return TradeResult(
                        timestamp=signal['timestamp'],
                        direction=direction,
                        entry=entry_price,
                        sl=sl_price,
                        tp=tp_price,
                        exit_price=tp_price,
                        exit_reason='TP',
                        exit_time=str(candle['timestamp']),
                        pnl=pnl,
                        duration_minutes=duration,
                        max_favorable=max_favorable,
                        max_adverse=max_adverse
                    )
            
            else:  # SHORT
                # Update extremes
                max_favorable = min(max_favorable, low)
                max_adverse = max(max_adverse, high)
                
                # Check SL first
                if high >= sl_price:
                    duration = int((candle['timestamp'] - entry_time).total_seconds() / 60)
                    pnl = entry_price - sl_price
                    return TradeResult(
                        timestamp=signal['timestamp'],
                        direction=direction,
                        entry=entry_price,
                        sl=sl_price,
                        tp=tp_price,
                        exit_price=sl_price,
                        exit_reason='SL',
                        exit_time=str(candle['timestamp']),
                        pnl=pnl,
                        duration_minutes=duration,
                        max_favorable=max_favorable,
                        max_adverse=max_adverse
                    )
                
                # Check TP
                if low <= tp_price:
                    duration = int((candle['timestamp'] - entry_time).total_seconds() / 60)
                    pnl = entry_price - tp_price
                    return TradeResult(
                        timestamp=signal['timestamp'],
                        direction=direction,
                        entry=entry_price,
                        sl=sl_price,
                        tp=tp_price,
                        exit_price=tp_price,
                        exit_reason='TP',
                        exit_time=str(candle['timestamp']),
                        pnl=pnl,
                        duration_minutes=duration,
                        max_favorable=max_favorable,
                        max_adverse=max_adverse
                    )
        
        # No exit found - still running
        final_candle = self.price_data.iloc[-1]
        final_price = float(final_candle['close'])
        duration = int((final_candle['timestamp'] - entry_time).total_seconds() / 60)
        
        if direction == 'LONG':
            pnl = final_price - entry_price
        else:
            pnl = entry_price - final_price
        
        return TradeResult(
            timestamp=signal['timestamp'],
            direction=direction,
            entry=entry_price,
            sl=sl_price,
            tp=tp_price,
            exit_price=final_price,
            exit_reason='NO_EXIT',
            exit_time=str(final_candle['timestamp']),
            pnl=pnl,
            duration_minutes=duration,
            max_favorable=max_favorable,
            max_adverse=max_adverse
        )
    
    def validate_all_signals(self, signals_df: pd.DataFrame) -> List[TradeResult]:
        """Validate all signals in the dataframe"""
        results = []
        
        print(f"Validating {len(signals_df)} signals...")
        
        for _, signal in signals_df.iterrows():
            result = self.validate_signal(signal.to_dict())
            results.append(result)
            
            # Progress indicator
            if len(results) % 5 == 0:
                print(f"Processed {len(results)}/{len(signals_df)} signals...")
        
        return results
    
    def generate_report(self, results: List[TradeResult]) -> Dict:
        """Generate comprehensive backtest report"""
        if not results:
            return {}
        
        total_trades = len(results)
        winning_trades = [r for r in results if r.pnl > 0]
        losing_trades = [r for r in results if r.pnl < 0]
        sl_trades = [r for r in results if r.exit_reason == 'SL']
        tp_trades = [r for r in results if r.exit_reason == 'TP']
        no_exit_trades = [r for r in results if r.exit_reason == 'NO_EXIT']
        
        total_pnl = sum(r.pnl for r in results)
        win_rate = len(winning_trades) / total_trades * 100
        avg_win = np.mean([r.pnl for r in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([r.pnl for r in losing_trades]) if losing_trades else 0
        
        # Duration analysis
        completed_trades = [r for r in results if r.exit_reason in ['SL', 'TP']]
        avg_duration = np.mean([r.duration_minutes for r in completed_trades]) if completed_trades else 0
        
        report = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'sl_hits': len(sl_trades),
            'tp_hits': len(tp_trades),
            'no_exit': len(no_exit_trades),
            'sl_rate': len(sl_trades) / total_trades * 100,
            'tp_rate': len(tp_trades) / total_trades * 100,
            'avg_duration_minutes': avg_duration,
            'avg_duration_hours': avg_duration / 60
        }
        
        return report

def run_backtest_validation(signals_file: str, price_data_file: str) -> None:
    """
    Run complete backtest validation
    
    Args:
        signals_file: Path to signals CSV
        price_data_file: Path to price data CSV
    """
    print("🔍 SMC Backtest Validator")
    print("=" * 50)
    
    # Load data
    print(f"Loading signals from: {signals_file}")
    signals_df = pd.read_csv(signals_file)
    
    print(f"Loading price data from: {price_data_file}")
    price_df = pd.read_csv(price_data_file)
    
    # Initialize validator
    validator = BacktestValidator(price_df)
    
    # Validate signals
    results = validator.validate_all_signals(signals_df)
    
    # Generate report
    report = validator.generate_report(results)
    
    # Print detailed results
    print("\n" + "=" * 50)
    print("📊 BACKTEST RESULTS")
    print("=" * 50)
    
    print(f"Total Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate']:.1f}%")
    print(f"Total P&L: ${report['total_pnl']:.2f}")
    print(f"Average Win: ${report['avg_win']:.2f}")
    print(f"Average Loss: ${report['avg_loss']:.2f}")
    print(f"Profit Factor: {report['profit_factor']:.2f}")
    
    print(f"\n📈 Exit Analysis:")
    print(f"TP Hits: {report['tp_hits']} ({report['tp_rate']:.1f}%)")
    print(f"SL Hits: {report['sl_hits']} ({report['sl_rate']:.1f}%)")
    print(f"No Exit: {report['no_exit']}")
    
    print(f"\n⏱️ Timing:")
    print(f"Average Duration: {report['avg_duration_hours']:.1f} hours")
    
    # Save detailed results
    results_df = pd.DataFrame([{
        'timestamp': r.timestamp,
        'direction': r.direction,
        'entry': r.entry,
        'exit_price': r.exit_price,
        'exit_reason': r.exit_reason,
        'pnl': r.pnl,
        'duration_hours': r.duration_minutes / 60,
        'max_favorable': r.max_favorable,
        'max_adverse': r.max_adverse
    } for r in results])
    
    results_df.to_csv('backtest_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: backtest_results.csv")
    
    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.backtester signals.csv price_data.csv")
        sys.exit(1)
    
    run_backtest_validation(sys.argv[1], sys.argv[2])
