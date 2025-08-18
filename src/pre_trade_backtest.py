"""
Pre-trade backtest module for SMC trading bot
Automatically runs backtest before starting live trading
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .backtester import BacktestValidator, TradeResult
from .signal_generator import generate_signals
from .data_loader import load_csv, prepare_data

logger = logging.getLogger(__name__)

class PreTradeBacktest:
    """Runs comprehensive backtest before starting live trading"""
    
    def __init__(self, symbol: str, config: Dict):
        """
        Initialize pre-trade backtest
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            config: Configuration dictionary with parameters
        """
        self.symbol = symbol.upper()
        self.config = config
        self.results_dir = Path("backtest_results") / self.symbol
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Default backtest parameters
        self.backtest_days = config.get('backtest_days', 30)
        self.ltf_timeframe = config.get('ltf_timeframe', '15m')
        self.htf_timeframe = config.get('htf_timeframe', '4h')
        
        # Data paths
        self.data_dir = Path("data")
        self.ltf_file = self.data_dir / f"{self.symbol.lower()}_{self.ltf_timeframe}_{self.backtest_days}d.csv"
        self.htf_file = self.data_dir / f"{self.symbol.lower()}_{self.htf_timeframe}_{self.backtest_days}d.csv"
        
    def check_data_availability(self) -> bool:
        """Check if required data files exist"""
        if not self.ltf_file.exists():
            logger.warning(f"LTF data file not found: {self.ltf_file}")
            return False
        if not self.htf_file.exists():
            logger.warning(f"HTF data file not found: {self.htf_file}")
            return False
        return True
    
    def download_missing_data(self) -> bool:
        """Download missing data if needed"""
        try:
            from .data_downloader import BinanceDataDownloader
            
            logger.info(f"Downloading missing data for {self.symbol}")
            
            downloader = BinanceDataDownloader()
            
            # Download LTF data
            if not self.ltf_file.exists():
                logger.info(f"Downloading {self.ltf_timeframe} data for {self.symbol}")
                start_date = (datetime.now() - timedelta(days=self.backtest_days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                df = downloader.download_data(
                    symbol=self.symbol,
                    timeframe=self.ltf_timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                df.to_csv(self.ltf_file, index=False)
                logger.info(f"Saved {len(df)} rows to {self.ltf_file}")
            
            # Download HTF data
            if not self.htf_file.exists():
                logger.info(f"Downloading {self.htf_timeframe} data for {self.symbol}")
                start_date = (datetime.now() - timedelta(days=self.backtest_days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                df = downloader.download_data(
                    symbol=self.symbol,
                    timeframe=self.htf_timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                df.to_csv(self.htf_file, index=False)
                logger.info(f"Saved {len(df)} rows to {self.htf_file}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return False
    
    def run_backtest(self) -> Dict:
        """
        Run complete backtest for the symbol
        
        Returns:
            Dictionary with backtest results and statistics
        """
        logger.info(f"Starting backtest for {self.symbol}")
        
        # Check and download data if needed
        if not self.check_data_availability():
            if not self.download_missing_data():
                return {
                    'success': False,
                    'error': 'Failed to obtain required data',
                    'symbol': self.symbol
                }
        
        try:
            # Load and prepare data
            df_ltf, df_htf = prepare_data(str(self.ltf_file), str(self.htf_file))
            
            # Generate signals
            signals = generate_signals(
                df_ltf=df_ltf,
                df_htf=df_htf,
                left=self.config.get('fractal_left', 2),
                right=self.config.get('fractal_right', 2),
                rr_min=self.config.get('min_risk_reward', 3.0)
            )
            
            if signals.empty:
                return {
                    'success': True,
                    'symbol': self.symbol,
                    'total_signals': 0,
                    'message': 'No signals generated for the period',
                    'recommendation': 'Consider adjusting parameters or timeframe'
                }
            
            # Validate signals using backtester
            validator = BacktestValidator(df_ltf)
            results = validator.validate_all_signals(signals)
            
            # Generate comprehensive report
            report = validator.generate_report(results)
            
            # Add additional metrics
            report.update({
                'symbol': self.symbol,
                'backtest_period_days': self.backtest_days,
                'ltf_timeframe': self.ltf_timeframe,
                'htf_timeframe': self.htf_timeframe,
                'data_start': df_ltf['timestamp'].min().isoformat(),
                'data_end': df_ltf['timestamp'].max().isoformat(),
                'total_candles': len(df_ltf),
                'success': True
            })
            
            # Generate recommendations
            report['recommendation'] = self._generate_recommendation(report)
            report['risk_level'] = self._assess_risk_level(report)
            
            # Save results
            self._save_results(report, results)
            
            return report
            
        except Exception as e:
            logger.error(f"Backtest failed for {self.symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': self.symbol
            }
    
    def _generate_recommendation(self, report: Dict) -> str:
        """Generate trading recommendation based on backtest results"""
        win_rate = report.get('win_rate', 0)
        profit_factor = report.get('profit_factor', 0)
        total_trades = report.get('total_trades', 0)
        
        if total_trades < 10:
            return "Insufficient data for reliable recommendation"
        
        if win_rate >= 70 and profit_factor >= 2.0:
            return "EXCELLENT - Strong strategy performance, safe to trade"
        elif win_rate >= 60 and profit_factor >= 1.5:
            return "GOOD - Strategy shows promise, consider trading"
        elif win_rate >= 50 and profit_factor >= 1.2:
            return "FAIR - Strategy needs improvement, trade with caution"
        else:
            return "POOR - Strategy underperforming, avoid trading"
    
    def _assess_risk_level(self, report: Dict) -> str:
        """Assess overall risk level"""
        win_rate = report.get('win_rate', 0)
        profit_factor = report.get('profit_factor', 0)
        sl_rate = report.get('sl_rate', 0)
        
        if win_rate >= 70 and profit_factor >= 2.0 and sl_rate <= 25:
            return "LOW"
        elif win_rate >= 60 and profit_factor >= 1.5 and sl_rate <= 35:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _save_results(self, report: Dict, results: List[TradeResult]):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save detailed results
        results_data = []
        for result in results:
            results_data.append({
                'timestamp': str(result.timestamp),  # Convert to string
                'direction': result.direction,
                'entry': result.entry,
                'sl': result.sl,
                'tp': result.tp,
                'exit_price': result.exit_price,
                'exit_reason': result.exit_reason,
                'pnl': result.pnl,
                'duration_minutes': result.duration_minutes,
                'max_favorable': result.max_favorable,
                'max_adverse': result.max_adverse
            })
        
        # Save to JSON
        latest_file = self.results_dir / "latest_results.json"
        history_file = self.results_dir / "history" / f"{timestamp}.json"
        history_file.parent.mkdir(exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        report_for_json = report.copy()
        if 'data_start' in report_for_json:
            report_for_json['data_start'] = str(report_for_json['data_start'])
        if 'data_end' in report_for_json:
            report_for_json['data_end'] = str(report_for_json['data_end'])
        
        with open(latest_file, 'w') as f:
            json.dump({
                'report': report_for_json,
                'results': results_data,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)
        
        with open(history_file, 'w') as f:
            json.dump({
                'report': report_for_json,
                'results': results_data,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save summary to CSV
        summary_file = self.results_dir / "summary.csv"
        summary_data = {
            'timestamp': [timestamp],
            'symbol': [self.symbol],
            'total_trades': [report['total_trades']],
            'win_rate': [report['win_rate']],
            'profit_factor': [report['profit_factor']],
            'total_pnl': [report['total_pnl']],
            'risk_level': [report['risk_level']],
            'recommendation': [report['recommendation']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        if summary_file.exists():
            existing_df = pd.read_csv(summary_file)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def get_latest_results(self) -> Optional[Dict]:
        """Get latest backtest results"""
        latest_file = self.results_dir / "latest_results.json"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all backtests"""
        summary_file = self.results_dir / "summary.csv"
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()

def run_symbol_backtest(symbol: str, config: Dict) -> Dict:
    """
    Convenience function to run backtest for a symbol
    
    Args:
        symbol: Trading symbol
        config: Configuration dictionary
        
    Returns:
        Backtest results dictionary
    """
    backtest = PreTradeBacktest(symbol, config)
    return backtest.run_backtest()

def run_multiple_backtests(symbols: List[str], config: Dict) -> Dict[str, Dict]:
    """
    Run backtests for multiple symbols
    
    Args:
        symbols: List of trading symbols
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping symbols to their backtest results
    """
    results = {}
    
    for symbol in symbols:
        logger.info(f"Running backtest for {symbol}")
        result = run_symbol_backtest(symbol, config)
        results[symbol] = result
        
        # Small delay between backtests
        import time
        time.sleep(1)
    
    return results

if __name__ == "__main__":
    # Test the module
    config = {
        'min_risk_reward': 3.0,
        'fractal_left': 2,
        'fractal_right': 2,
        'backtest_days': 30
    }
    
    result = run_symbol_backtest("SOLUSDT", config)
    print(json.dumps(result, indent=2))
