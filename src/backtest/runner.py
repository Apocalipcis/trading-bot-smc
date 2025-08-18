"""
Asynchronous backtest runner with queue management
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from config.models import PairConfig

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Manages asynchronous backtest execution with concurrency limits"""
    
    def __init__(self, concurrent_limit: int = 2):
        self.concurrent_limit = concurrent_limit
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running_backtests: Dict[str, asyncio.Task] = {}
        self.results_cache: Dict[str, Dict[str, Any]] = {}
        
        # Start queue processor
        self._processor_task = asyncio.create_task(self._process_queue())
    
    async def run_backtest(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule and run backtest for a symbol"""
        # Check if already running
        if symbol in self.running_backtests and not self.running_backtests[symbol].done():
            logger.info(f"Backtest for {symbol} is already running")
            return {"success": False, "error": "Backtest already running"}
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, config)
        if cache_key in self.results_cache:
            cached_result = self.results_cache[cache_key]
            # Check if cache is recent (within 1 hour)
            if 'timestamp' in cached_result:
                cache_time = datetime.fromisoformat(cached_result['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=1):
                    logger.info(f"Returning cached backtest result for {symbol}")
                    return cached_result
        
        # Create backtest task
        future = asyncio.Future()
        request = {
            'symbol': symbol,
            'config': config,
            'future': future
        }
        
        # Add to queue
        await self.queue.put(request)
        
        # Wait for result
        try:
            result = await future
            
            # Cache result
            result['timestamp'] = datetime.now().isoformat()
            self.results_cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_queue(self):
        """Process backtest queue with concurrency limit"""
        while True:
            try:
                # Wait for request
                request = await self.queue.get()
                
                # Wait for available slot
                while len(self.running_backtests) >= self.concurrent_limit:
                    # Remove completed tasks
                    completed = [
                        symbol for symbol, task in self.running_backtests.items()
                        if task.done()
                    ]
                    for symbol in completed:
                        del self.running_backtests[symbol]
                    
                    if len(self.running_backtests) >= self.concurrent_limit:
                        await asyncio.sleep(1)
                
                # Start backtest
                symbol = request['symbol']
                config = request['config']
                future = request['future']
                
                task = asyncio.create_task(self._run_single_backtest(symbol, config, future))
                self.running_backtests[symbol] = task
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Backtest queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in backtest queue processor: {e}")
                await asyncio.sleep(1)
    
    async def _run_single_backtest(self, symbol: str, config: Dict[str, Any], future: asyncio.Future):
        """Run a single backtest"""
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            # Import backtest function
            from ..pre_trade_backtest import run_symbol_backtest
            
            # Run in thread to avoid blocking
            result = await asyncio.to_thread(run_symbol_backtest, symbol, config)
            
            if not future.done():
                future.set_result(result)
            
            logger.info(f"Backtest completed for {symbol}: success={result.get('success', False)}")
            
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            if not future.done():
                future.set_exception(e)
    
    def _get_cache_key(self, symbol: str, config: Dict[str, Any]) -> str:
        """Generate cache key for backtest"""
        import hashlib
        import json
        
        config_str = json.dumps(config, sort_keys=True)
        cache_data = f"{symbol}_{config_str}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    async def stop(self):
        """Stop the backtest runner"""
        # Cancel processor
        self._processor_task.cancel()
        try:
            await self._processor_task
        except asyncio.CancelledError:
            pass
        
        # Cancel running backtests
        for symbol, task in self.running_backtests.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.running_backtests.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get backtest runner status"""
        running_symbols = [
            symbol for symbol, task in self.running_backtests.items()
            if not task.done()
        ]
        
        return {
            'concurrent_limit': self.concurrent_limit,
            'queue_size': self.queue.qsize(),
            'running_backtests': len(running_symbols),
            'running_symbols': running_symbols,
            'cache_size': len(self.results_cache)
        }


# Global backtest runner instance
_backtest_runner: Optional[BacktestRunner] = None


def get_backtest_runner(concurrent_limit: int = 2) -> BacktestRunner:
    """Get or create global backtest runner"""
    global _backtest_runner
    
    if _backtest_runner is None:
        _backtest_runner = BacktestRunner(concurrent_limit)
    
    return _backtest_runner


async def run_backtest_async(symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to run backtest asynchronously"""
    runner = get_backtest_runner()
    return await runner.run_backtest(symbol, config)
