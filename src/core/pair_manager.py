"""
Pair Manager for handling multiple trading pairs with concurrency limits
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime

from config.models import AppConfig, PairConfig, PairStatus, Signal
from .exchange_gateway import ExchangeGateway
from .strategy import Strategy, StrategyFactory

logger = logging.getLogger(__name__)


class PairWorker:
    """Worker for handling a single trading pair"""
    
    def __init__(self, pair_config: PairConfig, gateway: ExchangeGateway, strategy: Strategy):
        self.config = pair_config
        self.gateway = gateway
        self.strategy = strategy
        self.symbol = pair_config.symbol
        
        # State
        self.status = PairStatus(
            symbol=self.symbol,
            enabled=pair_config.enabled,
            backtest_enabled=pair_config.backtest_enabled,
            status='stopped'
        )
        
        self.task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # Callbacks
        self.signal_callbacks: List[Callable[[Signal], None]] = []
        self.status_callbacks: List[Callable[[PairStatus], None]] = []
        
        # Data
        self.ltf_candles = []
        self.htf_candles = []
        self.last_update = datetime.now()
    
    async def start(self):
        """Start the pair worker"""
        if self.task and not self.task.done():
            logger.warning(f"Worker for {self.symbol} is already running")
            return
        
        self.stop_event.clear()
        self.status.status = 'starting'
        self._notify_status_callbacks()
        
        self.task = asyncio.create_task(self._run())
        logger.info(f"Started worker for {self.symbol}")
    
    async def stop(self):
        """Stop the pair worker"""
        if not self.task or self.task.done():
            return
        
        self.stop_event.set()
        
        try:
            await asyncio.wait_for(self.task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Worker for {self.symbol} didn't stop gracefully, cancelling")
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        self.status.status = 'stopped'
        self._notify_status_callbacks()
        logger.info(f"Stopped worker for {self.symbol}")
    
    async def _run(self):
        """Main worker loop"""
        try:
            self.status.status = 'running'
            self._notify_status_callbacks()
            
            # Initial data fetch
            await self._fetch_initial_data()
            
            # Main loop
            while not self.stop_event.is_set():
                try:
                    # Update data
                    await self._update_data()
                    
                    # Generate signals
                    if len(self.ltf_candles) >= 50 and len(self.htf_candles) >= 20:
                        signals = await self.strategy.generate_signals(self.ltf_candles, self.htf_candles)
                        
                        for signal in signals:
                            self._notify_signal_callbacks(signal)
                    
                    # Update status
                    self.status.current_price = await self.gateway.get_current_price(self.symbol)
                    self.status.htf_bias = getattr(self.strategy, 'htf_bias', 'neutral')
                    self.last_update = datetime.now()
                    
                    self._notify_status_callbacks()
                    
                    # Wait before next iteration
                    await asyncio.sleep(30)  # 30 seconds between updates
                    
                except Exception as e:
                    logger.error(f"Error in worker loop for {self.symbol}: {e}")
                    self.status.last_error = str(e)
                    self.status.status = 'error'
                    self._notify_status_callbacks()
                    
                    # Wait before retrying
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info(f"Worker for {self.symbol} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in worker for {self.symbol}: {e}")
            self.status.status = 'error'
            self.status.last_error = str(e)
            self._notify_status_callbacks()
    
    async def _fetch_initial_data(self):
        """Fetch initial historical data"""
        try:
            # Fetch LTF and HTF data
            self.ltf_candles = await self.gateway.fetch_candles(self.symbol, '15m', 500)
            self.htf_candles = await self.gateway.fetch_candles(self.symbol, '4h', 200)
            
            logger.info(f"Fetched initial data for {self.symbol}: LTF={len(self.ltf_candles)}, HTF={len(self.htf_candles)}")
            
        except Exception as e:
            logger.error(f"Failed to fetch initial data for {self.symbol}: {e}")
            raise
    
    async def _update_data(self):
        """Update candlestick data"""
        try:
            # Fetch latest candles (just a few to update)
            new_ltf = await self.gateway.fetch_candles(self.symbol, '15m', 50)
            new_htf = await self.gateway.fetch_candles(self.symbol, '4h', 20)
            
            # Update candle lists (keep last 500 LTF, 200 HTF)
            if new_ltf:
                self.ltf_candles = new_ltf[-500:]
            if new_htf:
                self.htf_candles = new_htf[-200:]
                
        except Exception as e:
            logger.error(f"Failed to update data for {self.symbol}: {e}")
    
    def add_signal_callback(self, callback: Callable[[Signal], None]):
        """Add signal callback"""
        self.signal_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[PairStatus], None]):
        """Add status callback"""
        self.status_callbacks.append(callback)
    
    def _notify_signal_callbacks(self, signal: Signal):
        """Notify all signal callbacks"""
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")
    
    def _notify_status_callbacks(self):
        """Notify all status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")


class PairManager:
    """Manages multiple trading pairs with concurrency limits"""
    
    def __init__(self, gateway: ExchangeGateway, config: AppConfig):
        self.gateway = gateway
        self.config = config
        
        # Workers
        self.workers: Dict[str, PairWorker] = {}
        self.running_workers: Set[str] = set()
        
        # Callbacks
        self.signal_callbacks: List[Callable[[Signal], None]] = []
        self.status_callbacks: List[Callable[[Dict[str, PairStatus]], None]] = []
        
        # Backtest management
        self.backtest_tasks: Dict[str, asyncio.Task] = {}
        self.backtest_queue = asyncio.Queue(maxsize=config.backtest_concurrent_limit)
        
        logger.info(f"PairManager initialized with max {config.max_concurrent_pairs} concurrent pairs")
    
    async def start(self):
        """Start the pair manager"""
        logger.info("Starting PairManager")
        
        # Create workers for all configured pairs
        for pair_config in self.config.pairs:
            await self._create_worker(pair_config)
        
        # Start enabled pairs
        await self._start_enabled_pairs()
        
        # Start backtest queue processor
        asyncio.create_task(self._process_backtest_queue())
        
    async def stop(self):
        """Stop all workers and backtests"""
        logger.info("Stopping PairManager")
        
        # Stop all workers
        tasks = []
        for worker in self.workers.values():
            tasks.append(worker.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cancel backtest tasks
        for symbol, task in self.backtest_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.workers.clear()
        self.running_workers.clear()
        self.backtest_tasks.clear()
    
    async def apply_config(self, new_config: AppConfig):
        """Apply new configuration hot reload"""
        logger.info("Applying new configuration")
        
        # Validate new config
        errors = new_config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        old_config = self.config
        self.config = new_config
        
        try:
            # Handle removed pairs
            old_symbols = {pair.symbol for pair in old_config.pairs}
            new_symbols = {pair.symbol for pair in new_config.pairs}
            
            removed_symbols = old_symbols - new_symbols
            for symbol in removed_symbols:
                await self._remove_pair(symbol)
            
            # Handle new pairs
            added_symbols = new_symbols - old_symbols
            for symbol in added_symbols:
                pair_config = new_config.get_pair_config(symbol)
                if pair_config:
                    await self._create_worker(pair_config)
            
            # Update existing pairs
            for pair_config in new_config.pairs:
                symbol = pair_config.symbol
                if symbol in self.workers:
                    await self._update_worker_config(symbol, pair_config)
            
            # Start/stop workers based on enabled status
            await self._start_enabled_pairs()
            await self._stop_disabled_pairs()
            
            # Trigger backtests for newly enabled backtest pairs
            await self._schedule_backtests()
            
            self._notify_status_callbacks()
            
        except Exception as e:
            # Rollback on error
            logger.error(f"Error applying config, rolling back: {e}")
            self.config = old_config
            raise
    
    async def _create_worker(self, pair_config: PairConfig):
        """Create a worker for a pair"""
        symbol = pair_config.symbol
        
        if symbol in self.workers:
            logger.warning(f"Worker for {symbol} already exists")
            return
        
        try:
            # Create strategy
            strategy = StrategyFactory.create_strategy(
                pair_config.strategy, 
                self.gateway, 
                pair_config
            )
            
            # Create worker
            worker = PairWorker(pair_config, self.gateway, strategy)
            
            # Add callbacks
            worker.add_signal_callback(self._on_signal)
            worker.add_status_callback(self._on_status_update)
            
            self.workers[symbol] = worker
            
            logger.info(f"Created worker for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to create worker for {symbol}: {e}")
            raise
    
    async def _remove_pair(self, symbol: str):
        """Remove a pair and stop its worker"""
        if symbol in self.workers:
            worker = self.workers[symbol]
            await worker.stop()
            del self.workers[symbol]
            
            if symbol in self.running_workers:
                self.running_workers.remove(symbol)
            
            logger.info(f"Removed worker for {symbol}")
    
    async def _update_worker_config(self, symbol: str, new_config: PairConfig):
        """Update worker configuration"""
        if symbol not in self.workers:
            return
        
        worker = self.workers[symbol]
        old_enabled = worker.config.enabled
        old_backtest_enabled = worker.config.backtest_enabled
        
        # Update config
        worker.config = new_config
        worker.status.enabled = new_config.enabled
        worker.status.backtest_enabled = new_config.backtest_enabled
        
        # Recreate strategy if needed
        if (new_config.strategy != worker.strategy.get_strategy_name() or
            new_config.min_risk_reward != worker.strategy.config.min_risk_reward):
            
            worker.strategy = StrategyFactory.create_strategy(
                new_config.strategy,
                self.gateway,
                new_config
            )
        
        # Schedule backtest if newly enabled
        if not old_backtest_enabled and new_config.backtest_enabled:
            await self._schedule_backtest(symbol)
    
    async def _start_enabled_pairs(self):
        """Start workers for enabled pairs"""
        enabled_pairs = self.config.get_enabled_pairs()
        
        # Respect concurrent limit
        pairs_to_start = enabled_pairs[:self.config.max_concurrent_pairs]
        
        for pair_config in pairs_to_start:
            symbol = pair_config.symbol
            if symbol in self.workers and symbol not in self.running_workers:
                worker = self.workers[symbol]
                await worker.start()
                self.running_workers.add(symbol)
        
        # Queue remaining pairs
        for pair_config in enabled_pairs[self.config.max_concurrent_pairs:]:
            symbol = pair_config.symbol
            if symbol in self.workers:
                self.workers[symbol].status.status = 'queued'
    
    async def _stop_disabled_pairs(self):
        """Stop workers for disabled pairs"""
        enabled_symbols = {pair.symbol for pair in self.config.get_enabled_pairs()}
        
        symbols_to_stop = self.running_workers - enabled_symbols
        
        for symbol in symbols_to_stop:
            if symbol in self.workers:
                worker = self.workers[symbol]
                await worker.stop()
                self.running_workers.remove(symbol)
    
    async def _schedule_backtests(self):
        """Schedule backtests for enabled pairs"""
        backtest_pairs = self.config.get_backtest_enabled_pairs()
        
        for pair_config in backtest_pairs:
            symbol = pair_config.symbol
            if symbol not in self.backtest_tasks or self.backtest_tasks[symbol].done():
                await self._schedule_backtest(symbol)
    
    async def _schedule_backtest(self, symbol: str):
        """Schedule a backtest for a symbol"""
        if symbol in self.backtest_tasks and not self.backtest_tasks[symbol].done():
            logger.warning(f"Backtest for {symbol} is already running")
            return
        
        # Create backtest task
        task = asyncio.create_task(self._run_backtest(symbol))
        self.backtest_tasks[symbol] = task
        
        # Update status
        if symbol in self.workers:
            self.workers[symbol].status.backtest_running = True
    
    async def _run_backtest(self, symbol: str):
        """Run backtest for a symbol"""
        try:
            # Wait for slot in backtest queue
            await self.backtest_queue.put(symbol)
            
            logger.info(f"Starting backtest for {symbol}")
            
            # Import and run backtest
            from ..pre_trade_backtest import run_symbol_backtest
            
            pair_config = self.config.get_pair_config(symbol)
            if not pair_config:
                logger.error(f"No config found for {symbol}")
                return
            
            config_dict = {
                'min_risk_reward': pair_config.min_risk_reward,
                'fractal_left': pair_config.fractal_left,
                'fractal_right': pair_config.fractal_right,
                'backtest_days': 30
            }
            
            # Run backtest in thread to avoid blocking
            result = await asyncio.to_thread(run_symbol_backtest, symbol, config_dict)
            
            logger.info(f"Backtest completed for {symbol}: {result.get('success', False)}")
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
        finally:
            # Release queue slot
            try:
                self.backtest_queue.get_nowait()
                self.backtest_queue.task_done()
            except asyncio.QueueEmpty:
                pass
            
            # Update status
            if symbol in self.workers:
                self.workers[symbol].status.backtest_running = False
    
    async def _process_backtest_queue(self):
        """Process backtest queue to maintain concurrency limit"""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
    
    def _on_signal(self, signal: Signal):
        """Handle signal from worker"""
        logger.info(f"Received signal from {signal.symbol}: {signal.direction} at {signal.entry}")
        
        # Update worker status
        if signal.symbol in self.workers:
            worker = self.workers[signal.symbol]
            worker.status.active_signals += 1
            worker.status.last_signal_time = signal.timestamp
        
        # Notify callbacks
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")
    
    def _on_status_update(self, status: PairStatus):
        """Handle status update from worker"""
        # Notify callbacks with all statuses
        self._notify_status_callbacks()
    
    def _notify_status_callbacks(self):
        """Notify status callbacks with all pair statuses"""
        statuses = {symbol: worker.status for symbol, worker in self.workers.items()}
        
        for callback in self.status_callbacks:
            try:
                callback(statuses)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def add_signal_callback(self, callback: Callable[[Signal], None]):
        """Add signal callback"""
        self.signal_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[Dict[str, PairStatus]], None]):
        """Add status callback"""
        self.status_callbacks.append(callback)
    
    def get_pair_status(self, symbol: str) -> Optional[PairStatus]:
        """Get status for a specific pair"""
        if symbol in self.workers:
            return self.workers[symbol].status
        return None
    
    def get_all_statuses(self) -> Dict[str, PairStatus]:
        """Get all pair statuses"""
        return {symbol: worker.status for symbol, worker in self.workers.items()}
