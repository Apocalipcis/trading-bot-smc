"""
Live SMC (Smart Money Concepts) analysis engine for real-time trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import asyncio
import logging

# Suppress asyncio debug warnings
logging.getLogger('asyncio').setLevel(logging.WARNING)

from .smc_detector import (
    fractal_pivots, detect_bos_choch, detect_fvg, 
    detect_ob, premium_discount
)
from .models import Signal
from .futures_data import LiveDataManager

class LiveSMCEngine:
    """Real-time SMC analysis engine"""
    
    def __init__(self, symbol: str, config: Dict):
        self.symbol = symbol.upper()
        self.config = config
        self.data_manager = LiveDataManager(symbol, ['15m', '4h'])
        
        # SMC state
        self.ltf_data = pd.DataFrame()
        self.htf_data = pd.DataFrame()
        self.current_signals = []
        self.htf_bias = 'neutral'
        self.last_analysis = None
        
        # Callbacks
        self.signal_callbacks = []
        self.update_callbacks = []
        
        # Setup data callbacks
        self.data_manager.add_data_callback(
            symbol, '15m', self._on_ltf_update
        )
        self.data_manager.add_data_callback(
            symbol, '4h', self._on_htf_update
        )
        
    async def start(self):
        """Start live SMC analysis"""
        logging.info(f"Starting live SMC analysis for {self.symbol}")
        
        # Initialize with historical data
        await self._initialize_data()
        
        # Start data stream
        return await self.data_manager.start()
        
    async def stop(self):
        """Stop live analysis"""
        await self.data_manager.stop()
        logging.info("Live SMC analysis stopped")
        
    async def _initialize_data(self):
        """Initialize with historical data"""
        # Get historical data
        self.ltf_data = self.data_manager.get_dataframe('15m', 500)
        self.htf_data = self.data_manager.get_dataframe('4h', 200)
        
        if len(self.ltf_data) > 0 and len(self.htf_data) > 0:
            # Initial full analysis
            await self._full_analysis()
            logging.info("Initial SMC analysis completed")
        else:
            logging.warning("Insufficient historical data for analysis")
            
    async def _on_ltf_update(self, kline_data: Dict):
        """Handle LTF (15m) data update"""
        # Update LTF data
        self.ltf_data = self.data_manager.get_dataframe('15m', 500)
        
        # Quick analysis for new signals
        if len(self.ltf_data) >= 50:  # Minimum data required
            await self._check_new_signals()
            
        # Notify update callbacks
        for callback in self.update_callbacks:
            await callback('ltf', kline_data)
            
    async def _on_htf_update(self, kline_data: Dict):
        """Handle HTF (4h) data update"""
        # Update HTF data
        self.htf_data = self.data_manager.get_dataframe('4h', 200)
        
        # Full analysis on HTF update
        if len(self.htf_data) >= 20:  # Minimum data required
            await self._full_analysis()
            
        # Notify update callbacks
        for callback in self.update_callbacks:
            await callback('htf', kline_data)
            
    async def _full_analysis(self):
        """Perform full SMC analysis"""
        try:
            if len(self.ltf_data) < 50 or len(self.htf_data) < 20:
                return
                
            # Determine HTF bias
            self.htf_bias = self._get_htf_bias()
            
            # Generate signals
            new_signals = self._generate_signals()
            
            # Check for new signals
            for signal in new_signals:
                if not self._signal_exists(signal):
                    self.current_signals.append(signal)
                    
                    # Notify signal callbacks
                    for callback in self.signal_callbacks:
                        await callback(signal)
                        
            # Clean old signals (older than 4 hours)
            self._cleanup_old_signals()
            
            self.last_analysis = datetime.now()
            
        except Exception as e:
            logging.error(f"Error in full analysis: {e}")
            
    async def _check_new_signals(self):
        """Quick check for new signals on LTF update"""
        try:
            if self.htf_bias == 'neutral':
                return
                
            # Generate signals with current bias
            new_signals = self._generate_signals()
            
            # Check for truly new signals
            for signal in new_signals:
                if not self._signal_exists(signal):
                    self.current_signals.append(signal)
                    
                    # Notify signal callbacks
                    for callback in self.signal_callbacks:
                        await callback(signal)
                        
        except Exception as e:
            logging.error(f"Error checking new signals: {e}")
            
    def _get_htf_bias(self) -> str:
        """Determine HTF bias from 4h data"""
        try:
            swings_htf = fractal_pivots(
                self.htf_data, 
                self.config.get('fractal_left', 2),
                self.config.get('fractal_right', 2)
            )
            
            if len(swings_htf) < 2:
                return 'neutral'
                
            bos_htf = detect_bos_choch(self.htf_data, swings_htf)
            
            if bos_htf and 'up' in bos_htf[-1][1]:
                return 'bull'
            elif bos_htf and 'down' in bos_htf[-1][1]:
                return 'bear'
            else:
                return 'neutral'
                
        except Exception as e:
            logging.error(f"Error determining HTF bias: {e}")
            return 'neutral'
            
    def _generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        try:
            if self.htf_bias == 'neutral':
                return []
                
            # SMC detection
            swings = fractal_pivots(
                self.ltf_data,
                self.config.get('fractal_left', 2),
                self.config.get('fractal_right', 2)
            )
            
            if len(swings) < 3:
                return []
                
            bos_events = detect_bos_choch(self.ltf_data, swings)
            fvgs = detect_fvg(self.ltf_data)
            obs = detect_ob(self.ltf_data, bos_events)
            
            signals = []
            
            # Premium/Discount range
            lookback = min(len(self.ltf_data), 500)
            recent_data = self.ltf_data.iloc[-lookback:]
            range_low = recent_data['low'].min()
            range_high = recent_data['high'].max()
            
            # Process Order Blocks
            for ob in obs[-5:]:  # Only check recent OBs
                # HTF bias filter
                if self.htf_bias == 'bull' and ob.kind != 'bull':
                    continue
                if self.htf_bias == 'bear' and ob.kind != 'bear':
                    continue
                    
                # Check for recent mitigation
                signal = self._check_ob_mitigation(ob, fvgs, range_low, range_high)
                if signal:
                    signals.append(signal)
                    
            return signals
            
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return []
            
    def _check_ob_mitigation(self, ob, fvgs: List, range_low: float, range_high: float) -> Optional[Dict]:
        """Check if OB has been mitigated recently"""
        try:
            # Look for mitigation in last 20 candles
            recent_data = self.ltf_data.iloc[-20:]
            if recent_data.empty:
                return None
                
            ob_low, ob_high = ob.low, ob.high
            
            # Check for mitigation
            touched = ((recent_data['low'] <= ob_high) & 
                      (recent_data['high'] >= ob_low))
            touch_indices = np.where(touched)[0]
            
            if len(touch_indices) == 0:
                return None
                
            # Use latest mitigation
            relative_touch_idx = touch_indices[-1]
            actual_touch_idx = len(self.ltf_data) - 20 + relative_touch_idx
            
            # Check if it's very recent (last 3 candles)
            if relative_touch_idx < len(recent_data) - 3:
                return None
                
            # Get entry price
            entry_price = float(recent_data.iloc[relative_touch_idx]['close'])
            
            # FVG confluence check
            fvg_confluence = False
            for fvg in fvgs:
                if (ob.kind == 'bull' and fvg.direction == 'up' and
                    fvg.bottom <= ob_high and fvg.top >= ob_low):
                    fvg_confluence = True
                    break
                elif (ob.kind == 'bear' and fvg.direction == 'down' and
                      fvg.bottom <= ob_high and fvg.top >= ob_low):
                    fvg_confluence = True
                    break
                    
            # Premium/Discount filter
            pd_value = premium_discount(entry_price, range_low, range_high)
            if ob.kind == 'bull' and pd_value >= 0.5:
                return None
            if ob.kind == 'bear' and pd_value <= 0.5:
                return None
                
            # Calculate SL and TP
            rr_min = self.config.get('min_risk_reward', 3.0)
            
            if ob.kind == 'bull':
                sl = min(ob.low, recent_data.iloc[relative_touch_idx:]['low'].min())
                risk = entry_price - sl
                tp = entry_price + (risk * rr_min)
                direction = 'LONG'
            else:
                sl = max(ob.high, recent_data.iloc[relative_touch_idx:]['high'].max())
                risk = sl - entry_price
                tp = entry_price - (risk * rr_min)
                direction = 'SHORT'
                
            # Validate RR
            if direction == 'LONG':
                actual_rr = (tp - entry_price) / (entry_price - sl) if (entry_price - sl) > 0 else 0
            else:
                actual_rr = (entry_price - tp) / (sl - entry_price) if (sl - entry_price) > 0 else 0
                
            if actual_rr < rr_min:
                return None
                
            # Create signal
            signal = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'direction': direction,
                'entry': entry_price,
                'sl': float(sl),
                'tp': float(tp),
                'rr': float(actual_rr),
                'htf_bias': self.htf_bias,
                'fvg_confluence': fvg_confluence,
                'ob_idx': ob.start_idx,
                'confidence': self._calculate_confidence(fvg_confluence, actual_rr)
            }
            
            return signal
            
        except Exception as e:
            logging.error(f"Error checking OB mitigation: {e}")
            return None
            
    def _calculate_confidence(self, fvg_confluence: bool, rr: float) -> str:
        """Calculate signal confidence level"""
        score = 0
        
        if fvg_confluence:
            score += 1
        if rr >= 4.0:
            score += 1
        if rr >= 5.0:
            score += 1
            
        if score >= 2:
            return 'high'
        elif score == 1:
            return 'medium'
        else:
            return 'low'
            
    def _signal_exists(self, new_signal: Dict) -> bool:
        """Check if similar signal already exists"""
        for existing in self.current_signals:
            # Same direction and close entry price
            if (existing['direction'] == new_signal['direction'] and
                abs(existing['entry'] - new_signal['entry']) < new_signal['entry'] * 0.001):  # 0.1%
                return True
        return False
        
    def _cleanup_old_signals(self):
        """Remove signals older than 4 hours"""
        cutoff = datetime.now() - timedelta(hours=4)
        self.current_signals = [
            s for s in self.current_signals 
            if s['timestamp'] > cutoff
        ]
        
    def add_signal_callback(self, callback: Callable):
        """Add callback for new signals"""
        self.signal_callbacks.append(callback)
        
    def add_update_callback(self, callback: Callable):
        """Add callback for data updates"""
        self.update_callbacks.append(callback)
        
    def get_current_price(self) -> Optional[float]:
        """Get current price"""
        if not self.ltf_data.empty:
            return float(self.ltf_data.iloc[-1]['close'])
        return None
        
    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'symbol': self.symbol,
            'htf_bias': self.htf_bias,
            'current_price': self.get_current_price(),
            'active_signals': len(self.current_signals),
            'last_analysis': self.last_analysis,
            'ltf_candles': len(self.ltf_data),
            'htf_candles': len(self.htf_data),
            'websocket_connected': self.data_manager.stream.is_connected
        }
