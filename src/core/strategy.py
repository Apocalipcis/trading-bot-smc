"""
Strategy interface and SMC strategy implementation
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from config.models import Candle, Signal, PairConfig
from .exchange_gateway import ExchangeGateway

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, gateway: ExchangeGateway, config: PairConfig):
        self.gateway = gateway
        self.config = config
        self.symbol = config.symbol
    
    @abstractmethod
    async def generate_signals(self, ltf_candles: List[Candle], htf_candles: List[Candle]) -> List[Signal]:
        """Generate trading signals from candlestick data"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass


class SMCStrategy(Strategy):
    """Smart Money Concepts strategy implementation"""
    
    def __init__(self, gateway: ExchangeGateway, config: PairConfig):
        super().__init__(gateway, config)
        
        # Import SMC detection functions
        try:
            from ..smc_detector import (
                fractal_pivots, detect_bos_choch, detect_fvg, 
                detect_ob, premium_discount
            )
            self.fractal_pivots = fractal_pivots
            self.detect_bos_choch = detect_bos_choch
            self.detect_fvg = detect_fvg
            self.detect_ob = detect_ob
            self.premium_discount = premium_discount
        except ImportError:
            logger.error("Failed to import SMC detection functions")
            raise
        
        # Strategy state
        self.htf_bias = 'neutral'
        self.last_analysis_time: Optional[datetime] = None
    
    async def generate_signals(self, ltf_candles: List[Candle], htf_candles: List[Candle]) -> List[Signal]:
        """Generate SMC trading signals"""
        try:
            if len(ltf_candles) < 50 or len(htf_candles) < 20:
                logger.debug(f"Insufficient data for {self.symbol}: LTF={len(ltf_candles)}, HTF={len(htf_candles)}")
                return []
            
            # Convert candles to DataFrame format for SMC functions
            ltf_df = self._candles_to_dataframe(ltf_candles)
            htf_df = self._candles_to_dataframe(htf_candles)
            
            # Determine HTF bias
            self.htf_bias = self._get_htf_bias(htf_df)
            
            if self.htf_bias == 'neutral':
                logger.debug(f"Neutral HTF bias for {self.symbol}, skipping signal generation")
                return []
            
            # Generate signals
            signals = self._generate_smc_signals(ltf_df, htf_df)
            
            self.last_analysis_time = datetime.now()
            
            if signals:
                logger.info(f"Generated {len(signals)} signals for {self.symbol}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {self.symbol}: {e}")
            return []
    
    def _candles_to_dataframe(self, candles: List[Candle]):
        """Convert candle list to DataFrame for SMC functions"""
        import pandas as pd
        
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Ensure all numeric columns are float64
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _get_htf_bias(self, htf_df) -> str:
        """Determine HTF bias from higher timeframe data"""
        try:
            swings_htf = self.fractal_pivots(
                htf_df, 
                self.config.fractal_left,
                self.config.fractal_right
            )
            
            if len(swings_htf) < 2:
                return 'neutral'
            
            bos_htf = self.detect_bos_choch(htf_df, swings_htf)
            
            if bos_htf and 'up' in bos_htf[-1][1]:
                return 'bull'
            elif bos_htf and 'down' in bos_htf[-1][1]:
                return 'bear'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining HTF bias: {e}")
            return 'neutral'
    
    def _generate_smc_signals(self, ltf_df, htf_df) -> List[Signal]:
        """Generate SMC signals from LTF and HTF data"""
        try:
            # SMC detection
            swings = self.fractal_pivots(
                ltf_df,
                self.config.fractal_left,
                self.config.fractal_right
            )
            
            if len(swings) < 3:
                return []
            
            bos_events = self.detect_bos_choch(ltf_df, swings)
            fvgs = self.detect_fvg(ltf_df)
            obs = self.detect_ob(ltf_df, bos_events)
            
            signals = []
            
            # Premium/Discount range
            lookback = min(len(ltf_df), 500)
            recent_data = ltf_df.iloc[-lookback:]
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
                signal = self._check_ob_mitigation(ob, fvgs, range_low, range_high, ltf_df)
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in SMC signal generation: {e}")
            return []
    
    def _check_ob_mitigation(self, ob, fvgs: List, range_low: float, range_high: float, ltf_df) -> Optional[Signal]:
        """Check if Order Block has been mitigated recently"""
        try:
            # Look for mitigation in last 20 candles
            recent_data = ltf_df.iloc[-20:]
            if recent_data.empty:
                return None
            
            ob_low, ob_high = ob.low, ob.high
            
            # Check for mitigation
            import numpy as np
            touched = ((recent_data['low'] <= ob_high) & 
                      (recent_data['high'] >= ob_low))
            touch_indices = np.where(touched)[0]
            
            if len(touch_indices) == 0:
                return None
            
            # Use latest mitigation
            relative_touch_idx = touch_indices[-1]
            
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
            pd_value = self.premium_discount(entry_price, range_low, range_high)
            if ob.kind == 'bull' and pd_value >= 0.5:
                return None
            if ob.kind == 'bear' and pd_value <= 0.5:
                return None
            
            # Calculate SL and TP
            rr_min = self.config.min_risk_reward
            
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
            
            # Calculate confidence
            confidence = self._calculate_confidence(fvg_confluence, actual_rr)
            
            # Create signal
            signal = Signal(
                timestamp=datetime.now(),
                symbol=self.symbol,
                direction=direction,
                entry=float(entry_price),
                stop_loss=float(sl) if hasattr(sl, 'dtype') else sl,
                take_profit=float(tp) if hasattr(tp, 'dtype') else tp,
                risk_reward=float(actual_rr) if hasattr(actual_rr, 'dtype') else actual_rr,
                htf_bias=self.htf_bias,
                fvg_confluence=fvg_confluence,
                confidence=confidence,
                strategy=self.get_strategy_name()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error checking OB mitigation: {e}")
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
    
    def get_strategy_name(self) -> str:
        return "smc_v1"
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        return {
            'strategy': self.get_strategy_name(),
            'symbol': self.symbol,
            'htf_bias': self.htf_bias,
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'config': {
                'min_risk_reward': self.config.min_risk_reward,
                'fractal_left': self.config.fractal_left,
                'fractal_right': self.config.fractal_right
            }
        }


class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create_strategy(strategy_name: str, gateway: ExchangeGateway, config: PairConfig) -> Strategy:
        """Create strategy instance by name"""
        if strategy_name == "smc_v1":
            return SMCStrategy(gateway, config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy names"""
        return ["smc_v1"]
