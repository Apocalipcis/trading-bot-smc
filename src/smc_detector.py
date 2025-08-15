"""
Smart Money Concepts detection functions
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from .models import Swing, OB, FVG

def fractal_pivots(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[Swing]:
    """Detect fractal pivot points (swing highs/lows)"""
    swings: List[Swing] = []
    
    for i in range(left, len(df) - right):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        
        # Check for swing high
        if high == df['high'].iloc[i-left:i+right+1].max():
            swings.append(Swing(i, float(high), 'H'))
            
        # Check for swing low    
        if low == df['low'].iloc[i-left:i+right+1].min():
            swings.append(Swing(i, float(low), 'L'))
    
    # Deduplicate consecutive same-type swings, keep earliest
    filtered = []
    last = None
    for s in swings:
        if last and s.idx - last.idx <= 1 and s.kind == last.kind:
            continue
        filtered.append(s)
        last = s
        
    return filtered

def structure_from_swings(swings: List[Swing]) -> List[Tuple[int, str]]:
    """Label each swing as HH/HL/LH/LL structure"""
    out = []
    last_H = None
    last_L = None
    
    for s in swings:
        if s.kind == 'H':
            if last_H is None:
                label = 'H'
            else:
                label = 'HH' if s.price > last_H.price else 'LH'
            last_H = s
        else:  # s.kind == 'L'
            if last_L is None:
                label = 'L'
            else:
                label = 'HL' if s.price > last_L.price else 'LL'
            last_L = s
            
        out.append((s.idx, label))
        
    return out

def detect_bos_choch(df: pd.DataFrame, swings: List[Swing]) -> List[Tuple[int, str]]:
    """Detect Break of Structure (BOS) and Change of Character (CHOCH)"""
    events = []
    trend = None  # 'bull' or 'bear'
    
    for i in range(1, len(swings)):
        s = swings[i]
        
        if s.kind == 'H':
            # Look for previous high to break
            prev_highs = [x for x in swings[:i] if x.kind == 'H']
            if prev_highs:
                prev_H = prev_highs[-1]
                if df['close'].iloc[s.idx] > prev_H.price:
                    label = 'BOS_up' if trend in (None, 'bull') else 'CHOCH_up'
                    trend = 'bull'
                    events.append((s.idx, label))
                    
        else:  # s.kind == 'L'
            # Look for previous low to break
            prev_lows = [x for x in swings[:i] if x.kind == 'L']
            if prev_lows:
                prev_L = prev_lows[-1]
                if df['close'].iloc[s.idx] < prev_L.price:
                    label = 'BOS_down' if trend in (None, 'bear') else 'CHOCH_down'
                    trend = 'bear'
                    events.append((s.idx, label))
                    
    return events

def detect_fvg(df: pd.DataFrame) -> List[FVG]:
    """Detect Fair Value Gaps"""
    fvgs: List[FVG] = []
    
    for i in range(2, len(df)):
        # Bullish FVG: when low[i] > high[i-2] (gap up)
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            fvgs.append(FVG(
                start_idx=i-2,
                end_idx=i,
                top=float(df['low'].iloc[i]),
                bottom=float(df['high'].iloc[i-2]),
                direction='up'
            ))
            
        # Bearish FVG: when high[i] < low[i-2] (gap down)
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            fvgs.append(FVG(
                start_idx=i-2,
                end_idx=i,
                top=float(df['low'].iloc[i-2]),
                bottom=float(df['high'].iloc[i]),
                direction='down'
            ))
            
    return fvgs

def detect_ob(df: pd.DataFrame, bos_events: List[Tuple[int, str]]) -> List[OB]:
    """Detect Order Blocks based on BOS events"""
    obs: List[OB] = []
    
    for idx, kind in bos_events:
        if 'up' in kind:
            # Bullish OB: last bearish candle before BOS
            lookback_start = max(0, idx - 10)
            prior_candles = df.iloc[lookback_start:idx]
            bearish_candles = prior_candles[prior_candles['close'] < prior_candles['open']]
            
            if not bearish_candles.empty:
                ob_candle = bearish_candles.iloc[-1]
                obs.append(OB(
                    kind='bull',
                    start_idx=int(ob_candle.name),
                    end_idx=int(ob_candle.name),
                    open=float(ob_candle['open']),
                    high=float(ob_candle['high']),
                    low=float(ob_candle['low']),
                    close=float(ob_candle['close']),
                    bos_idx=idx
                ))
                
        else:  # 'down' in kind
            # Bearish OB: last bullish candle before BOS
            lookback_start = max(0, idx - 10)
            prior_candles = df.iloc[lookback_start:idx]
            bullish_candles = prior_candles[prior_candles['close'] > prior_candles['open']]
            
            if not bullish_candles.empty:
                ob_candle = bullish_candles.iloc[-1]
                obs.append(OB(
                    kind='bear',
                    start_idx=int(ob_candle.name),
                    end_idx=int(ob_candle.name),
                    open=float(ob_candle['open']),
                    high=float(ob_candle['high']),
                    low=float(ob_candle['low']),
                    close=float(ob_candle['close']),
                    bos_idx=idx
                ))
                
    return obs

def premium_discount(price: float, range_low: float, range_high: float) -> float:
    """Calculate premium/discount level (0-1 scale, 0.5 = equilibrium)"""
    if range_high == range_low:
        return 0.5
    return (price - range_low) / (range_high - range_low)
