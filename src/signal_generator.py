"""
Signal generation logic for SMC trading bot
"""
import pandas as pd
import numpy as np
from typing import List
from .models import Signal
from .smc_detector import (
    fractal_pivots, detect_bos_choch, detect_fvg, 
    detect_ob, premium_discount
)

def generate_signals(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, 
                    left: int = 2, right: int = 2, rr_min: float = 3.0) -> pd.DataFrame:
    """
    Generate trading signals based on SMC methodology
    
    Args:
        df_ltf: Lower timeframe dataframe
        df_htf: Higher timeframe dataframe  
        left: Fractal left bars
        right: Fractal right bars
        rr_min: Minimum risk/reward ratio
        
    Returns:
        DataFrame with trading signals
    """
    
    # Step 1: Determine HTF bias from last BOS
    swings_htf = fractal_pivots(df_htf, left, right)
    bos_htf = detect_bos_choch(df_htf, swings_htf)
    
    # Set bias based on last HTF BOS
    if bos_htf and 'up' in bos_htf[-1][1]:
        bias = 'bull'
    else:
        bias = 'bear'
    
    # Step 2: Detect LTF SMC elements
    swings = fractal_pivots(df_ltf, left, right)
    bos_events = detect_bos_choch(df_ltf, swings)
    fvgs = detect_fvg(df_ltf)
    obs = detect_ob(df_ltf, bos_events)
    
    signals = []
    
    # Step 3: Define recent range for Premium/Discount calculation
    lookback = min(len(df_ltf), 500)
    recent_data = df_ltf.iloc[-lookback:]
    range_low = recent_data['low'].min()
    range_high = recent_data['high'].max()
    
    # Step 4: Process each Order Block for potential signals
    for ob in obs:
        # Filter by HTF bias alignment
        if bias == 'bull' and ob.kind != 'bull':
            continue
        if bias == 'bear' and ob.kind != 'bear':
            continue
        
        # Step 5: Check for mitigation (price returning to OB)
        ob_low, ob_high = ob.low, ob.high
        
        # Find candles that touched the OB after it was formed
        # Start searching from the candle *after* the OB candle
        post_ob_data = df_ltf.iloc[ob.end_idx + 1:]
        if post_ob_data.empty:
            continue
            
        # Check for mitigation
        touched = ((post_ob_data['low'] <= ob_high) & 
                  (post_ob_data['high'] >= ob_low))
        touch_indices = np.where(touched)[0]
        
        if len(touch_indices) == 0:
            continue
            
        # Use last mitigation point
        relative_touch_idx = touch_indices[-1]
        # Adjust because post_ob_data starts from ob.end_idx + 1
        actual_touch_idx = ob.end_idx + 1 + relative_touch_idx
        
        # Step 6: Check for FVG confluence
        fvg_confluence = False
        for fvg in fvgs:
            # Check if FVG overlaps with mitigation zone and aligns with OB direction
            if (ob.kind == 'bull' and fvg.direction == 'up' and
                fvg.bottom <= ob_high and fvg.top >= ob_low):
                fvg_confluence = True
                break
            elif (ob.kind == 'bear' and fvg.direction == 'down' and
                  fvg.bottom <= ob_high and fvg.top >= ob_low):
                fvg_confluence = True
                break
        
        # Step 7: Premium/Discount filter
        entry_price = float(df_ltf['close'].iloc[actual_touch_idx])
        pd_value = premium_discount(entry_price, range_low, range_high)
        
        # For longs: enter in discount (< 0.5), for shorts: enter in premium (> 0.5)
        if ob.kind == 'bull' and pd_value >= 0.5:
            continue
        if ob.kind == 'bear' and pd_value <= 0.5:
            continue
        
        # Step 8: Calculate entry, SL, TP
        if ob.kind == 'bull':
            # Long signal
            entry = entry_price
            
            # SL below OB low or recent swing low
            post_touch_data = df_ltf.iloc[actual_touch_idx:actual_touch_idx+5]
            if not post_touch_data.empty:
                sl = min(ob.low, float(post_touch_data['low'].min()))
            else:
                sl = ob.low
                
            # TP based on RR ratio
            risk = entry - sl
            tp = entry + (risk * rr_min)
            direction = 'LONG'
            
        else:
            # Short signal  
            entry = entry_price
            
            # SL above OB high or recent swing high
            post_touch_data = df_ltf.iloc[actual_touch_idx:actual_touch_idx+5]
            if not post_touch_data.empty:
                sl = max(ob.high, float(post_touch_data['high'].max()))
            else:
                sl = ob.high
                
            # TP based on RR ratio
            risk = sl - entry
            tp = entry - (risk * rr_min)
            direction = 'SHORT'
        
        # Step 9: Validate RR ratio
        if direction == 'LONG':
            actual_rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0
        else:
            actual_rr = (entry - tp) / (sl - entry) if (sl - entry) > 0 else 0
            
        if actual_rr < rr_min:
            continue
        
        # Step 10: Create signal
        signal = {
            'timestamp': df_ltf['timestamp'].iloc[actual_touch_idx],
            'direction': direction,
            'entry': entry,
            'sl': float(sl),
            'tp': float(tp),
            'rr': float(actual_rr),
            'htf_bias': bias,
            'ob_idx': ob.start_idx,
            'bos_idx': ob.bos_idx,
            'fvg_confluence': fvg_confluence
        }
        
        signals.append(signal)
    
    return pd.DataFrame(signals)
