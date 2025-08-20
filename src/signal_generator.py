"""
Signal generation logic for SMC trading bot
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .models import Signal
from .smc_detector import (
    fractal_pivots, detect_bos_choch, detect_fvg, 
    detect_ob, premium_discount
)

def calculate_exit_conditions(df: pd.DataFrame, entry_idx: int, entry_price: float, 
                            sl: float, tp: float, direction: str) -> Dict:
    """
    Calculate exit conditions and timing for a trading signal
    
    Args:
        df: Price dataframe
        entry_idx: Index of entry candle
        entry_price: Entry price
        sl: Stop loss price
        tp: Take profit price
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Dictionary with exit information
    """
    # Look forward from entry to find exit
    post_entry_data = df.iloc[entry_idx + 1:]
    
    if post_entry_data.empty:
        return {
            'exit_time': df.iloc[entry_idx]['timestamp'],
            'exit_price': entry_price,
            'exit_reason': 'NO_EXIT',
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'duration_minutes': 0
        }
    
    exit_idx = None
    exit_price = entry_price
    exit_reason = 'NO_EXIT'
    
    for idx, candle in post_entry_data.iterrows():
        high = candle['high']
        low = candle['low']
        
        if direction == 'LONG':
            # Check for TP hit
            if high >= tp:
                exit_idx = idx
                exit_price = tp
                exit_reason = 'TP'
                break
            # Check for SL hit
            elif low <= sl:
                exit_idx = idx
                exit_price = sl
                exit_reason = 'SL'
                break
        else:  # SHORT
            # Check for TP hit
            if low <= tp:
                exit_idx = idx
                exit_price = tp
                exit_reason = 'TP'
                break
            # Check for SL hit
            elif high >= sl:
                exit_idx = idx
                exit_price = sl
                exit_reason = 'SL'
                break
    
    # Calculate P&L
    if direction == 'LONG':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    
    pnl_percent = (pnl / entry_price) * 100
    
    # Calculate duration
    if exit_idx is not None:
        entry_time = pd.to_datetime(df.iloc[entry_idx]['timestamp'])
        exit_time = pd.to_datetime(df.iloc[exit_idx]['timestamp'])
        duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
        exit_time_str = str(exit_time)
    else:
        # Use last available candle time
        last_time = pd.to_datetime(df.iloc[-1]['timestamp'])
        entry_time = pd.to_datetime(df.iloc[entry_idx]['timestamp'])
        duration_minutes = int((last_time - entry_time).total_seconds() / 60)
        exit_time_str = str(last_time)
    
    return {
        'exit_time': exit_time_str,
        'exit_price': float(exit_price),
        'exit_reason': exit_reason,
        'pnl': float(pnl),
        'pnl_percent': float(pnl_percent),
        'duration_minutes': duration_minutes
    }

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
        DataFrame with trading signals including exit conditions and P&L
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
        
        # Step 10: Calculate exit conditions and P&L
        exit_info = calculate_exit_conditions(
            df_ltf, actual_touch_idx, entry, sl, tp, direction
        )
        
        # Step 11: Create signal with extended information
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
            'fvg_confluence': fvg_confluence,
            # New fields
            'exit_time': exit_info['exit_time'],
            'exit_price': exit_info['exit_price'],
            'exit_reason': exit_info['exit_reason'],
            'pnl': exit_info['pnl'],
            'pnl_percent': exit_info['pnl_percent'],
            'duration_minutes': exit_info['duration_minutes']
        }
        
        signals.append(signal)
    
    # Convert to DataFrame and sort by timestamp (entry time)
    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        signals_df = signals_df.sort_values('timestamp', ascending=False)
        signals_df['timestamp'] = signals_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return signals_df
