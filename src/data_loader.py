"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column is properly formatted as datetime"""
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        try:
            # Try unix milliseconds first
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        except Exception:
            # Fallback to general datetime parsing
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    return df.sort_values('timestamp').reset_index(drop=True)

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV file and validate required columns
    
    Args:
        path: Path to CSV file
        
    Returns:
        Cleaned and validated DataFrame
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If file doesn't exist
    """
    # Check if file exists
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    # Check required columns
    required_columns = {'timestamp', 'open', 'high', 'low', 'close'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(f'CSV must contain columns: {required_columns}. Missing: {missing_columns}')
    
    # Add volume column if missing
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    
    # Normalize timestamp to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    except Exception:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        except Exception as e:
            raise ValueError(f"Could not parse timestamp column: {e}")
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Sort by timestamp and reset index
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Validate OHLC data
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning")
    
    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if (df[col] <= 0).any():
            print(f"Warning: Found non-positive values in {col} column")
    
    # Basic OHLC validation
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    
    if invalid_ohlc.any():
        print(f"Warning: Found {invalid_ohlc.sum()} rows with invalid OHLC data")
        # Remove invalid rows
        df = df[~invalid_ohlc].reset_index(drop=True)
    
    return df

def prepare_data(ltf_path: str, htf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare both LTF and HTF data
    
    Args:
        ltf_path: Path to lower timeframe CSV
        htf_path: Path to higher timeframe CSV
        
    Returns:
        Tuple of (ltf_df, htf_df)
    """
    print(f"Loading LTF data from: {ltf_path}")
    df_ltf = load_csv(ltf_path)
    print(f"Loaded {len(df_ltf)} LTF candles")
    
    print(f"Loading HTF data from: {htf_path}")
    df_htf = load_csv(htf_path)
    print(f"Loaded {len(df_htf)} HTF candles")
    
    # Basic validation - ensure HTF timeframe is actually higher
    if len(df_htf) > len(df_ltf):
        print("Warning: HTF has more candles than LTF. Check your timeframes.")
    
    # Ensure both dataframes have data
    if df_ltf.empty or df_htf.empty:
        raise ValueError("One or both dataframes are empty")
    
    return df_ltf, df_htf

def validate_timeframe_alignment(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> bool:
    """
    Validate that timeframes are properly aligned
    
    Args:
        df_ltf: Lower timeframe dataframe
        df_htf: Higher timeframe dataframe
        
    Returns:
        True if alignment looks correct
    """
    if df_ltf.empty or df_htf.empty:
        return False
    
    # Check time ranges overlap
    ltf_start, ltf_end = df_ltf['timestamp'].iloc[0], df_ltf['timestamp'].iloc[-1]
    htf_start, htf_end = df_htf['timestamp'].iloc[0], df_htf['timestamp'].iloc[-1]
    
    # Check for reasonable overlap
    overlap_start = max(ltf_start, htf_start)
    overlap_end = min(ltf_end, htf_end)
    
    if overlap_start >= overlap_end:
        print("Warning: No time overlap between LTF and HTF data")
        return False
    
    return True
