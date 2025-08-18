"""
Data models for Smart Money Concepts trading bot
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Swing:
    """Represents a fractal pivot point (swing high/low)"""
    idx: int
    price: float
    kind: str  # 'H' or 'L'

@dataclass
class OB:
    """Order Block structure"""
    kind: str  # 'bull' or 'bear'
    start_idx: int
    end_idx: int
    open: float
    high: float
    low: float
    close: float
    bos_idx: int

@dataclass
class FVG:
    """Fair Value Gap structure"""
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    direction: str  # 'up' or 'down'

@dataclass
class Signal:
    """Trading signal structure"""
    timestamp: str  # ISO format string
    direction: str  # 'LONG' or 'SHORT'
    entry: float
    sl: float
    tp: float
    rr: float
    htf_bias: str
    ob_idx: int
    bos_idx: int
    fvg_confluence: bool
