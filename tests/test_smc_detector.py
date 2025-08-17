import sys
from pathlib import Path

# Ensure project root on sys.path before importing from src
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.smc_detector import detect_fvg


def test_detect_fvg_bullish():
    df = pd.DataFrame({
        "open": [10, 10, 13],
        "high": [11, 10.5, 14],
        "low": [9, 9.5, 12],
        "close": [10, 9.8, 13.5],
    })
    fvgs = detect_fvg(df)
    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.top == 12
    assert fvg.bottom == 11
    assert fvg.direction == "up"


def test_detect_fvg_bearish():
    df = pd.DataFrame({
        "open": [11, 10.5, 8],
        "high": [12, 10.8, 8.5],
        "low": [10, 9.5, 7],
        "close": [11, 10, 7.5],
    })
    fvgs = detect_fvg(df)
    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.top == 10
    assert fvg.bottom == 8.5
    assert fvg.direction == "down"
