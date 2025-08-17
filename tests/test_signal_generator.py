import sys
from pathlib import Path

# Ensure project root on sys.path before importing from src
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.models import OB
from src.signal_generator import generate_signals


def test_no_signal_without_mitigation(monkeypatch):
    # Create simple dataset with 5 candles
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="min"),
        "open": [100, 101, 102, 103, 100],
        "high": [101, 102, 103, 104, 102],
        "low": [99, 100, 101, 102, 99],
        "close": [100.5, 101.5, 102.5, 103.5, 100.5],
        "volume": [1] * 5,
    })
    df_htf = df.copy()

    # Patch SMC detection functions to return controlled values
    monkeypatch.setattr('src.signal_generator.fractal_pivots', lambda df, left, right: [])
    monkeypatch.setattr('src.signal_generator.detect_bos_choch', lambda df, swings: [(0, 'BOS_up')])
    monkeypatch.setattr('src.signal_generator.detect_fvg', lambda df: [])

    # Order block at last candle; there are no candles afterwards for mitigation
    ob = OB(kind='bull', start_idx=4, end_idx=4, open=100, high=102, low=99, close=100.5, bos_idx=4)
    monkeypatch.setattr('src.signal_generator.detect_ob', lambda df, bos_events: [ob])

    signals = generate_signals(df, df_htf, left=1, right=1)
    # No candles after the OB, so no valid mitigation should occur
    assert signals.empty
