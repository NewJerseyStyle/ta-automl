"""Convert raw TA-Lib output (float series) to {-1, 0, +1} signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Indicators whose primary output should be compared to price (moving-average family)
_PRICE_CROSS_INDICATORS = {
    "DEMA", "EMA", "HT_TRENDLINE", "KAMA", "MA", "MAMA", "MAVP",
    "MIDPOINT", "SAR", "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA",
}

# Indicators where crossing zero is the signal (MACD histogram, etc.)
_TREND_INDICATORS = {
    "APO", "AROONOSC", "BOP", "CMO", "DX", "MACD__macdhist",
    "MACDEXT__macdhist", "MACDFIX__macdhist", "MFI", "MINUS_DI",
    "PLUS_DI", "PPO", "TRIX", "ULTOSC",
}

# Pattern recognition: TA-Lib returns 100 / -100 / 0
_PATTERN_PREFIX = "CDL"


def infer_method(key: str) -> str:
    """Auto-select a binarization method based on the indicator key."""
    base = key.split("__")[0]
    if base.startswith(_PATTERN_PREFIX):
        return "pattern"
    if base in _PRICE_CROSS_INDICATORS:
        return "price_cross"
    if key in _TREND_INDICATORS or base in _TREND_INDICATORS:
        return "trend"
    # Everything else (oscillators, momentum, volatility …) → percentile
    return "percentile"


BINARIZE_METHODS = ("percentile", "zscore", "trend", "price_cross", "pattern")
METHOD_TO_INT = {m: i for i, m in enumerate(BINARIZE_METHODS)}
INT_TO_METHOD = {i: m for i, m in enumerate(BINARIZE_METHODS)}


def binarize(
    key: str,
    raw: pd.Series,
    df: pd.DataFrame,
    method: str | None = None,
) -> pd.Series:
    """
    Convert a raw float Series to {-1, 0, +1}.

    key    — indicator key (used to infer method when method is None)
    raw    — raw output from compute_raw()
    df     — full OHLCV DataFrame (needed for price_cross)
    method — override the auto-detected method
    """
    if method is None:
        method = infer_method(key)

    raw = pd.Series(raw, index=df.index, dtype=float)
    close = df["Close"].astype(float)
    signal = pd.Series(0, index=df.index, dtype=int)

    if method == "pattern":
        # TA-Lib: 100 = bullish, -100 = bearish, 0 = no signal
        signal = (raw / 100).fillna(0).round().astype(int)

    elif method == "price_cross":
        signal[close > raw] = 1
        signal[close < raw] = -1

    elif method == "trend":
        # Positive value → bullish, negative value → bearish
        signal[raw > 0] = 1
        signal[raw < 0] = -1

    elif method == "percentile":
        # Compute rolling 252-day percentile rank (avoids lookahead)
        def _rolling_pct(s: pd.Series) -> pd.Series:
            return s.rolling(252, min_periods=20).apply(
                lambda x: float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1),
                raw=True,
            )
        pct = _rolling_pct(raw)
        signal[pct <= 0.20] = 1    # oversold / low → buy
        signal[pct >= 0.80] = -1   # overbought / high → sell

    elif method == "zscore":
        mu = raw.rolling(252, min_periods=20).mean()
        sigma = raw.rolling(252, min_periods=20).std().replace(0, np.nan)
        z = (raw - mu) / sigma
        signal[z <= -1.0] = 1
        signal[z >= 1.0] = -1

    else:
        raise ValueError(f"Unknown binarize method: {method!r}")

    return signal.fillna(0).astype(int)
