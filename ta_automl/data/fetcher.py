from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path | str = ".cache",
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance; cache as parquet for reuse.

    Returns DataFrame with columns Open, High, Low, Close, Volume
    (exact case required by backtesting.py).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_{start}_{end}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned for {symbol} [{start} – {end}]")

        # yfinance may return MultiIndex columns; flatten if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Ensure standard column names (capital first letter)
        rename = {}
        for col in raw.columns:
            cap = col.capitalize()
            if cap != col:
                rename[col] = cap
        if rename:
            raw = raw.rename(columns=rename)

        # Keep only the five columns backtesting.py expects
        needed = ["Open", "High", "Low", "Close", "Volume"]
        df = raw[[c for c in needed if c in raw.columns]].copy()

        df.to_parquet(cache_file)

    _validate(df, symbol)
    return df


def _validate(df: pd.DataFrame, symbol: str) -> None:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    if df.isnull().any().any():
        n_nan = df.isnull().sum().sum()
        # Forward-fill small gaps (weekends/holidays sometimes leak through)
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        import warnings
        warnings.warn(f"{symbol}: filled {n_nan} NaN values")

    if len(df) < 100:
        import warnings
        warnings.warn(f"{symbol}: only {len(df)} rows — results may be unreliable")
