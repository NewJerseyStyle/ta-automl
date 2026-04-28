"""User-supplied indicator registry.

A user indicator is a callable:
    fn(df: pd.DataFrame, **params) -> pd.Series

It receives an OHLCV dataframe with columns Open, High, Low, Close, Volume and
must return a Series aligned to df.index. The series can be:
  • {-1, 0, +1}   → already a signal; used as-is
  • {0, 1}        → boolean state; mapped 0 → -1, 1 → +1
  • float / any   → passed through the standard binarizer (z-score / percentile)

Register with @register_indicator("my_name"). Once registered, the indicator
can be used by validate_idea(...) and is visible in the GUI Developer tab.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd


class IndicatorFn(Protocol):
    def __call__(self, df: pd.DataFrame, **params: Any) -> pd.Series: ...


# name -> {fn, doc, defaults}
INDICATOR_REGISTRY: dict[str, dict[str, Any]] = {}


def register_indicator(name: str) -> Callable[[IndicatorFn], IndicatorFn]:
    """Decorator: register a user indicator under `name`.

    The decorated function is validated for signature on import — it must take
    a df parameter and return a Series. Default values from the signature are
    captured as the indicator's default parameters.
    """
    def _wrap(fn: IndicatorFn) -> IndicatorFn:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if not params:
            raise TypeError(
                f"@register_indicator('{name}'): function must accept a DataFrame "
                f"as its first argument."
            )
        defaults = {
            p.name: p.default for p in params[1:]
            if p.default is not inspect.Parameter.empty
        }
        INDICATOR_REGISTRY[name] = {
            "fn": fn,
            "doc": (fn.__doc__ or "").strip(),
            "defaults": defaults,
        }
        return fn
    return _wrap


def get_indicator(name: str) -> IndicatorFn:
    if name not in INDICATOR_REGISTRY:
        raise KeyError(
            f"Unknown indicator {name!r}. Registered: {sorted(INDICATOR_REGISTRY)}"
        )
    return INDICATOR_REGISTRY[name]["fn"]


def list_indicators() -> list[str]:
    return sorted(INDICATOR_REGISTRY.keys())


def compute_user_signal(
    name: str,
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Run a registered indicator and coerce the output to {-1, 0, +1} ints.

    Coercion rules (in order):
      • already in {-1, 0, +1}  → cast to int
      • only {0, 1}             → 0 → -1, 1 → +1   (boolean state)
      • only {True, False}      → same as above
      • anything else (float)   → z-score percentile binarize
    """
    entry = INDICATOR_REGISTRY[name]
    merged = {**entry["defaults"], **(params or {})}
    out = entry["fn"](df, **merged)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index)
    out = out.reindex(df.index)

    # Booleans
    if out.dtype == bool:
        return out.map({True: 1, False: -1}).fillna(0).astype(int)

    # Inspect unique values among non-NaN
    non_na = out.dropna()
    if len(non_na) == 0:
        return pd.Series(0, index=df.index, dtype=int)

    uniq = set(np.unique(non_na.values))
    if uniq.issubset({-1, 0, 1}):
        return out.fillna(0).astype(int)
    if uniq.issubset({0, 1}):
        return out.map({0: -1, 1: 1}).fillna(0).astype(int)

    # Float fallback: percentile binarize at 30th/70th
    lo, hi = np.nanpercentile(out.values, [30, 70])
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[out > hi] = 1
    sig[out < lo] = -1
    return sig
