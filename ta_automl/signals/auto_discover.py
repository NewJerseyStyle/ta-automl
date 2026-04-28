"""Auto-discover all TA-Lib indicators and compute their raw outputs."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import talib
    import talib.abstract as ta_abstract
    _TALIB = True
except ImportError:
    _TALIB = False


def get_all_indicator_names() -> list[str]:
    """Return all TA-Lib indicator names, sorted."""
    if not _TALIB:
        raise RuntimeError("TA-Lib not available. Install TA-Lib C library first.")
    return sorted(talib.get_functions())


def get_indicator_info(name: str) -> dict[str, Any]:
    """Return metadata for a TA-Lib indicator: inputs, outputs, parameters."""
    fn = ta_abstract.Function(name)
    info = fn.info
    return {
        "name": name,
        "group": info.get("group", ""),
        "inputs": list(fn.input_names.values()),
        "output_names": fn.output_names,
        "parameters": dict(info.get("parameters", {})),
        "n_outputs": len(fn.output_names),
    }


def _make_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the input dict the Abstract API expects."""
    return {
        "open":   df["Open"].values.astype(float),
        "high":   df["High"].values.astype(float),
        "low":    df["Low"].values.astype(float),
        "close":  df["Close"].values.astype(float),
        "volume": df["Volume"].values.astype(float),
    }


def compute_raw(
    name: str,
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> dict[str, pd.Series]:
    """
    Compute a TA-Lib indicator with the given params.

    Returns a dict mapping output_name -> pd.Series aligned to df.index.
    Multi-output indicators (MACD, BBANDS, STOCH …) return multiple entries.
    """
    if not _TALIB:
        raise RuntimeError("TA-Lib not available")

    fn = ta_abstract.Function(name)
    inputs = _make_inputs(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if params:
            result = fn(inputs, **params)
        else:
            result = fn(inputs)

    output_names = fn.output_names
    if not isinstance(result, (tuple, list)):
        result = [result]

    out: dict[str, pd.Series] = {}
    for oname, arr in zip(output_names, result):
        key = f"{name}__{oname}" if len(output_names) > 1 else name
        out[key] = pd.Series(arr, index=df.index, dtype=float)

    return out


def default_params(name: str) -> dict[str, Any]:
    """Return TA-Lib's built-in default parameters for an indicator."""
    fn = ta_abstract.Function(name)
    return dict(fn.info.get("parameters", {}))


def param_search_space(name: str) -> dict[str, tuple[int | float, int | float, type]]:
    """
    Return a Vizier-friendly search space for an indicator's parameters.

    Format: {param_name: (lo, hi, python_type)}
    Integer params are searched as int; float params (e.g. nbdevup) as float.
    CDL* pattern indicators have no parameters — returns empty dict.
    """
    space: dict[str, tuple] = {}
    for p_name, p_default in default_params(name).items():
        if not isinstance(p_default, (int, float)):
            continue
        p_type = int if isinstance(p_default, int) else float
        if p_type is int:
            lo = max(2, int(p_default * 0.3))
            hi = max(lo + 1, int(p_default * 3))
            # Cap fast/slow period combos at reasonable limits
            if "fast" in p_name.lower():
                hi = min(hi, 50)
            elif "slow" in p_name.lower() or p_name.lower() in ("timeperiod", "period"):
                hi = min(hi, 200)
        else:
            lo = round(p_default * 0.3, 2)
            hi = round(p_default * 3.0, 2)
            if lo == hi:
                hi = lo + 0.1
        space[p_name] = (lo, hi, p_type)
    return space
