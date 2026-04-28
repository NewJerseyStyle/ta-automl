"""Trial evaluator: params → weighted signals → backtest → metrics."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ta_automl.backtest.strategy import run_backtest
from ta_automl.config import StudyConfig
from ta_automl.signals.auto_discover import compute_raw, param_search_space
from ta_automl.signals.binarizer import INT_TO_METHOD, binarize


def _indicator_base(key: str) -> str:
    """'MACD__macd' → 'MACD'"""
    return key.split("__")[0]


def evaluate_trial(
    params: dict[str, Any],
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    surviving_keys: list[str],
    config: StudyConfig,
) -> dict[str, float]:
    """
    Build weighted combination signal from trial params, run backtest on df_test.

    df       — full OHLCV (used for indicator computation to preserve warmup)
    df_test  — held-out portion only (backtest runs here)
    """
    threshold = float(params.get("combination_threshold", 0.3))
    min_trades = 5

    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for key in surviving_keys:
        weight = float(params.get(f"{key}__weight", 0.0))
        if weight < 0.05:
            continue

        base = _indicator_base(key)
        space = param_search_space(base)
        ind_params: dict[str, Any] = {}
        for p_name, (lo, hi, ptype) in space.items():
            full = f"{key}__{p_name}"
            if full in params:
                val = params[full]
                ind_params[p_name] = ptype(val)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_dict = compute_raw(base, df, ind_params or None)
        except Exception:
            continue

        raw_series = raw_dict.get(key)
        if raw_series is None:
            raw_series = next(iter(raw_dict.values()))

        method_idx = int(params.get(f"{key}__binarize", 0))
        method = INT_TO_METHOD.get(method_idx, "percentile")

        sig = binarize(key, raw_series, df, method=method)
        weighted_sum += weight * sig
        total_weight += weight

    if total_weight < 0.01:
        return {"sharpe_ratio": -999.0, "total_return": -999.0, "win_rate": 0.0, "num_trades": 0}

    # Normalize and apply threshold
    weighted_sum /= total_weight
    combined = pd.Series(0, index=df.index, dtype=int)
    combined[weighted_sum >  threshold] = 1
    combined[weighted_sum < -threshold] = -1

    # Slice to test window (no lookahead: all indicator computation used full df)
    combined_test = combined.reindex(df_test.index).fillna(0).astype(int)

    try:
        result = run_backtest(
            df_test,
            combined_test,
            cash=config.cash,
            commission=config.commission,
            allow_short=config.allow_short,
        )
    except Exception:
        return {"sharpe_ratio": -999.0, "total_return": -999.0, "win_rate": 0.0, "num_trades": 0}

    # Penalise degenerate never-trade solutions
    if result["num_trades"] < min_trades:
        result["sharpe_ratio"] *= 0.1

    return {k: v for k, v in result.items() if not k.startswith("_")}


def build_vizier_param_space(surviving_keys: list[str]) -> dict:
    """
    Return a description of the Vizier parameter space as a plain dict.
    Format: {param_name: ("float"|"int"|"categorical", lo, hi, [choices])}
    """
    from ta_automl.signals.binarizer import BINARIZE_METHODS

    space = {}
    space["combination_threshold"] = ("float", 0.05, 0.80, None)

    processed_bases: set[str] = set()
    for key in surviving_keys:
        base = _indicator_base(key)

        # Weight
        space[f"{key}__weight"] = ("float", 0.0, 1.0, None)

        # Binarization method (int index into BINARIZE_METHODS)
        space[f"{key}__binarize"] = ("int", 0, len(BINARIZE_METHODS) - 1, None)

        # Indicator-specific params (shared per base indicator, deduplicated)
        if base not in processed_bases:
            processed_bases.add(base)
            for p_name, (lo, hi, ptype) in param_search_space(base).items():
                full = f"{key}__{p_name}"
                kind = "int" if ptype is int else "float"
                space[full] = (kind, lo, hi, None)

    return space
