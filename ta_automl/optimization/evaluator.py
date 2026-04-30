"""Trial evaluator: params → weighted signals → backtest → metrics."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ta_automl.backtest.strategy import run_backtest
from ta_automl.config import StudyConfig
from ta_automl.optimization.loss import LossContext, LossFn, get_loss
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
    loss_fn: str | LossFn | None = None,
    loss_extra: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Build weighted combination signal from trial params, run backtest on df_test.

    df       — full OHLCV (used for indicator computation to preserve warmup)
    df_test  — held-out portion only (backtest runs here)
    """
    threshold = float(params.get("combination_threshold", 0.3))
    aggregator = getattr(config, "aggregator", "weighted_sum")
    buy_floor = float(params.get("buy_conviction", 0.20))
    sell_floor = float(params.get("sell_conviction", 0.20))
    min_trades = 5

    weighted_sum = pd.Series(0.0, index=df.index)
    pos_weight = pd.Series(0.0, index=df.index)   # Σ wᵢ where sᵢ = +1
    neg_weight = pd.Series(0.0, index=df.index)   # Σ wᵢ where sᵢ = -1
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
        pos_weight += weight * (sig > 0).astype(float)
        neg_weight += weight * (sig < 0).astype(float)
        total_weight += weight

    # Resolve loss function up front — used to populate the 'objective' key
    fn = get_loss(loss_fn) if loss_fn is not None else get_loss(config.loss)
    ctx = LossContext(params=params, min_trades=5, extra=loss_extra or {})

    if total_weight < 0.01:
        bad = {"sharpe_ratio": -999.0, "total_return": -999.0, "win_rate": 0.0,
               "num_trades": 0, "max_drawdown": -100.0}
        bad["objective"] = float(fn(bad, ctx))
        return bad

    # Normalize and apply threshold
    weighted_sum /= total_weight
    pos_frac = pos_weight / total_weight
    neg_frac = neg_weight / total_weight
    combined = pd.Series(0, index=df.index, dtype=int)

    if aggregator == "clamped_sum":
        # Conviction-floor rule: BUY only when long-side weight share clears the
        # buy_floor AND beats short-side share by `threshold`. Symmetric for SELL.
        # Stepper surface than weighted_sum; encodes "BUY ⇒ real buy agreement".
        net = pos_frac - neg_frac
        combined[(pos_frac >= buy_floor) & (net > threshold)] = 1
        combined[(neg_frac >= sell_floor) & (net < -threshold)] = -1
    else:
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
        bad = {"sharpe_ratio": -999.0, "total_return": -999.0, "win_rate": 0.0,
               "num_trades": 0, "max_drawdown": -100.0}
        bad["objective"] = float(fn(bad, ctx))
        return bad

    # Penalise degenerate never-trade solutions
    if result["num_trades"] < min_trades:
        result["sharpe_ratio"] *= 0.1
        result["total_return"] = result["total_return"] * 0.1 - 1.0  # penalty for dd-loss

    metrics = {k: v for k, v in result.items() if not k.startswith("_")}
    metrics["objective"] = float(fn(metrics, ctx))
    return metrics


def build_vizier_param_space(surviving_keys: list[str], aggregator: str = "weighted_sum") -> dict:
    """
    Return a description of the Vizier parameter space as a plain dict.
    Format: {param_name: ("float"|"int"|"categorical", lo, hi, [choices])}
    """
    from ta_automl.signals.binarizer import BINARIZE_METHODS

    space = {}
    space["combination_threshold"] = ("float", 0.05, 0.80, None)
    if aggregator == "clamped_sum":
        space["buy_conviction"] = ("float", 0.05, 0.70, None)
        space["sell_conviction"] = ("float", 0.05, 0.70, None)

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
