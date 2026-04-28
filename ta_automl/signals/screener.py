"""Stage 1: filter TA-Lib indicators by data quality (remove degenerate outputs).

Individual TA indicators rarely produce statistically significant next-day return
predictions in isolation (efficient market hypothesis). This screener is a pure
data-quality pass — degenerate indicators removed, rest passed to Vizier for
hyperparameter search and combination optimization.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ta_automl.config import ScreenConfig
from ta_automl.signals.auto_discover import (
    compute_raw,
    default_params,
    get_all_indicator_names,
)
from ta_automl.signals.binarizer import binarize


def _quick_sharpe(signal: pd.Series, returns: pd.Series) -> float:
    """Annualised Sharpe of a long/short strategy driven by signal."""
    strat_ret = signal.shift(1) * returns
    strat_ret = strat_ret.dropna()
    if strat_ret.std() == 0 or len(strat_ret) < 20:
        return 0.0
    return float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))


def _significance_test(signal: pd.Series, returns: pd.Series) -> float:
    """Mann-Whitney U p-value: buy-day vs sell-day next returns."""
    from scipy import stats
    buy_ret  = returns[signal.shift(1) == 1].dropna()
    sell_ret = returns[signal.shift(1) == -1].dropna()
    if len(buy_ret) < 5 or len(sell_ret) < 5:
        return 1.0
    _, p = stats.mannwhitneyu(buy_ret, sell_ret, alternative="two-sided")
    return float(p)


def screen_indicators(
    df: pd.DataFrame,
    config: ScreenConfig,
    verbose: bool = True,
    return_tuned: bool = False,
):
    """
    Loop over all TA-Lib indicators (via talib.get_functions()), compute, binarize,
    and keep those passing data-quality (and optional significance) checks.

    Two modes:
      - default (config.tune_params=False): each indicator is evaluated with
        TA-Lib default parameters.
      - parameter-aware (config.tune_params=True): per-indicator hyperparameter
        search (Vizier / FLAML / random) finds a better config + binarization
        method before quality filtering.

    Returns:
      - return_tuned=False (default): list[str] of indicator keys
      - return_tuned=True: tuple (list[str], dict) where dict maps
        key -> {"params": ..., "binarize": ..., "score": ...}
    """
    try:
        all_names = get_all_indicator_names()
    except RuntimeError as e:
        warnings.warn(str(e))
        return []

    next_ret = df["Close"].pct_change().shift(-1)

    p_thresh = config.p_threshold
    if config.p_filter and config.bonferroni:
        p_thresh = config.p_threshold / len(all_names)

    survivors: list[str] = []
    tuned_map: dict[str, dict] = {}  # key -> {"params": ..., "binarize": ..., "score": ...}

    if verbose:
        try:
            from rich.progress import (
                BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
            )
            _rich = True
        except ImportError:
            _rich = False
    else:
        _rich = False

    def _process_one(name: str) -> list[str]:
        keys_kept = []

        # Parameter-aware path: find best (params, method) per output key first
        if config.tune_params:
            from ta_automl.signals.tuner import tune_one_indicator
            try:
                tuned = tune_one_indicator(
                    name, df, next_ret,
                    n_trials=config.tune_trials,
                    optimizer=config.tune_optimizer,
                    metric=config.tune_metric,
                    tune_method=config.tune_method_choice,
                )
            except Exception:
                tuned = {}
        else:
            tuned = {}

        # Compute the raw outputs we'll evaluate. For tuned mode, use the
        # winning config of each output key; for default mode, TA-Lib defaults.
        if tuned:
            raw_dict = {}
            # Group keys by their winning config to compute_raw once per config
            cfg_to_keys: dict[tuple, list[str]] = {}
            for key, info in tuned.items():
                p_tuple = tuple(sorted((info.get("params") or {}).items()))
                cfg_to_keys.setdefault(p_tuple, []).append(key)
            for p_tuple, keys in cfg_to_keys.items():
                p = dict(p_tuple)
                try:
                    rd = compute_raw(name, df, p or None)
                except Exception:
                    continue
                for key in keys:
                    if key in rd:
                        raw_dict[key] = rd[key]
        else:
            try:
                params = default_params(name)
                raw_dict = compute_raw(name, df, params)
            except Exception:
                return keys_kept

        for key, raw_series in raw_dict.items():
            # Quality check 1: too many NaNs
            nan_frac = float(raw_series.isna().mean())
            if nan_frac > config.max_nan_frac:
                continue

            # Quality check 2: binarize (using tuned method if available)
            method = tuned.get(key, {}).get("binarize") if tuned else None
            try:
                sig = binarize(key, raw_series, df, method=method)
            except Exception:
                continue

            non_zero_frac = float((sig != 0).mean())
            if non_zero_frac < config.min_signal_frac:
                continue

            # Optional Sharpe filter
            if config.min_sharpe > -10.0:
                sharpe = _quick_sharpe(sig, next_ret)
                if sharpe < config.min_sharpe:
                    continue

            # Optional p-value filter
            if config.p_filter:
                p_val = _significance_test(sig, next_ret)
                if p_val > p_thresh:
                    continue

            keys_kept.append(key)
            if tuned and key in tuned:
                tuned_map[key] = tuned[key]

        return keys_kept

    if _rich:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Screening[/bold cyan] {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("survivors: {task.fields[surv]}"),
        ) as prog:
            task = prog.add_task("indicators", total=len(all_names), surv=0)
            for name in all_names:
                prog.update(task, description=f"[dim]{name:<20}[/dim]")
                kept = _process_one(name)
                survivors.extend(kept)
                prog.update(task, advance=1, surv=len(survivors))
    else:
        for i, name in enumerate(all_names):
            if verbose and i % 20 == 0:
                print(f"  {i}/{len(all_names)} screened ... {len(survivors)} survivors so far")
            survivors.extend(_process_one(name))

    if verbose:
        mode = "tuned" if config.tune_params else "default-params"
        print(f"\nStage 1 complete ({mode}): {len(survivors)} survivors "
              f"from {len(all_names)} indicators")
        if config.tune_params and tuned_map:
            top = sorted(tuned_map.items(), key=lambda kv: kv[1]["score"], reverse=True)[:5]
            print("  top tuned indicators:")
            for k, v in top:
                print(f"    {k:<28} score={v['score']:.3f}  "
                      f"binarize={v['binarize']}  params={v['params']}")

    if return_tuned:
        return survivors, tuned_map
    return survivors
