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
) -> list[str]:
    """
    Loop over all TA-Lib indicators (via talib.get_functions()), compute with
    default params, binarize, and keep those passing data-quality checks.

    Optional p-value / Sharpe filtering is available via config.p_filter=True.
    Returns a list of indicator keys (e.g. 'RSI', 'MACD__macd', 'BBANDS__upperband').
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

            # Quality check 2: binarize and ensure enough non-zero days
            try:
                sig = binarize(key, raw_series, df)
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
        print(f"\nStage 1 complete: {len(survivors)} survivors from {len(all_names)} indicators")

    return survivors
