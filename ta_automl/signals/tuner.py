"""Per-indicator hyperparameter tuner used during Stage-1 screening.

Given an indicator name, run a small budget of trials (Vizier / FLAML / random)
to find the parameter combination + binarization method that maximizes a
quick screening metric (e.g. |Sharpe| of the signal applied to next-day returns).

Output is per indicator-key:
    {key: {"score": float, "params": {...}, "binarize": str}}
which the screener then uses instead of TA-Lib defaults.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ta_automl.signals.auto_discover import (
    compute_raw,
    default_params,
    param_search_space,
)
from ta_automl.signals.binarizer import BINARIZE_METHODS, binarize, infer_method


def _quick_score(signal: pd.Series, returns: pd.Series, metric: str) -> float:
    """Compute the per-indicator screening score (higher = better)."""
    sr = signal.shift(1) * returns
    sr = sr.dropna()
    if len(sr) < 20 or sr.std() == 0:
        return 0.0

    if metric == "abs_sharpe":
        return float(abs(sr.mean() / sr.std() * np.sqrt(252)))
    if metric == "sharpe":
        return float(sr.mean() / sr.std() * np.sqrt(252))
    if metric == "neg_p_value":
        from scipy import stats
        buy_ret  = returns[signal.shift(1) == 1].dropna()
        sell_ret = returns[signal.shift(1) == -1].dropna()
        if len(buy_ret) < 5 or len(sell_ret) < 5:
            return -1.0
        _, p = stats.mannwhitneyu(buy_ret, sell_ret, alternative="two-sided")
        return float(-p)
    raise ValueError(f"Unknown tune_metric: {metric}")


def _sample_random(space: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Randomly sample one config from a Vizier-style search space."""
    out: dict[str, Any] = {}
    for p_name, (lo, hi, ptype) in space.items():
        if ptype is int:
            out[p_name] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            out[p_name] = float(rng.uniform(float(lo), float(hi)))
    return out


def tune_one_indicator(
    name: str,
    df: pd.DataFrame,
    next_ret: pd.Series,
    *,
    n_trials: int = 8,
    optimizer: str = "vizier",
    metric: str = "abs_sharpe",
    tune_method: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, Any]]:
    """Tune one TA-Lib indicator. Returns per-output-key best config.

    Returns:
        {indicator_key: {"score": float, "params": dict, "binarize": str}}
    Empty dict if the indicator can't be evaluated.
    """
    rng = rng or np.random.default_rng(42)
    space = param_search_space(name)
    methods_to_try = list(BINARIZE_METHODS) if tune_method else [None]

    # Edge case: indicator has no parameters and we're not searching methods
    if not space and not tune_method:
        try:
            raw_dict = compute_raw(name, df, default_params(name))
        except Exception:
            return {}
        out = {}
        for key, raw in raw_dict.items():
            try:
                sig = binarize(key, raw, df)
            except Exception:
                continue
            score = _quick_score(sig, next_ret, metric)
            out[key] = {"score": score, "params": dict(default_params(name)), "binarize": infer_method(key)}
        return out

    # Always include defaults as a baseline trial
    baseline = dict(default_params(name))
    trial_configs: list[dict[str, Any]] = [baseline]

    if optimizer == "random" or not space:
        for _ in range(max(0, n_trials - 1)):
            trial_configs.append(_sample_random(space, rng) if space else dict(baseline))

    elif optimizer == "vizier":
        try:
            import logging as _logging
            _logging.getLogger("absl").setLevel(_logging.ERROR)
            try:
                from absl import logging as _absl_logging
                _absl_logging.set_verbosity(_absl_logging.ERROR)
            except Exception:
                pass
            from vizier.service import clients, pyvizier as vz

            sc = vz.StudyConfig(algorithm="GAUSSIAN_PROCESS_BANDIT")
            sc.metric_information.append(
                vz.MetricInformation(metric, goal=vz.ObjectiveMetricGoal.MAXIMIZE)
            )
            for p_name, (lo, hi, ptype) in space.items():
                if ptype is int:
                    sc.search_space.root.add_int_param(p_name, int(lo), int(hi))
                else:
                    sc.search_space.root.add_float_param(p_name, float(lo), float(hi))

            study = clients.Study.from_study_config(
                sc, owner="ta_automl_tuner", study_id=f"tune_{name}_{id(df)}",
            )
            collected: list[dict[str, Any]] = []

            def _eval(p: dict, m_method: str | None) -> float:
                # Evaluate one config; returns score over all output keys (mean)
                try:
                    raw_dict = compute_raw(name, df, p or None)
                except Exception:
                    return -1e9
                scores = []
                for key, raw in raw_dict.items():
                    try:
                        sig = binarize(key, raw, df, method=m_method)
                    except Exception:
                        continue
                    scores.append(_quick_score(sig, next_ret, metric))
                return float(max(scores)) if scores else -1e9

            for _ in range(n_trials - 1):
                try:
                    suggestions = study.suggest(count=1)
                except Exception:
                    break
                for trial in suggestions:
                    p = {pp.name: pp.value for pp in trial.parameters}
                    # Cast ints back to int (Vizier returns floats sometimes)
                    for p_name, (_lo, _hi, ptype) in space.items():
                        if p_name in p and ptype is int:
                            p[p_name] = int(p[p_name])
                    score = _eval(p, None)
                    try:
                        meas = vz.Measurement()
                        meas.metrics[metric] = vz.Metric(value=score)
                        trial.complete(meas)
                    except Exception:
                        pass
                    trial_configs.append(p)
                    collected.append(p)
        except Exception:
            # Vizier unavailable / failed → fall back to random
            for _ in range(max(0, n_trials - 1)):
                trial_configs.append(_sample_random(space, rng))

    elif optimizer == "flaml":
        try:
            from flaml import tune as flaml_tune

            flaml_space: dict[str, Any] = {}
            for p_name, (lo, hi, ptype) in space.items():
                if ptype is int:
                    flaml_space[p_name] = flaml_tune.randint(int(lo), int(hi) + 1)
                else:
                    flaml_space[p_name] = flaml_tune.uniform(float(lo), float(hi))

            sampled: list[dict[str, Any]] = []

            def obj(cfg):
                try:
                    raw_dict = compute_raw(name, df, cfg or None)
                except Exception:
                    flaml_tune.report(score=-1e9)
                    return
                best = -1e9
                for key, raw in raw_dict.items():
                    try:
                        sig = binarize(key, raw, df)
                    except Exception:
                        continue
                    s = _quick_score(sig, next_ret, metric)
                    best = max(best, s)
                flaml_tune.report(score=best)
                sampled.append(dict(cfg))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flaml_tune.run(
                    obj, config=flaml_space, num_samples=max(1, n_trials - 1),
                    metric="score", mode="max", verbose=0,
                )
            trial_configs.extend(sampled)
        except Exception:
            for _ in range(max(0, n_trials - 1)):
                trial_configs.append(_sample_random(space, rng))
    else:
        raise ValueError(f"Unknown tune_optimizer: {optimizer}")

    # Now evaluate every (config, method) pair and pick the best per output key
    best_per_key: dict[str, dict[str, Any]] = {}
    for cfg in trial_configs:
        try:
            raw_dict = compute_raw(name, df, cfg or None)
        except Exception:
            continue
        for key, raw in raw_dict.items():
            for m in methods_to_try:
                try:
                    sig = binarize(key, raw, df, method=m)
                except Exception:
                    continue
                score = _quick_score(sig, next_ret, metric)
                cur = best_per_key.get(key)
                if cur is None or score > cur["score"]:
                    best_per_key[key] = {
                        "score": float(score),
                        "params": dict(cfg),
                        "binarize": m if m is not None else infer_method(key),
                    }
    return best_per_key
