"""FLAML BlendSearch optimizer — alternative to Vizier for high-dim spaces."""
from __future__ import annotations

import warnings
from typing import Any, Callable

from ta_automl.optimization.evaluator import build_vizier_param_space


def _to_flaml_space(param_space: dict) -> dict:
    """Convert our internal space format to FLAML tune search space."""
    try:
        from flaml import tune
    except ImportError:
        raise RuntimeError("FLAML not installed. Run: pip install flaml[blendsearch]")

    flaml_space = {}
    for p_name, (kind, lo, hi, _) in param_space.items():
        if kind == "float":
            flaml_space[p_name] = tune.uniform(float(lo), float(hi))
        elif kind == "int":
            flaml_space[p_name] = tune.randint(int(lo), int(hi) + 1)
    return flaml_space


def run_flaml_study(
    surviving_keys: list[str],
    eval_fn: Callable[[dict], dict],
    n_trials: int,
    metric: str = "objective",
    time_budget_s: int | None = None,
    verbose: bool = True,
    aggregator: str = "weighted_sum",
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Run FLAML BlendSearch to find best indicator params + combination weights.

    Returns (best_params, best_metrics).
    """
    try:
        from flaml import tune
    except ImportError:
        raise RuntimeError(
            "FLAML not installed. Run: pip install 'flaml[blendsearch]'"
        )

    param_space = build_vizier_param_space(surviving_keys, aggregator=aggregator)
    flaml_space = _to_flaml_space(param_space)

    best_metrics_ref: dict = {}

    def objective(config):
        metrics = eval_fn(config)
        val = float(metrics.get(metric, metrics.get("sharpe_ratio", -999.0)))
        tune.report(**{metric: val})
        best_metrics_ref.update(metrics)

    verbosity = 1 if verbose else 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = tune.run(
            objective,
            config=flaml_space,
            num_samples=n_trials,
            time_budget_s=time_budget_s,
            metric=metric,
            mode="max",
            verbose=verbosity,
        )

    best_params = analysis.best_config
    # Re-evaluate best config to get full metrics dict
    final_metrics = eval_fn(best_params)
    return best_params, final_metrics
