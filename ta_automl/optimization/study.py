"""Google Vizier in-process optimization study."""
from __future__ import annotations

import warnings
from typing import Any, Callable

from ta_automl.optimization.evaluator import build_vizier_param_space


def run_vizier_study(
    surviving_keys: list[str],
    eval_fn: Callable[[dict], dict],
    n_trials: int,
    study_name: str = "ta_automl",
    metric: str = "objective",
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Run a Vizier in-process GP-Bandit study.

    Returns (best_params, best_metrics).
    Falls back to random search if Vizier's GP algorithm (requires JAX) is unavailable.
    """
    try:
        import logging as _logging
        _logging.getLogger("absl").setLevel(_logging.ERROR)
        try:
            from absl import logging as _absl_logging
            _absl_logging.set_verbosity(_absl_logging.ERROR)
        except Exception:
            pass
        from vizier.service import clients
        from vizier.service import pyvizier as vz
        _vizier_ok = True
    except ImportError:
        warnings.warn("google-vizier not installed; falling back to FLAML.")
        _vizier_ok = False

    if not _vizier_ok:
        from ta_automl.optimization.flaml_search import run_flaml_study
        return run_flaml_study(surviving_keys, eval_fn, n_trials, metric=metric, verbose=verbose)

    # In-process mode — no server required
    try:
        clients.environment_variables.server_endpoint = clients.environment_variables.NO_ENDPOINT
    except AttributeError:
        pass  # older vizier versions expose it differently

    # Build StudyConfig
    sc = vz.StudyConfig(algorithm="GAUSSIAN_PROCESS_BANDIT")
    sc.metric_information.append(
        vz.MetricInformation(metric, goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )

    param_space = build_vizier_param_space(surviving_keys)
    root = sc.search_space.root

    for p_name, (kind, lo, hi, _) in param_space.items():
        if kind == "float":
            root.add_float_param(p_name, float(lo), float(hi))
        elif kind == "int":
            root.add_int_param(p_name, int(lo), int(hi))

    try:
        study = clients.Study.from_study_config(
            sc,
            owner="ta_automl_owner",
            study_id=study_name.replace(" ", "_"),
        )
    except Exception as e:
        warnings.warn(f"Vizier study creation failed ({e}); falling back to FLAML.")
        from ta_automl.optimization.flaml_search import run_flaml_study
        return run_flaml_study(surviving_keys, eval_fn, n_trials, metric=metric, verbose=verbose)

    best_metric = float("-inf")
    best_params: dict = {}
    best_metrics: dict = {}

    if verbose:
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
            _rich = True
        except ImportError:
            _rich = False
    else:
        _rich = False

    def _run_trials(prog=None, task=None):
        nonlocal best_metric, best_params, best_metrics
        for i in range(n_trials):
            try:
                suggestions = study.suggest(count=1)
            except Exception as e:
                warnings.warn(f"Vizier suggest() failed at trial {i}: {e}")
                break

            for trial in suggestions:
                raw_params = {p.name: p.value for p in trial.parameters}
                metrics = eval_fn(raw_params)
                val = float(metrics.get(metric, metrics.get("sharpe_ratio", -999.0)))

                try:
                    measurement = vz.Measurement()
                    measurement.metrics[metric] = vz.Metric(value=val)
                    trial.complete(measurement)
                except Exception:
                    pass

                if val > best_metric:
                    best_metric = val
                    best_params = raw_params
                    best_metrics = metrics

            if prog and task is not None:
                prog.update(
                    task,
                    advance=1,
                    description=f"[cyan]best {metric}[/cyan] = [bold green]{best_metric:.4f}[/bold green]",
                )
            elif verbose:
                print(f"  trial {i+1}/{n_trials}  {metric}={val:.4f}  best={best_metric:.4f}")

    if _rich:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as prog:
            task = prog.add_task(f"Vizier ({n_trials} trials)", total=n_trials)
            _run_trials(prog, task)
    else:
        _run_trials()

    return best_params, best_metrics
