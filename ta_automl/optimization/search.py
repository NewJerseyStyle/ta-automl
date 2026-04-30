"""Pluggable search strategies — alternatives to the default weighted-combination search.

A search strategy is a callable that accepts a `SearchContext` and returns a
`SearchResult`. The framework's CLI/API selects a strategy by name (or by passing
a callable directly).

Built-in strategies:
- "weighted"   — the default: Vizier/FLAML over indicator weights + thresholds
                 (delegates to optimization.study / flaml_search)
- "shap"       — FLAML tunes a CatBoost classifier over ALL indicator features,
                 then SHAP attributions are used to identify which indicators
                 matter as a combined signal. Great for capturing event-driven
                 signals that don't show up in revenue-search alone.

Users can register their own with @register_search and select via
`--search-strategy <name>` or `--search-strategy module:fn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from ta_automl.config import StudyConfig


@dataclass
class SearchContext:
    """Everything a search strategy needs."""
    df: pd.DataFrame                          # full OHLCV
    df_test: pd.DataFrame                     # held-out test slice
    survivors: list[str]                      # indicator keys from screener
    config: StudyConfig
    loss_fn: Any | None = None                # name or callable for loss
    loss_extra: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # arbitrary user state
    tuned: dict[str, dict[str, Any]] = field(default_factory=dict)
    # tuned: key -> {"params": ..., "binarize": ..., "score": ...} from Stage-1 tuner


@dataclass
class SearchResult:
    """Uniform output across search strategies."""
    best_params: dict[str, Any]               # parameters of the winning trial
    best_metrics: dict[str, float]            # metric dict (sharpe, return, dd, objective ...)
    signals_df: pd.DataFrame                  # one column per indicator key, {-1,0,+1}
    combined: pd.Series                       # final combined signal series (int)
    importance: dict[str, float] = field(default_factory=dict)  # per-key importance / weight
    extra: dict[str, Any] = field(default_factory=dict)         # strategy-specific output


class SearchFn(Protocol):
    def __call__(self, ctx: SearchContext) -> SearchResult: ...


# ── Registry ─────────────────────────────────────────────────────────────────
SEARCH_REGISTRY: dict[str, SearchFn] = {}


def register_search(name: str) -> Callable[[SearchFn], SearchFn]:
    """Decorator to register a search strategy under `name`."""
    def _wrap(fn: SearchFn) -> SearchFn:
        SEARCH_REGISTRY[name] = fn
        return fn
    return _wrap


def get_search(name_or_fn) -> SearchFn:
    """Look up by name, or return the callable directly. Supports 'module:fn'."""
    if callable(name_or_fn):
        return name_or_fn
    if isinstance(name_or_fn, str) and ":" in name_or_fn:
        import importlib
        mod_name, fn_name = name_or_fn.rsplit(":", 1)
        return getattr(importlib.import_module(mod_name), fn_name)
    if name_or_fn not in SEARCH_REGISTRY:
        raise KeyError(
            f"Unknown search strategy {name_or_fn!r}. "
            f"Registered: {sorted(SEARCH_REGISTRY)}"
        )
    return SEARCH_REGISTRY[name_or_fn]


def list_searches() -> list[str]:
    return sorted(SEARCH_REGISTRY.keys())


# ── Helper: build the per-indicator binarized feature matrix once ────────────
def build_signals_df(df: pd.DataFrame, survivors: list[str]) -> pd.DataFrame:
    """Compute each surviving indicator with default params and binarize.

    Used by strategies that need a full indicator-feature matrix.
    """
    from ta_automl.signals.auto_discover import compute_raw, default_params
    from ta_automl.signals.binarizer import binarize

    out: dict[str, pd.Series] = {}
    for key in survivors:
        base = key.split("__")[0]
        try:
            raw_dict = compute_raw(base, df, default_params(base))
        except Exception:
            continue
        raw = raw_dict.get(key)
        if raw is None:
            continue
        try:
            out[key] = binarize(key, raw, df)
        except Exception:
            continue
    return pd.DataFrame(out, index=df.index).fillna(0).astype(int)


# ── Built-in: "weighted" (the original search) ───────────────────────────────
@register_search("weighted")
def search_weighted(ctx: SearchContext) -> SearchResult:
    """Default search: Vizier/FLAML over indicator weights + binarize methods + threshold.

    Each trial: weighted sum of all surviving indicator signals -> threshold
    -> backtest -> loss. The optimizer maximizes the loss.
    """
    from ta_automl.optimization.evaluator import evaluate_trial
    from ta_automl.signals.auto_discover import compute_raw, param_search_space
    from ta_automl.signals.binarizer import INT_TO_METHOD, binarize

    cfg = ctx.config

    def eval_fn(params: dict) -> dict:
        # If Stage-1 tuning produced configs, fill in any missing per-indicator
        # params/binarize from the tuned map (Vizier still searches the rest).
        if ctx.tuned:
            for key, info in ctx.tuned.items():
                bin_key = f"{key}__binarize"
                if bin_key not in params and "binarize" in info:
                    from ta_automl.signals.binarizer import METHOD_TO_INT
                    params[bin_key] = METHOD_TO_INT.get(info["binarize"], 0)
                for p_name, p_val in (info.get("params") or {}).items():
                    full = f"{key}__{p_name}"
                    params.setdefault(full, p_val)
        return evaluate_trial(
            params, ctx.df, ctx.df_test, ctx.survivors, cfg,
            loss_fn=ctx.loss_fn, loss_extra=ctx.loss_extra,
        )

    aggregator = getattr(cfg, "aggregator", "weighted_sum")
    if cfg.optimizer == "vizier":
        from ta_automl.optimization.study import run_vizier_study
        best_params, best_metrics = run_vizier_study(
            ctx.survivors, eval_fn, cfg.trials,
            study_name=f"{cfg.symbol}_{cfg.start}_{cfg.end}",
            metric="objective",
            aggregator=aggregator,
        )
    else:
        from ta_automl.optimization.flaml_search import run_flaml_study
        best_params, best_metrics = run_flaml_study(
            ctx.survivors, eval_fn, cfg.trials, metric="objective",
            aggregator=aggregator,
        )

    # Rebuild signals + combined with the optimal params
    threshold = float(best_params.get("combination_threshold", 0.3))
    signals: dict[str, pd.Series] = {}
    weighted = pd.Series(0.0, index=ctx.df.index)
    total_w = 0.0
    importance: dict[str, float] = {}

    for key in ctx.survivors:
        w = float(best_params.get(f"{key}__weight", 0.0))
        if w < 0.05:
            continue
        base = key.split("__")[0]
        ind_params = {}
        for p_name, (lo, hi, ptype) in param_search_space(base).items():
            full = f"{key}__{p_name}"
            if full in best_params:
                ind_params[p_name] = ptype(best_params[full])
        try:
            raw_dict = compute_raw(base, ctx.df, ind_params or None)
        except Exception:
            continue
        raw = raw_dict.get(key)
        if raw is None:
            raw = next(iter(raw_dict.values()))
        method_idx = int(best_params.get(f"{key}__binarize", 0))
        sig = binarize(key, raw, ctx.df, method=INT_TO_METHOD.get(method_idx, "percentile"))
        signals[key] = sig
        weighted += w * sig
        total_w += w
        importance[key] = w

    if total_w > 0:
        weighted /= total_w
    combined = pd.Series(0, index=ctx.df.index, dtype=int)
    combined[weighted >  threshold] = 1
    combined[weighted < -threshold] = -1

    signals_df = pd.DataFrame(signals, index=ctx.df.index).fillna(0).astype(int)
    return SearchResult(
        best_params=best_params,
        best_metrics=best_metrics,
        signals_df=signals_df,
        combined=combined,
        importance=importance,
    )


# ── Built-in: "shap" — FLAML + CatBoost + SHAP attribution ────────────────────
def _train_automl_classifier(ctx: SearchContext):
    """Shared AutoML core: train a tree-classifier on indicator features,
    return (model, X_full, y_full, X_test, signal_test, automl)."""
    try:
        from flaml.automl import AutoML
    except ImportError:
        try:
            from flaml import AutoML
        except ImportError as e:
            raise RuntimeError(
                "FLAML AutoML not importable. Run: uv pip install -U 'flaml[automl]'"
            ) from e

    cfg = ctx.config
    df = ctx.df

    X_full = build_signals_df(df, ctx.survivors)
    if X_full.shape[1] == 0:
        raise RuntimeError("No usable indicator features after binarization")

    next_ret = df["Close"].pct_change().shift(-1)
    y_full = pd.Series(0, index=df.index, dtype=int)
    y_full[next_ret > 0] = 1
    y_full[next_ret < 0] = -1
    y_full = y_full.iloc[:-1]
    X_full = X_full.iloc[:-1]

    split = int(len(X_full) * cfg.train_ratio)
    X_train, X_test = X_full.iloc[:split], X_full.iloc[split:]
    y_train = y_full.iloc[:split]

    automl = AutoML()
    time_budget = int(ctx.extra.get("shap_time_budget_s",
                       ctx.extra.get("automl_time_budget_s", max(20, cfg.trials))))
    estimator_list = ctx.extra.get("shap_estimator_list",
                       ctx.extra.get("automl_estimator_list", "auto"))
    fit_kwargs: dict[str, Any] = dict(
        X_train=X_train.values, y_train=y_train.values,
        task="classification",
        time_budget=time_budget,
        metric="accuracy",
        eval_method="cv", n_splits=3,
        seed=42, verbose=0,
    )
    if estimator_list != "auto":
        fit_kwargs["estimator_list"] = estimator_list
    automl.fit(**fit_kwargs)

    model = getattr(automl, "model", None)
    if model is not None and hasattr(model, "estimator"):
        model = model.estimator
    if model is None:
        raise RuntimeError("FLAML did not produce a usable model")

    proba = model.predict_proba(X_test.values)
    classes = list(getattr(model, "classes_", [-1, 0, 1]))
    def _idx(c):
        return classes.index(c) if c in classes else None
    i_up, i_dn = _idx(1), _idx(-1)
    p_up = proba[:, i_up] if i_up is not None else np.zeros(len(X_test))
    p_dn = proba[:, i_dn] if i_dn is not None else np.zeros(len(X_test))
    edge = p_up - p_dn

    threshold = float(ctx.extra.get("shap_threshold",
                      ctx.extra.get("automl_threshold", 0.10)))
    signal_test = pd.Series(0, index=X_test.index, dtype=int)
    signal_test[edge >  threshold] = 1
    signal_test[edge < -threshold] = -1
    return model, automl, X_full, X_test, signal_test, threshold


def _backtest_automl_signal(
    ctx: SearchContext, signal_test: pd.Series,
) -> tuple[pd.Series, dict[str, float]]:
    from ta_automl.backtest.strategy import run_backtest
    from ta_automl.optimization.loss import LossContext, get_loss

    cfg = ctx.config
    combined = pd.Series(0, index=ctx.df.index, dtype=int)
    combined.loc[signal_test.index] = signal_test.values
    df_test = ctx.df_test
    combined_test = combined.reindex(df_test.index).fillna(0).astype(int)
    bt = run_backtest(df_test, combined_test, cash=cfg.cash,
                      commission=cfg.commission, allow_short=cfg.allow_short)
    metrics = {k: v for k, v in bt.items() if not k.startswith("_")}
    loss_fn = get_loss(ctx.loss_fn) if ctx.loss_fn else get_loss(cfg.loss)
    metrics["objective"] = float(loss_fn(metrics, LossContext(extra=ctx.loss_extra)))
    return combined, metrics


@register_search("automl")
def search_automl(ctx: SearchContext) -> SearchResult:
    """FLAML AutoML over all surviving indicator signals — black-box, no SHAP.

    Use this when you want the AutoML pipeline (FLAML picks lgbm/xgb/rf/...
    and tunes hyperparams) but don't need per-feature interpretability.
    The displayed "importance" comes from the model's built-in
    feature_importances_ (or get_feature_importance()).

    Pipeline:
      1. Build feature matrix X = signals_df, label y = sign(next_day_return).
      2. Train/test split aligned to ctx.config.train_ratio.
      3. FLAML AutoML picks & tunes the best classifier within time budget.
      4. Predict on test; convert P(up) - P(down) into ternary signal via threshold.
      5. Backtest -> metrics; report model.feature_importances_ for ranking.
    """
    model, automl, X_full, X_test, signal_test, threshold = _train_automl_classifier(ctx)
    combined, metrics = _backtest_automl_signal(ctx, signal_test)

    # Built-in model importance (no SHAP)
    importance_arr = None
    for attr in ("feature_importances_", "get_feature_importance"):
        if hasattr(model, attr):
            fi = getattr(model, attr)
            importance_arr = np.asarray(fi() if callable(fi) else fi)
            break
    if importance_arr is None or len(importance_arr) != X_full.shape[1]:
        importance_arr = np.zeros(X_full.shape[1])

    importance = {col: float(v) for col, v in zip(X_full.columns, importance_arr)}
    best_params: dict[str, Any] = {f"{k}__importance": v for k, v in importance.items()}
    best_params["combination_threshold"] = threshold
    best_params["model_type"] = type(model).__name__
    best_params["best_estimator"] = getattr(automl, "best_estimator", "unknown")

    return SearchResult(
        best_params=best_params,
        best_metrics=metrics,
        signals_df=X_full,
        combined=combined,
        importance=importance,
        extra={"model": model, "threshold": threshold,
               "best_estimator": getattr(automl, "best_estimator", "unknown")},
    )


@register_search("shap")
def search_shap(ctx: SearchContext) -> SearchResult:
    """Same AutoML training as 'automl', plus SHAP attribution for per-feature
    importance — identifies indicators that matter for special events
    (drawdowns, reversals) even when their average revenue contribution is small.

    Use 'automl' instead if you don't need SHAP interpretability — it's the
    same model search without the SHAP dependency.
    """
    try:
        import shap
    except ImportError as e:
        raise RuntimeError(
            "search_shap requires shap. Install with: "
            "uv pip install -e '.[shap]'  (or: pip install shap)\n"
            "If you don't need per-feature interpretability, use --search-strategy automl"
        ) from e

    model, automl, X_full, X_test, signal_test, threshold = _train_automl_classifier(ctx)
    combined, metrics = _backtest_automl_signal(ctx, signal_test)
    X_train = X_full.iloc[:int(len(X_full) * ctx.config.train_ratio)]

    # SHAP attributions on test set — try TreeExplainer (works for
    # lgbm/xgboost/rf/catboost), then fall back to model-agnostic Explainer,
    # then to whatever feature_importances_ the model exposes.
    importance_arr = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.values)
        # multiclass returns list of arrays per class; aggregate by mean(|shap|)
        if isinstance(shap_values, list):
            arr = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            arr = np.abs(shap_values)
            if arr.ndim == 3:    # (n_samples, n_features, n_classes)
                arr = arr.mean(axis=2)
        importance_arr = arr.mean(axis=0)
    except Exception:
        try:
            # Model-agnostic fallback (slower; uses the unified Explainer interface)
            explainer = shap.Explainer(model.predict, X_train.values[:200])
            sv = explainer(X_test.values).values
            importance_arr = np.abs(sv).mean(axis=tuple(range(sv.ndim - 1)))
        except Exception:
            for attr in ("feature_importances_", "get_feature_importance"):
                if hasattr(model, attr):
                    fi = getattr(model, attr)
                    importance_arr = np.asarray(fi() if callable(fi) else fi)
                    break
    if importance_arr is None or len(importance_arr) != X_full.shape[1]:
        importance_arr = np.zeros(X_full.shape[1])

    importance = {col: float(v) for col, v in zip(X_full.columns, importance_arr)}

    # Best-params dict here = SHAP importances + which estimator FLAML picked
    best_params: dict[str, Any] = {f"{k}__importance": v for k, v in importance.items()}
    best_params["combination_threshold"] = threshold
    best_params["model_type"] = type(model).__name__
    best_params["best_estimator"] = getattr(automl, "best_estimator", "unknown")
    if hasattr(model, "get_params"):
        for k, v in model.get_params().items():
            if isinstance(v, (int, float, str, bool)):
                best_params[f"model_{k}"] = v

    return SearchResult(
        best_params=best_params,
        best_metrics=metrics,
        signals_df=X_full,
        combined=combined,
        importance=importance,
        extra={"model": model, "shap_threshold": threshold},
    )
