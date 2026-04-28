"""User-supplied combiner registry.

A combiner takes the per-indicator signal matrix and produces a single
{-1, 0, +1} series. This is the "no-AutoML" path: you decide the rule.

    fn(signals: pd.DataFrame, df: pd.DataFrame, **params) -> pd.Series

Examples of valid rules:
    • intersection: BUY only when ALL indicators agree
    • voting:       sign of the sum
    • gated:        ema_state must be UP, then RSI extremes trigger entries
    • regime-aware: different rule when ATR > median(ATR)

Register with @register_combiner("my_name"). The combiner is also auto-bridged
into SEARCH_REGISTRY so it can be picked from the CLI / GUI exactly like
weighted/automl/shap.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from ta_automl.optimization.search import (
    SearchContext,
    SearchResult,
    register_search,
)


class CombinerFn(Protocol):
    def __call__(
        self, signals: pd.DataFrame, df: pd.DataFrame, **params: Any
    ) -> pd.Series: ...


COMBINER_REGISTRY: dict[str, dict[str, Any]] = {}


def register_combiner(
    name: str,
    *,
    indicators: list[str] | None = None,
    expose_to_search: bool = True,
) -> Callable[[CombinerFn], CombinerFn]:
    """Decorator: register a user combiner under `name`.

    Args:
        indicators: optional default indicator list this combiner expects. If
            omitted, the combiner accepts whatever signal columns it gets and
            should reference them by name internally.
        expose_to_search: if True (default), also register a search strategy
            of the same name that runs this combiner as a no-AutoML one-shot.
            Pick it via --search-strategy <name> or in the GUI.
    """
    def _wrap(fn: CombinerFn) -> CombinerFn:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) < 2:
            raise TypeError(
                f"@register_combiner('{name}'): function must accept "
                f"(signals, df, **params)."
            )
        defaults = {
            p.name: p.default for p in params[2:]
            if p.default is not inspect.Parameter.empty
        }
        COMBINER_REGISTRY[name] = {
            "fn": fn,
            "doc": (fn.__doc__ or "").strip(),
            "defaults": defaults,
            "indicators": list(indicators or []),
        }

        if expose_to_search:
            _expose_as_search(name)
        return fn
    return _wrap


def get_combiner(name: str) -> CombinerFn:
    if name not in COMBINER_REGISTRY:
        raise KeyError(
            f"Unknown combiner {name!r}. Registered: {sorted(COMBINER_REGISTRY)}"
        )
    return COMBINER_REGISTRY[name]["fn"]


def list_combiners() -> list[str]:
    return sorted(COMBINER_REGISTRY.keys())


def apply_combiner(
    name: str,
    signals: pd.DataFrame,
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Run a registered combiner and coerce output to {-1, 0, +1} ints."""
    entry = COMBINER_REGISTRY[name]
    merged = {**entry["defaults"], **(params or {})}
    out = entry["fn"](signals, df, **merged)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index)
    return out.reindex(df.index).fillna(0).clip(-1, 1).astype(int)


# ── Bridge: combiner -> search strategy ──────────────────────────────────────
def _expose_as_search(name: str) -> None:
    """Wrap a combiner as a no-AutoML search strategy of the same name.

    The "search" runs once: build the survivors signal matrix, apply the user's
    combiner, run the held-out backtest, return metrics. The optimizer's trial
    count is ignored — that's the whole point of bring-your-own-rule.
    """
    @register_search(name)
    def _runner(ctx: SearchContext) -> SearchResult:
        from ta_automl.backtest.strategy import run_backtest
        from ta_automl.optimization.search import build_signals_df

        # Combine TA-Lib survivors with any registered user indicators that
        # the combiner expects but didn't come from screening.
        from ta_automl.sdk.indicators import (
            INDICATOR_REGISTRY, compute_user_signal,
        )

        survivors_df = build_signals_df(ctx.df, ctx.survivors)

        # Layer in user indicators
        user_cols: dict[str, pd.Series] = {}
        for ind_name in INDICATOR_REGISTRY:
            try:
                user_cols[ind_name] = compute_user_signal(ind_name, ctx.df)
            except Exception:
                continue
        if user_cols:
            user_df = pd.DataFrame(user_cols, index=ctx.df.index)
            signals_df = pd.concat([survivors_df, user_df], axis=1)
        else:
            signals_df = survivors_df

        combined = apply_combiner(name, signals_df, ctx.df)
        # Backtest on test slice
        combined_test = combined.reindex(ctx.df_test.index).fillna(0).astype(int)
        bt = run_backtest(
            ctx.df_test, combined_test,
            cash=ctx.config.cash,
            commission=ctx.config.commission,
            allow_short=ctx.config.allow_short,
        )
        metrics = {
            "sharpe_ratio": float(bt.get("sharpe_ratio", 0) or 0),
            "total_return": float(bt.get("total_return", 0) or 0),
            "max_drawdown": float(bt.get("max_drawdown", 0) or 0),
            "win_rate":     float(bt.get("win_rate", 0) or 0),
            "n_trades":     int(bt.get("n_trades", 0) or 0),
            "objective":    float(bt.get("sharpe_ratio", 0) or 0),
        }
        importance = {c: float(np.corrcoef(
            signals_df[c].fillna(0).values,
            combined.fillna(0).values,
        )[0, 1] or 0) for c in signals_df.columns}
        return SearchResult(
            best_params={"combiner": name, **COMBINER_REGISTRY[name]["defaults"]},
            best_metrics=metrics,
            signals_df=signals_df,
            combined=combined,
            importance=importance,
        )
    _runner.__doc__ = f"User combiner: {name} (no AutoML — runs the rule once)."
