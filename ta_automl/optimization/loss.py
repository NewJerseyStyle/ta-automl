"""Pluggable loss functions for the optimizer.

A loss function maps a backtest result (dict of metrics) to a single scalar
that the optimizer (Vizier / FLAML) MAXIMIZES. To minimize a quantity (e.g.
drawdown), return its negative.

Built-in losses are registered in LOSS_REGISTRY. Users can add their own via
the @register_loss decorator or by passing a callable directly to the CLI /
StudyConfig.

The signature is:
    loss(metrics: dict[str, float], context: LossContext) -> float

`metrics` keys (set by run_backtest):
    sharpe_ratio, total_return, win_rate, num_trades, max_drawdown
`context` carries optimizer-side info (params, # trades penalty etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class LossContext:
    """Extra info passed to loss functions.

    params       — the trial's hyperparameter dict
    min_trades   — backtests with fewer trades will be penalized externally
    extra        — caller-supplied dict for arbitrary state
    """
    params: dict[str, Any] = field(default_factory=dict)
    min_trades: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


class LossFn(Protocol):
    """Type signature for a loss function."""
    def __call__(self, metrics: dict[str, float], context: LossContext) -> float: ...


# ── Registry ─────────────────────────────────────────────────────────────────
LOSS_REGISTRY: dict[str, LossFn] = {}


def register_loss(name: str) -> Callable[[LossFn], LossFn]:
    """Decorator to register a loss function under `name`.

    Example:
        @register_loss("my_loss")
        def my_loss(metrics, context):
            return metrics["total_return"] - 0.1 * abs(metrics["max_drawdown"])
    """
    def _wrap(fn: LossFn) -> LossFn:
        LOSS_REGISTRY[name] = fn
        return fn
    return _wrap


def get_loss(name_or_fn: str | LossFn) -> LossFn:
    """Look up a loss by name, or return the callable as-is."""
    if callable(name_or_fn):
        return name_or_fn
    if name_or_fn not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss {name_or_fn!r}. "
            f"Registered: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name_or_fn]


# ── Built-in losses ──────────────────────────────────────────────────────────
@register_loss("sharpe")
def loss_sharpe(metrics: dict[str, float], context: LossContext) -> float:
    """Maximize Sharpe ratio (default)."""
    return float(metrics.get("sharpe_ratio", -999.0))


@register_loss("return")
def loss_total_return(metrics: dict[str, float], context: LossContext) -> float:
    """Maximize total return %."""
    return float(metrics.get("total_return", -999.0))


@register_loss("winrate")
def loss_win_rate(metrics: dict[str, float], context: LossContext) -> float:
    """Maximize win rate %."""
    return float(metrics.get("win_rate", 0.0))


@register_loss("min_drawdown")
def loss_min_drawdown(metrics: dict[str, float], context: LossContext) -> float:
    """Minimize maximum drawdown (returned as negative DD so optimizer maximizes).

    backtesting.py reports max_drawdown as a non-positive percentage
    (e.g. -25.4 means a 25.4% drawdown). Returning that value directly causes
    the optimizer (which maximizes) to push drawdown toward 0 — i.e. lower DD.

    Adds a small return-incentive so the optimizer doesn't trivially win by
    never trading: tie-broken by total_return / 100.
    """
    dd = float(metrics.get("max_drawdown", -999.0))         # in [-100, 0]
    ret = float(metrics.get("total_return", 0.0))           # in % (any sign)
    # Strongly weight DD reduction; weakly reward returns to avoid no-trade optima
    return dd + 0.01 * ret


@register_loss("calmar")
def loss_calmar(metrics: dict[str, float], context: LossContext) -> float:
    """Maximize Calmar ratio = annualized_return / |max_drawdown|.

    Useful when both return and drawdown matter. Bounded by a small epsilon
    to avoid division-by-zero when no trades occur.
    """
    ret = float(metrics.get("total_return", 0.0))
    dd  = abs(float(metrics.get("max_drawdown", 0.0)))
    return ret / max(dd, 1e-3)


@register_loss("sharpe_dd_penalty")
def loss_sharpe_with_dd_penalty(metrics: dict[str, float], context: LossContext) -> float:
    """Sharpe minus a drawdown penalty.

    Penalty weight can be customized via context.extra['dd_weight'] (default 0.05).
    """
    sh = float(metrics.get("sharpe_ratio", -999.0))
    dd = abs(float(metrics.get("max_drawdown", 0.0)))
    weight = float(context.extra.get("dd_weight", 0.05))
    return sh - weight * dd


def list_losses() -> list[str]:
    """Return all registered loss names."""
    return sorted(LOSS_REGISTRY.keys())
