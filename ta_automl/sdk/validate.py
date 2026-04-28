"""One-shot helper: take an idea, return a backtest result.

This is the primary surface for tutorial users. It hides the screener,
optimizer, and search registry — instead it computes ONLY the indicators
the user listed, applies their combiner (or a default 'sum-of-signs' rule),
runs a train/test backtest, and returns metrics + a Plotly figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ta_automl.sdk.combiners import apply_combiner, COMBINER_REGISTRY
from ta_automl.sdk.indicators import INDICATOR_REGISTRY, compute_user_signal


@dataclass
class IdeaResult:
    """Output of validate_idea(). All fields are picklable / JSON-friendly."""
    symbol: str
    metrics: dict[str, float] = field(default_factory=dict)
    signals: pd.DataFrame | None = None    # per-indicator {-1,0,+1} matrix
    combined: pd.Series | None = None      # final combined signal
    equity: pd.Series | None = None        # equity curve on test slice
    df_train: pd.DataFrame | None = None
    df_test: pd.DataFrame | None = None
    figure: Any = None                     # plotly Figure

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.symbol}  Sharpe={m.get('sharpe_ratio', 0):.2f}  "
            f"Return={m.get('total_return', 0):.1f}%  "
            f"MaxDD={m.get('max_drawdown', 0):.1f}%  "
            f"Trades={m.get('n_trades', 0)}"
        )


def validate_idea(
    *,
    symbol: str,
    start: str,
    end: str,
    indicators: list[str],
    combiner: str | None = None,
    indicator_params: dict[str, dict[str, Any]] | None = None,
    combiner_params: dict[str, Any] | None = None,
    cash: float = 10_000.0,
    commission: float = 0.002,
    allow_short: bool = True,
    train_ratio: float = 0.70,
    cache_dir: str | Path = ".cache",
    plot: bool = True,
) -> IdeaResult:
    """Validate a user-defined indicator combination via backtest.

    Args:
        indicators: names previously decorated with @register_indicator. The
            list also accepts plain TA-Lib indicator KEYS (e.g. "RSI",
            "MACD__macdhist"); they're computed with default params.
        combiner: name from @register_combiner. If None, uses "sum_of_signs"
            (which is auto-registered: BUY when sum > 0, SELL when < 0).

    Returns:
        IdeaResult with metrics + Plotly figure.
    """
    import ta_automl  # noqa: F401  triggers compat patches
    from ta_automl.data.fetcher import fetch_ohlcv

    df = fetch_ohlcv(symbol, start, end, cache_dir=cache_dir)
    if df is None or len(df) < 60:
        raise RuntimeError(
            f"Not enough data for {symbol} {start}→{end} "
            f"(got {0 if df is None else len(df)} rows)."
        )

    split_idx = int(len(df) * train_ratio)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    # ── Compute the requested indicators ─────────────────────────────────
    indicator_params = indicator_params or {}
    sig_cols: dict[str, pd.Series] = {}
    for name in indicators:
        if name in INDICATOR_REGISTRY:
            sig_cols[name] = compute_user_signal(
                name, df, indicator_params.get(name)
            )
        else:
            # Treat as TA-Lib key
            sig_cols[name] = _compute_talib_signal(name, df)

    signals_df = pd.DataFrame(sig_cols, index=df.index).fillna(0).astype(int)

    # ── Combine ──────────────────────────────────────────────────────────
    combiner_name = combiner or "sum_of_signs"
    if combiner_name not in COMBINER_REGISTRY:
        raise KeyError(
            f"Unknown combiner {combiner_name!r}. "
            f"Registered: {sorted(COMBINER_REGISTRY)}"
        )
    combined = apply_combiner(combiner_name, signals_df, df, combiner_params)

    # ── Backtest on test slice ───────────────────────────────────────────
    from ta_automl.backtest.strategy import run_backtest
    combined_test = combined.reindex(df_test.index).fillna(0).astype(int)
    bt = run_backtest(
        df_test, combined_test,
        cash=cash, commission=commission, allow_short=allow_short,
    )
    metrics = {
        "sharpe_ratio": float(bt.get("sharpe_ratio", 0) or 0),
        "total_return": float(bt.get("total_return", 0) or 0),
        "max_drawdown": float(bt.get("max_drawdown", 0) or 0),
        "win_rate":     float(bt.get("win_rate", 0) or 0),
        "n_trades":     int(bt.get("n_trades", 0) or 0),
    }
    equity = _equity_curve(df_test, combined_test, cash, commission, allow_short)

    fig = _make_figure(symbol, df_test, equity, combined_test) if plot else None

    return IdeaResult(
        symbol=symbol, metrics=metrics,
        signals=signals_df, combined=combined, equity=equity,
        df_train=df_train, df_test=df_test, figure=fig,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────
def _compute_talib_signal(key: str, df: pd.DataFrame) -> pd.Series:
    from ta_automl.signals.auto_discover import compute_raw, default_params
    from ta_automl.signals.binarizer import binarize

    base = key.split("__")[0]
    raw = compute_raw(base, df, default_params(base)).get(
        key, compute_raw(base, df, default_params(base)).get(base)
    )
    if raw is None:
        return pd.Series(0, index=df.index, dtype=int)
    return binarize(key, raw, df).reindex(df.index).fillna(0).astype(int)


def _equity_curve(df, signal, cash, commission, allow_short):
    px = df["Close"].astype(float)
    rets = px.pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0).astype(float)
    if not allow_short:
        pos = pos.clip(lower=0)
    pnl = pos * rets - pos.diff().abs().fillna(0.0) * commission
    return (1.0 + pnl).cumprod() * cash


def _make_figure(symbol, df_test, equity, combined):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    bh = (1 + df_test["Close"].pct_change().fillna(0)).cumprod() * float(equity.iloc[0])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} — strategy vs buy & hold", "Combined signal"),
    )
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy",
                             line=dict(color="#2e86de", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold",
                             line=dict(color="#aaa", width=2, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=combined.index, y=combined.values, name="Signal",
                             mode="lines", line=dict(color="#27ae60", width=1)),
                  row=2, col=1)
    fig.update_layout(hovermode="x unified", height=620,
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ── Default combiner: sum-of-signs ───────────────────────────────────────────
from ta_automl.sdk.combiners import register_combiner as _reg_combiner


@_reg_combiner("sum_of_signs", expose_to_search=False)
def _sum_of_signs(signals: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Default combiner: BUY when more indicators say buy than sell.

    The signed sum across all signal columns, then sign() of the result.
    Equivalent to a simple majority vote.
    """
    return np.sign(signals.sum(axis=1)).astype(int)
