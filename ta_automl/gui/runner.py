"""Background pipeline runner used by the GUI.

The full pipeline (fetch → screen → search) takes minutes, so we run it on a
worker thread and let the Dash app poll a small in-memory state object to draw
progress + final results.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class RunState:
    run_id: str
    status: str = "pending"            # pending | running | done | error
    step: str = ""                     # human-readable current step
    progress: float = 0.0              # 0..1
    log: list[str] = field(default_factory=list)
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0

    def append(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")


# Module-level registry. The Dash app shares this between callbacks via run_id.
_RUNS: dict[str, RunState] = {}
_LOCK = threading.Lock()


def get_run(run_id: str) -> Optional[RunState]:
    with _LOCK:
        return _RUNS.get(run_id)


def _set(run_id: str, **fields: Any) -> None:
    with _LOCK:
        st = _RUNS[run_id]
        for k, v in fields.items():
            setattr(st, k, v)


def start_run(params: dict[str, Any]) -> str:
    """Start a pipeline run in a background thread; return the run_id."""
    run_id = uuid.uuid4().hex[:8]
    state = RunState(run_id=run_id, status="pending", started_at=time.time())
    with _LOCK:
        _RUNS[run_id] = state
    threading.Thread(target=_worker, args=(run_id, params), daemon=True).start()
    return run_id


def _worker(run_id: str, params: dict[str, Any]) -> None:
    st = _RUNS[run_id]
    try:
        st.status = "running"
        _execute(st, params)
        st.status = "done"
        st.progress = 1.0
        st.ended_at = time.time()
        st.append(f"Completed in {st.ended_at - st.started_at:.1f}s")
    except Exception as exc:  # noqa: BLE001
        st.status = "error"
        st.error = f"{type(exc).__name__}: {exc}"
        st.append(f"ERROR: {st.error}")
        st.append(traceback.format_exc())
        st.ended_at = time.time()


def _execute(st: RunState, params: dict[str, Any]) -> None:
    """Mirror of ta_automl.main.cli, instrumented for progress."""
    import ta_automl  # noqa: F401  triggers compat patches
    from ta_automl.config import ScreenConfig, StudyConfig
    from ta_automl.data.fetcher import fetch_ohlcv
    from ta_automl.optimization.search import SearchContext, get_search
    from ta_automl.signals.screener import screen_indicators

    symbol = params["symbol"].upper().strip()
    start = params["start"]
    end = params["end"]

    config = StudyConfig(
        symbol=symbol,
        start=start,
        end=end,
        trials=int(params["trials"]),
        loss=params["loss"],
        cash=float(params["cash"]),
        commission=float(params["commission"]),
        train_ratio=float(params["train_ratio"]),
        allow_short=bool(params["allow_short"]),
        optimizer=params["optimizer"],
        top_n=int(params["top_n"]),
        lookback=int(params["lookback"]),
        save_html=False,
        output_dir=Path(params.get("output_dir", "results")),
        cache_dir=Path(params.get("cache_dir", ".cache")),
        screen=ScreenConfig(
            tune_params=bool(params["tune_screen"]),
            tune_trials=int(params["tune_trials"]),
            tune_optimizer=params.get("tune_optimizer", "random"),
        ),
    )

    # ── Fetch ─────────────────────────────────────────────────────────────
    st.step = "Fetching OHLCV data"
    st.progress = 0.05
    st.append(f"Fetching {symbol} {start} → {end}")
    df = fetch_ohlcv(symbol, start, end, cache_dir=config.cache_dir)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {symbol} between {start} and {end}.")
    st.append(f"Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")

    split_idx = int(len(df) * config.train_ratio)
    df_test = df.iloc[split_idx:]

    # ── Screen ────────────────────────────────────────────────────────────
    st.step = "Stage 1: screening ~158 indicators"
    st.progress = 0.15
    if config.screen.tune_params:
        st.append(f"Parameter-aware screening "
                  f"(tuner={config.screen.tune_optimizer}, "
                  f"{config.screen.tune_trials} trials/indicator)")
    else:
        st.append("Default-parameter screening")

    result = screen_indicators(df, config.screen, verbose=False, return_tuned=True)
    survivors, tuned_map = result if isinstance(result, tuple) else (result, {})
    if not survivors:
        raise RuntimeError("No indicators passed Stage-1 screening. "
                           "Try a longer date range or a different symbol.")
    st.append(f"{len(survivors)} indicators survived screening")

    # ── Stage-2 search ────────────────────────────────────────────────────
    st.step = (f"Stage 2: searching combinations "
               f"({config.optimizer}, {config.trials} trials)")
    st.progress = 0.40

    search_strategy = params["search_strategy"]
    search_ctx = SearchContext(
        df=df, df_test=df_test, survivors=survivors, config=config,
        loss_fn=config.loss, loss_extra={}, tuned=tuned_map,
    )
    search_callable = get_search(search_strategy)
    sresult = search_callable(search_ctx)

    st.progress = 0.90
    st.step = "Building charts"

    best_metrics = sresult.best_metrics
    signals_df = sresult.signals_df
    combined = sresult.combined
    importance = sresult.importance or {}

    # Score columns by absolute weight if importance is empty
    if not importance and signals_df is not None:
        importance = {c: float(signals_df[c].abs().mean()) for c in signals_df.columns}

    # Build a compact result the GUI can plot from
    payload = _payload(df, df_test, signals_df, combined, importance,
                       best_metrics, sresult.best_params, survivors,
                       symbol, config)
    st.result = payload
    st.append(f"Sharpe={best_metrics.get('sharpe_ratio', 0):.3f}  "
              f"Return={best_metrics.get('total_return', 0):.1f}%  "
              f"MaxDD={best_metrics.get('max_drawdown', 0):.1f}%")


def _payload(df, df_test, signals_df, combined, importance,
             best_metrics, best_params, survivors,
             symbol, config) -> dict[str, Any]:
    """Convert pandas/numpy objects to plain JSON-serializable dicts."""
    top_n = config.top_n
    lookback = config.lookback

    # Top-N indicators by |importance|
    items = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_keys = [k for k, _ in items[:top_n]]

    # Recent traffic-light slice
    if signals_df is not None and len(signals_df):
        sig_recent = signals_df[top_keys].tail(lookback) if top_keys else signals_df.tail(lookback)
    else:
        sig_recent = pd.DataFrame()
    combined_recent = combined.tail(lookback) if combined is not None else pd.Series(dtype=int)

    # Equity curve estimate from the combined signal on the test set
    equity = _equity_curve(df_test, combined.reindex(df_test.index).fillna(0).astype(int),
                           cash=config.cash, commission=config.commission,
                           allow_short=config.allow_short)
    bh = _buy_hold(df_test, cash=config.cash)

    return {
        "symbol": symbol,
        "metrics": {
            "sharpe": float(best_metrics.get("sharpe_ratio", 0.0) or 0.0),
            "total_return": float(best_metrics.get("total_return", 0.0) or 0.0),
            "max_drawdown": float(best_metrics.get("max_drawdown", 0.0) or 0.0),
            "win_rate": float(best_metrics.get("win_rate", 0.0) or 0.0),
            "n_trades": int(best_metrics.get("n_trades", 0) or 0),
        },
        "top_keys": top_keys,
        "importance": {k: float(v) for k, v in items[: max(20, top_n)]},
        "n_survivors": len(survivors),
        # series — encoded as (timestamps, values) lists for JSON
        "price": _series_to_lists(df["Close"]),
        "price_test": _series_to_lists(df_test["Close"]),
        "combined_recent": _series_to_lists(combined_recent),
        "signals_recent": {
            "index": [t.isoformat() for t in sig_recent.index],
            "columns": list(sig_recent.columns),
            "values": sig_recent.values.tolist(),
        },
        "equity": equity,
        "buy_hold": bh,
        "best_params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                        for k, v in best_params.items()},
    }


def _series_to_lists(s: pd.Series) -> dict[str, list]:
    if s is None or len(s) == 0:
        return {"x": [], "y": []}
    return {
        "x": [t.isoformat() for t in s.index],
        "y": [float(v) if pd.notna(v) else None for v in s.values],
    }


def _equity_curve(df, signal, cash=10_000.0, commission=0.002, allow_short=True):
    """Quick equity-curve estimate: hold position = signal, P&L = signal[t-1] * pct_change[t]."""
    px = df["Close"].astype(float)
    rets = px.pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0).astype(float)
    if not allow_short:
        pos = pos.clip(lower=0)
    pnl = pos * rets
    # Commission: charge on absolute change in position
    turnover = pos.diff().abs().fillna(0.0)
    pnl = pnl - turnover * commission
    equity = (1.0 + pnl).cumprod() * cash
    return _series_to_lists(equity)


def _buy_hold(df, cash=10_000.0):
    px = df["Close"].astype(float)
    rets = px.pct_change().fillna(0.0)
    eq = (1.0 + rets).cumprod() * cash
    return _series_to_lists(eq)
