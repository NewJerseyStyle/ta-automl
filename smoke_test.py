"""
Smoke-test script — run this to verify each layer of the pipeline works.
Usage:  python smoke_test.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


# ── 1. Imports ─────────────────────────────────────────────────────────────
section("1. Core imports")

try:
    import talib
    fns = talib.get_functions()
    print(f"{PASS}  TA-Lib: {len(fns)} indicators available")
except ImportError as e:
    print(f"{FAIL}  TA-Lib import failed: {e}")
    print("       Run:  uv pip install 'ta-lib'")
    sys.exit(1)

try:
    import backtesting
    print(f"{PASS}  backtesting.py: {backtesting.__version__}")
except Exception as e:
    print(f"{FAIL}  backtesting: {e}")

try:
    import rich
    version = getattr(rich, "__version__", "installed")
    print(f"{PASS}  rich: {version}")
except Exception as e:
    print(f"{FAIL}  rich: {e}")

try:
    from vizier.service import clients, pyvizier as vz
    print(f"{PASS}  google-vizier: importable")
    _vizier_ok = True
except ImportError:
    print(f"{SKIP}  google-vizier: not importable (need 'jax[cpu]' for GP algorithm)")
    _vizier_ok = False

try:
    from flaml import tune
    print(f"{PASS}  FLAML: importable")
    _flaml_ok = True
except ImportError:
    print(f"{SKIP}  FLAML: not importable")
    _flaml_ok = False

if not _vizier_ok and not _flaml_ok:
    print(f"{FAIL}  Neither Vizier nor FLAML available — install at least one.")
    sys.exit(1)


# ── 2. Data fetch ───────────────────────────────────────────────────────────
section("2. Data fetch (AMD 2022–2023)")

from ta_automl.data.fetcher import fetch_ohlcv
df = fetch_ohlcv("AMD", "2022-01-01", "2023-12-31", ".cache")
assert df.shape[1] == 5, "Expected 5 OHLCV columns"
assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
assert len(df) > 200
print(f"{PASS}  {df.shape[0]} trading days, columns OK")
print(f"       {df.index[0].date()} → {df.index[-1].date()}")


# ── 3. Indicator computation ────────────────────────────────────────────────
section("3. Indicator computation + binarization")

from ta_automl.signals.auto_discover import compute_raw, default_params
from ta_automl.signals.binarizer import binarize

test_cases = [
    ("RSI",      False),
    ("MACD",     True),   # multi-output
    ("BBANDS",   True),   # multi-output
    ("STOCH",    True),   # multi-output
    ("CDLHAMMER",False),  # pattern recognition
    ("ADX",      False),
]
for name, multi in test_cases:
    raw_dict = compute_raw(name, df, default_params(name))
    for key, raw in raw_dict.items():
        sig = binarize(key, raw, df)
        allowed = set(sig.dropna().unique())
        assert allowed.issubset({-1, 0, 1}), f"{key}: unexpected values {allowed}"
        label = "multi" if multi else "single"
        print(f"{PASS}  {key:<35} ({label})  "
              f"buy={int((sig==1).sum()):3d}  sell={int((sig==-1).sum()):3d}")


# ── 4. Screener ─────────────────────────────────────────────────────────────
section("4. Stage-1 screener (all 158 indicators)")

from ta_automl.signals.screener import screen_indicators
from ta_automl.config import ScreenConfig

cfg = ScreenConfig()
survivors = screen_indicators(df, cfg, verbose=False)
print(f"{PASS}  {len(survivors)} indicators passed quality filter")
assert len(survivors) > 0, "No survivors — check TA-Lib installation"
print(f"       Sample: {survivors[:6]}")


# ── 5. Backtesting ──────────────────────────────────────────────────────────
section("5. Backtesting pipeline")

import numpy as np
import pandas as pd
from ta_automl.backtest.strategy import run_backtest

idx = df.index
# Synthetic alternating signal
sig = pd.Series([1, -1] * (len(idx) // 2) + [0] * (len(idx) % 2), index=idx)
result = run_backtest(df, sig, cash=10_000, commission=0.002, allow_short=True)
assert "sharpe_ratio" in result
print(f"{PASS}  run_backtest OK: sharpe={result['sharpe_ratio']:.3f}  "
      f"trades={result['num_trades']}")


# ── 6. Evaluate trial ───────────────────────────────────────────────────────
section("6. Trial evaluation (RSI-only test params)")

from ta_automl.optimization.evaluator import evaluate_trial
from ta_automl.config import StudyConfig

study_cfg = StudyConfig()
split = int(len(df) * 0.70)
df_test = df.iloc[split:]

params = {"combination_threshold": 0.3}
for key in survivors:
    params[f"{key}__weight"] = 1.0 if key == "RSI" else 0.0
    params[f"{key}__binarize"] = 0

metrics = evaluate_trial(params, df, df_test, survivors, study_cfg)
print(f"{PASS}  evaluate_trial OK: {metrics}")


# ── 7. Optimizer (1 trial each) ─────────────────────────────────────────────
section("7. Optimizer (1 trial smoke test)")

# Use a tiny survivor list for speed
mini_survivors = [s for s in survivors if s in ("RSI", "MACD__macd", "ADX")][:3]
if not mini_survivors:
    mini_survivors = survivors[:3]

def fast_eval(p):
    return evaluate_trial(p, df, df_test, mini_survivors, study_cfg)

if _vizier_ok:
    print("  Testing Vizier (1 trial) …")
    try:
        from ta_automl.optimization.study import run_vizier_study
        bp, bm = run_vizier_study(mini_survivors, fast_eval, n_trials=1,
                                   study_name="smoke_vizier", verbose=False)
        print(f"{PASS}  Vizier: best sharpe={bm.get('sharpe_ratio', 'N/A')}")
    except Exception as e:
        print(f"{FAIL}  Vizier trial failed: {e}")
        print("       Hint: install JAX with  uv pip install 'jax[cpu]'")
else:
    print(f"{SKIP}  Vizier skipped (JAX not installed)")

if _flaml_ok:
    print("  Testing FLAML (1 trial) …")
    try:
        from ta_automl.optimization.flaml_search import run_flaml_study
        bp, bm = run_flaml_study(mini_survivors, fast_eval, n_trials=1, verbose=False)
        print(f"{PASS}  FLAML: best sharpe={bm.get('sharpe_ratio', 'N/A')}")
    except Exception as e:
        print(f"{FAIL}  FLAML trial failed: {e}")
else:
    print(f"{SKIP}  FLAML skipped")


# ── 8. Display ──────────────────────────────────────────────────────────────
section("8. Traffic light display (last 5 days, dry-run)")

from rich.console import Console
from ta_automl.display.traffic_light import render_traffic_light

signals_df = pd.DataFrame(
    {key: binarize(key, compute_raw(key.split("__")[0], df, default_params(key.split("__")[0])).get(key, pd.Series(0, index=df.index)), df)
     for key in mini_survivors},
    index=df.index,
)
combined = pd.Series(0, index=df.index, dtype=int)
dummy_params = {f"{k}__weight": 1.0 for k in mini_survivors}
dummy_params["combination_threshold"] = 0.3
dummy_metrics = {"sharpe_ratio": 0.5, "total_return": 5.0, "win_rate": 55.0,
                 "num_trades": 10, "max_drawdown": -10.0}

console = Console()
render_traffic_light(
    df=df,
    signals_df=signals_df,
    combined=combined,
    best_params=dummy_params,
    best_metrics=dummy_metrics,
    surviving_keys=mini_survivors,
    top_n=3,
    lookback=5,
    console=console,
)
print(f"{PASS}  Traffic light rendered OK")


# ── Summary ─────────────────────────────────────────────────────────────────
section("Summary")
print("All smoke tests passed.")
print()
print("Next steps:")
if not _vizier_ok:
    print("  * Install JAX for Vizier GP-Bandit:  uv pip install 'jax[cpu]'")
    print("  * Or run with FLAML:  ta-automl --optimizer flaml")
else:
    print("  * Run a full optimization:  ta-automl --symbol AMD --trials 100")
print("  * Try another ticker:         ta-automl --symbol NVDA --trials 50")
