# Parameter-Aware Stage-1 Screening

By default, Stage-1 screening evaluates every TA-Lib indicator using its
**TA-Lib default parameters** (e.g. `RSI(timeperiod=14)`). The quality filter
then keeps non-degenerate outputs.

**Problem:** an indicator might be useless at default settings but excellent
at a different period. The default-only screen would drop it before Stage 2
ever sees it.

**Solution:** `--tune-screen` runs a small per-indicator hyperparameter search
(Vizier / FLAML / random) **before** the quality filter, picking the best
`(params, binarization_method)` combo for each indicator. The winning configs
are then forwarded to Stage 2 as warm-start anchors.

---

## CLI usage

```bash
# Vizier (GP-Bandit) tuner, 8 trials per indicator (~1264 evals total)
ta-automl --symbol AMD --tune-screen

# FLAML BlendSearch tuner
ta-automl --symbol AMD --tune-screen --tune-optimizer flaml

# Random sampler — fastest, useful as a baseline / when JAX unavailable
ta-automl --symbol AMD --tune-screen --tune-optimizer random --tune-trials 4

# Score function for the per-indicator tuner
ta-automl --symbol AMD --tune-screen --tune-metric abs_sharpe   # default
ta-automl --symbol AMD --tune-screen --tune-metric sharpe       # signed
ta-automl --symbol AMD --tune-screen --tune-metric neg_p_value  # statistical

# Search binarization method too (default true)
ta-automl --symbol AMD --tune-screen --no-tune-method
```

### Cost

For 158 indicators × `tune_trials` trials each:

| `--tune-trials` | Total evals | Wall time (rough) |
|-----------------|-------------|-------------------|
| 4 (random) | ~600 | seconds |
| 8 (vizier) | ~1200 | tens of seconds |
| 20 (vizier) | ~3000 | minutes |

The tuner is intentionally cheap — it's a per-indicator filter, not the main
optimization. Stage 2 still does the heavy lifting on the survivors.

---

## What gets searched

For each indicator name `N`:

1. **Hyperparameters** — every numeric parameter from
   `talib.abstract.Function(N).info["parameters"]` is searched in the same
   range used for Stage 2 (see `signals/auto_discover.param_search_space`).
2. **Binarization method** (when `--tune-method`) — `percentile`, `zscore`,
   `trend`, `price_cross`, `pattern`. Some methods only make sense for
   certain output types, but the tuner just picks the highest-scoring one.

The tuner returns the **best (params, method)** per output key (so for
multi-output indicators like `MACD`, each of `MACD__macd`, `MACD__macdsignal`,
`MACD__macdhist` independently picks its own optimum).

---

## Score functions

`--tune-metric` selects what the per-indicator tuner maximizes:

| Name | Definition |
|------|------------|
| `abs_sharpe` (default) | `|annualized_sharpe|` of the signal applied to next-day returns. Good general-purpose: rewards either-direction predictiveness. |
| `sharpe` | Signed Sharpe; only rewards correctly-oriented signals. Use when you don't want the tuner picking inverted indicators. |
| `neg_p_value` | `-p` from a Mann-Whitney U test of buy-day vs sell-day next-day returns. Rewards statistical separation regardless of magnitude. |

---

## Programmatic API

```python
from ta_automl.config import ScreenConfig
from ta_automl.data.fetcher import fetch_ohlcv
from ta_automl.signals.screener import screen_indicators

df = fetch_ohlcv("AMD", "2018-01-01", "2024-12-31", ".cache")

cfg = ScreenConfig(
    tune_params=True,
    tune_trials=8,
    tune_optimizer="vizier",      # 'vizier' | 'flaml' | 'random'
    tune_metric="abs_sharpe",
    tune_method_choice=True,
)
survivors, tuned_map = screen_indicators(df, cfg, return_tuned=True)

# tuned_map[key] = {"score": float, "params": {...}, "binarize": str}
for key, info in sorted(tuned_map.items(), key=lambda kv: kv[1]["score"], reverse=True)[:5]:
    print(key, info)
```

The `tuned_map` is the data structure passed to the search-strategy layer via
`SearchContext.tuned`, where Stage 2 strategies (e.g. `weighted`) use it as
warm-start anchors:

```python
# in search_weighted (search.py):
for key, info in ctx.tuned.items():
    params.setdefault(f"{key}__{p_name}", p_val)            # tuner's params
    params.setdefault(f"{key}__binarize", METHOD_TO_INT[m]) # tuner's method
```

This means Stage 2 doesn't have to rediscover what Stage 1 already found —
Vizier converges faster and to better local optima.

---

## Tuner internals

The work is in `ta_automl/signals/tuner.py`. The key function is:

```python
def tune_one_indicator(
    name: str,
    df: pd.DataFrame,
    next_ret: pd.Series,
    *,
    n_trials: int = 8,
    optimizer: str = "vizier",     # 'vizier' | 'flaml' | 'random'
    metric: str = "abs_sharpe",
    tune_method: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, Any]]:
    """Returns {key: {"score": float, "params": dict, "binarize": str}}."""
```

If `vizier` fails (e.g. JAX missing), the tuner falls back to random sampling
silently — Stage 1 will not crash. The same is true for `flaml` if FLAML is
not installed.

---

## Extending: custom screening tuners

Want a smarter per-indicator search (e.g. one that uses cross-validation, or
walk-forward folds)? Drop in a replacement and call it from a custom search
strategy. There's no separate registry for tuners — they're called from the
screener; if you need a fundamentally different screening flow, write a custom
search strategy (`docs/CUSTOM_SEARCH.md`) that handles its own indicator
selection from scratch.

A typical pattern: keep `tune_params=False` (skip the built-in tuner) and have
your search strategy compute indicators with whatever logic you want, then
report a uniform `SearchResult`.

---

## Where to look in the code

| File | Purpose |
|------|---------|
| `ta_automl/signals/tuner.py` | Per-indicator hyperparameter search (Vizier/FLAML/random) |
| `ta_automl/signals/screener.py` | Calls the tuner when `cfg.tune_params=True`; returns `tuned_map` |
| `ta_automl/optimization/search.py` | `SearchContext.tuned` carries the map; `weighted` strategy uses it as warm-start |
| `ta_automl/main.py` | `--tune-screen`, `--tune-trials`, `--tune-optimizer`, `--tune-metric`, `--tune-method` flags |
