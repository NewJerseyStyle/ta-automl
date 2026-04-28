# ta-automl

AutoML hyperparameter optimizer for TA-Lib technical analysis signals with backtesting.

> **v0.2.1 — GUI + Developer SDK.** Two new ways to use this tool:
>
> 1. **Plotly Dash GUI** — `ta-automl-gui` opens a browser at
>    <http://127.0.0.1:8050>. Every option has a **(?)** button that explains
>    the concept with a plain-English analogy. Includes a **🎓 Tutorial mode**
>    overlay (good for live workshops).
> 2. **Developer SDK** — write your own indicator + your own combination rule
>    and let ta-automl **validate the idea via backtest** (no AutoML required).
>    Scaffold with `ta-automl-dev new-indicator <name>` /
>    `new-combiner <name>`. See [`docs/EXTENDING.md`](docs/EXTENDING.md),
>    [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md), and the
>    [tutorial](docs/TUTORIAL_CUSTOM_INDICATOR.md) /
>    [workshop](docs/TUTORIAL_ALGO_TRADING_WORKSHOP.md) docs.


<details>
  <summary>Click to view large screenshot</summary>
  <img width="100%" alt="image" src="https://github.com/user-attachments/assets/818ff255-9f21-46e2-892d-9677a26185aa" />
  <img width="100%" alt="image" src="https://github.com/user-attachments/assets/7ea93ca2-d7b4-4065-8531-b5194243b26a" />
</details>


## How it works

**Stage 1 — Screening:** Loops over all ~158 TA-Lib indicators via `talib.get_functions()`, computes each (default mode: TA-Lib defaults; with `--tune-screen`: a small per-indicator Vizier/FLAML/random search picks better params and binarization method), binarizes the output to `{-1, 0, +1}`, and keeps those with sufficient signal density. Typically yields 100–150 indicator outputs as candidates. The tuned configs are then handed to Stage 2 as warm-start anchors so Vizier doesn't re-discover them.

**Stage 2 — Optimization:** Uses Google Vizier (GP-Bandit, in-process) or FLAML (BlendSearch) to search a high-dimensional parameter space: indicator-specific period/threshold hyperparameters + per-indicator combination weights + a global threshold. Each trial runs a full backtest on the held-out test set and returns the Sharpe ratio. Best combination is displayed as a traffic-light terminal table.

---

## Setup

### Requirements
- Python 3.11

### Install

```bash
pip install ta-automl
```

---

## Usage

```bash
# Default: AMD, 2018–2024, 100 Vizier trials
ta-automl --symbol AMD

# Use FLAML if Vizier/JAX is not available
ta-automl --symbol AMD --optimizer flaml

# Parameter-aware Stage-1 screening (better quality survivors, slower)
ta-automl --symbol AMD --tune-screen --tune-trials 8

# Quick tuned screening with random search (fastest tuning option)
ta-automl --symbol AMD --tune-screen --tune-optimizer random --tune-trials 4

# Custom symbol and date range
ta-automl --symbol NVDA --start 2020-01-01 --end 2024-12-31 --trials 50

# Fewer trials for a quick test
ta-automl --symbol AMD --start 2022-01-01 --end 2023-12-31 --trials 5

# Long-only strategy, save HTML chart
ta-automl --symbol TSLA --no-short --save-html

# All options
ta-automl --help
```

### All CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | `AMD` | Ticker symbol |
| `--start` | `2018-01-01` | Start date |
| `--end` | `2024-12-31` | End date |
| `--trials` | `100` | Optimizer trial count |
| `--optimizer` | `vizier` | `vizier` or `flaml` |
| `--loss` | `sharpe` | Loss function name (see `--list-losses`) or `module:fn` |
| `--list-losses` | off | Print all registered losses and exit |
| `--metric` | (deprecated) | Legacy alias; prefer `--loss` |
| `--top-n` | `8` | Indicators shown in traffic light |
| `--lookback` | `30` | Recent days in traffic light |
| `--cash` | `10000` | Starting cash for backtest |
| `--commission` | `0.002` | Per-trade commission (0.2%) |
| `--train-ratio` | `0.70` | Train/test split (70% train) |
| `--no-short` | off | Long-only (default: long+short) |
| `--save-html` | off | Save interactive backtesting chart |
| `--p-threshold` | `0.20` | Stage-1 p-value cutoff (if `--p-filter` enabled) |
| `--min-sharpe` | `-2.0` | Stage-1 minimum quick Sharpe |
| `--no-bonferroni` | off | Disable Bonferroni correction |
| `--tune-screen` | off | Stage-1 hyperparameter search per indicator (Vizier/FLAML/random) |
| `--tune-trials` | `8` | Trials per indicator during Stage-1 tuning |
| `--tune-optimizer` | `vizier` | Optimizer for Stage-1 tuning: `vizier`, `flaml`, `random` |
| `--tune-metric` | `abs_sharpe` | Score: `abs_sharpe`, `sharpe`, `neg_p_value` |
| `--tune-method` / `--no-tune-method` | on | Also search binarization methods |
| `--output-dir` | `results` | Where to save JSON results |
| `--cache-dir` | `.cache` | Local OHLCV parquet cache |

---

## For Developer: Customizing the search

Two extension points, both registry-based and CLI-discoverable:

| Concept | Doc | Flags |
|---------|-----|-------|
| **Loss function** — what the optimizer maximizes | [docs/CUSTOM_LOSS.md](docs/CUSTOM_LOSS.md) | `--loss`, `--list-losses` |
| **Search strategy** — how indicators are combined | [docs/CUSTOM_SEARCH.md](docs/CUSTOM_SEARCH.md) | `--search-strategy`, `--list-searches` |
| **Parameter-aware Stage-1 screening** — find good per-indicator hyperparameters before screening | [docs/PARAMETER_AWARE_SCREENING.md](docs/PARAMETER_AWARE_SCREENING.md) | `--tune-screen`, `--tune-trials`, `--tune-optimizer` |

```bash
# List what's registered
ta-automl --list-losses
ta-automl --list-searches

# Built-in losses
ta-automl --symbol AMD --loss min_drawdown --trials 100
ta-automl --symbol AMD --loss calmar       --trials 100

# Built-in search strategies
ta-automl --symbol AMD --search-strategy weighted --trials 100   # default — Vizier over weights
ta-automl --symbol AMD --search-strategy automl                  # FLAML AutoML black-box (no SHAP)
ta-automl --symbol AMD --search-strategy shap                    # AutoML + SHAP attributions

# Note: --optimizer / --trials only apply to 'weighted'. 'automl' and 'shap'
# use FLAML's internal time budget. See docs/CUSTOM_SEARCH.md.

# Mix and match — user-supplied loss/search via 'module:fn'
ta-automl --symbol AMD --search-strategy my_search:my_fn --loss my_losses:my_loss
```

**When to use which strategy**:

- **`weighted`** — fastest to interpret. Vizier's per-indicator weights ARE the answer; no SHAP needed. Best when a fixed linear blend is the right model. Respects `--optimizer` and `--trials`.
- **`automl`** — black-box tree model that captures non-linear interactions. Returns global `feature_importances_` for the picked model. Use when you want AutoML quality but don't need per-sample interpretability.
- **`shap`** — `automl` + SHAP attributions. Required when indicators only matter on tail days / regime shifts (those don't show up in linear weights or coarse global importance). Pulls in the `shap` extra.

The Python API also accepts callables directly via
`evaluate_trial(..., loss_fn=my_fn)` and `get_search(my_search_fn)(ctx)`.

---

## Output

**Terminal:** Rich color-coded traffic-light table showing the most recent `lookback` trading days, one column per top indicator, plus a combined signal column. Green = BUY, Red = SELL, Yellow = HOLD.

**`results/{SYMBOL}/results_{start}_{end}.json`:** Full optimization output — best parameters, metrics, survivor list.

**`results/{SYMBOL}/chart_{start}_{end}.html`:** Interactive backtesting chart (only with `--save-html`).

---

## Project structure

```
ta_automl/
├── compat.py               # TA-Lib import guard, runtime patches
├── config.py               # StudyConfig, ScreenConfig dataclasses
├── data/fetcher.py         # yfinance download + parquet cache
├── signals/
│   ├── auto_discover.py    # talib.get_functions() wrapper, param space
│   ├── binarizer.py        # float series → {-1, 0, +1}
│   └── screener.py         # Stage 1 quality filter
├── backtest/strategy.py    # Dynamic backtesting.py Strategy factory
├── optimization/
│   ├── evaluator.py        # params → signals → backtest → metrics
│   ├── study.py            # Vizier in-process runner
│   └── flaml_search.py     # FLAML BlendSearch runner
├── display/traffic_light.py # Rich terminal table
└── main.py                  # Click CLI
```

---

## Known issues

| Issue | Fix |
|-------|-----|
| `No module named 'jax'` (Vizier) | `pip install "jax[cpu]"` or use `--optimizer flaml` |
| `np.bool8` error (bokeh + numpy 2.x) | Already fixed: `numpy<2.0` pinned in pyproject.toml |
| Very few screener survivors | Normal — Vizier uses weights to select; adjust threshold for example to `-99` by `--min-sharpe -99` to pass all (well... that's a bit too much) |
