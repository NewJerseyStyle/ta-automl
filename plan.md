# ta-automl: Plan

## What it does
Auto-discovers which TA-Lib technical analysis indicators and hyperparameters produce the most profitable trading signals for a given stock, using a two-stage pipeline:

1. **Stage 1 — Screening**: Loop over all ~158 TA-Lib indicators via `talib.get_functions()`, compute each with default params, binarize to {-1, 0, +1}, filter by Mann-Whitney U significance test (Bonferroni-corrected p < 0.05) and per-indicator Sharpe.
2. **Stage 2 — Optimization**: Run Google Vizier (GP-Bandit, in-process) or FLAML (BlendSearch) on the survivors to find optimal indicator hyperparameters + combination weights + threshold. Backtest each trial on held-out data. Render best signals in a traffic-light table.

## Architecture

```
ta_automl/
├── compat.py              # pandas-ta monkey-patch, TA-Lib import guard
├── config.py              # StudyConfig, ScreenConfig dataclasses
├── data/fetcher.py        # yfinance + parquet cache
├── signals/
│   ├── auto_discover.py   # talib.get_functions(), compute_raw(), param_search_space()
│   ├── binarizer.py       # raw float → {-1, 0, +1}
│   └── screener.py        # Stage 1 filter
├── backtest/strategy.py   # Dynamic Strategy factory for backtesting.py
├── optimization/
│   ├── evaluator.py       # params → weighted signals → backtest
│   ├── study.py           # Vizier in-process study runner
│   └── flaml_search.py    # FLAML BlendSearch alternative
├── display/traffic_light.py  # Rich terminal table
└── main.py                # Click CLI
```

## Usage

```bash
# Install (requires Python 3.11, uv)
uv venv .venv --python 3.11
uv pip install -e .

# Default run: AMD, 2018-2024, 100 Vizier trials
ta-automl --symbol AMD

# Custom
ta-automl --symbol NVDA --start 2020-01-01 --end 2024-12-31 --trials 50

# FLAML instead of Vizier
ta-automl --symbol AMD --optimizer flaml --trials 200

# Long-only, save chart
ta-automl --symbol TSLA --no-short --save-html
```

## Key design choices
- Signal combination: weighted sum of survivor signals, threshold ∈ [0.05, 0.8]
- Train/test split: 70/30 by time (no lookahead); screener uses full data (ok: it tests signals, not returns)
- Strategy: long + short by default; `--no-short` for long-only
- Degenerate "never trade" penalty: Sharpe × 0.1 if # trades < 5
- Binarization method (percentile/zscore/trend/price_cross/pattern) is also a Vizier parameter per indicator
