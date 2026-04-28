# Developer Guide — extending ta-automl

ta-automl has **four extension points**. Each one is a registry: write a function,
decorate it, and the CLI / GUI / one-shot helper all see it.

| Extension point        | What you supply                                  | When to use it                                                     |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------------------------ |
| `@register_indicator`  | `df → pd.Series` (a feature)                     | You have a domain insight TA-Lib doesn't (regime, volume profile…) |
| `@register_combiner`   | `(signals, df) → pd.Series`                      | You want **your rule**, not AutoML, to make the BUY/SELL decision  |
| `@register_loss`       | `(metrics, ctx) → float`                         | You want to optimize for something other than Sharpe               |
| `@register_search`     | `(SearchContext) → SearchResult`                 | You want a different AutoML loop entirely                          |

All four live under one import:

```python
from ta_automl.sdk import (
    register_indicator, register_combiner,
    register_loss, register_search,
    validate_idea,                       # one-shot helper, no AutoML
)
```

---

## The two new ones (v0.2.0): indicator + combiner

These are the ones most users want. Together they answer the question:
**"Does my hand-crafted strategy work, after honest train/test backtesting?"**
No machine-learning search — just your rule, validated.

### Custom indicator

```python
import pandas as pd
from ta_automl.sdk import register_indicator

@register_indicator("mean_reversion_zscore")
def mean_reversion_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """+1 when oversold (price > 1 stdev below mean), -1 when overbought."""
    mean = df["Close"].rolling(period).mean()
    std  = df["Close"].rolling(period).std()
    z = (df["Close"] - mean) / std
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[z < -1] =  1   # oversold → mean revert UP
    sig[z >  1] = -1   # overbought → revert DOWN
    return sig
```

**Acceptable return shapes** (auto-coerced):
- `{-1, 0, +1}` — already a signal
- `{0, 1}` — boolean state, mapped `0 → -1`, `1 → +1`
- `bool` — same as above
- `float` — auto-binarized at 30th / 70th percentiles

### Custom combiner (no AutoML)

```python
import numpy as np
from ta_automl.sdk import register_combiner

@register_combiner("trend_filtered_reversion")
def trend_filtered_reversion(signals, df, ema_period=200):
    """Long-only mean reversion, gated on a 200-day uptrend."""
    in_uptrend = df["Close"] > df["Close"].ewm(span=ema_period).mean()
    rev = signals.get("mean_reversion_zscore", 0)
    out = (rev * in_uptrend.astype(int)).clip(-1, 1)
    return out.astype(int)
```

Once decorated, your combiner is **also a search strategy** of the same name.
That means you can pick it from:

- the CLI: `ta-automl --search-strategy trend_filtered_reversion --plugins my_strategies/`
- the GUI: it shows up in the *Combination strategy* dropdown and the
  *Developer → Validate idea* form.

To opt out of the search bridge, use `@register_combiner("name", expose_to_search=False)`.

---

## One-shot validation (no AutoML, no Stage-1 screening)

```python
from ta_automl.sdk import validate_idea

result = validate_idea(
    symbol="AMD",
    start="2020-01-01", end="2024-12-31",
    indicators=["mean_reversion_zscore", "RSI", "ADX"],
    combiner="trend_filtered_reversion",
    indicator_params={"mean_reversion_zscore": {"period": 30}},
    cash=10_000, commission=0.002, allow_short=False,
    train_ratio=0.70,
)
print(result.summary())
# AMD  Sharpe=0.84  Return=22.3%  MaxDD=-14.1%  Trades=37

result.figure.show()        # interactive Plotly chart
```

`validate_idea` does **only** what tutorial users need:

1. Fetch OHLCV (cached)
2. Compute the indicators you listed (custom or TA-Lib by name)
3. Apply the combiner
4. Backtest on the held-out test slice
5. Return metrics, signals, equity curve, and a Plotly figure

Notably, it **skips** Stage-1 screening and Stage-2 search — your idea is
already specified, so there's nothing to search.

---

## Custom loss (already in v0.1.0, recap)

```python
from ta_automl.sdk import register_loss, LossContext

@register_loss("sharpe_minus_dd_pen")
def sharpe_with_drawdown_penalty(metrics: dict, ctx: LossContext) -> float:
    """Sharpe minus 0.05 × |max drawdown|."""
    return metrics["sharpe_ratio"] - 0.05 * abs(metrics["max_drawdown"])
```

See [`CUSTOM_LOSS.md`](CUSTOM_LOSS.md) for the full guide.

## Custom search strategy (already in v0.1.0, recap)

For when you want to **replace** the AutoML loop with your own. See
[`CUSTOM_SEARCH.md`](CUSTOM_SEARCH.md). Most users should use a combiner instead.

---

## Wiring it all together

### Recommended layout

```
my_strategies/
├── __init__.py            # auto-imports everything below
├── mean_reversion.py      # @register_indicator(s)
├── trend_filter.py        # @register_indicator(s)
└── my_combiners.py        # @register_combiner(s)
```

### Generate a starter

```bash
ta-automl-dev new-indicator my_idea
ta-automl-dev new-combiner  trend_filtered_reversion
ta-automl-dev list                    # show every registered extension
```

The scaffolder writes a working file with TODOs and creates an `__init__.py`
that auto-imports every plugin in the directory.

### Run it

```bash
# CLI
ta-automl --plugins my_strategies/ --search-strategy trend_filtered_reversion

# GUI
ta-automl-gui --plugins my_strategies/
# … then open the Developer tab in the browser
```

### Or skip the CLI entirely

```python
import my_strategies              # registries populate as a side-effect
from ta_automl.sdk import validate_idea
result = validate_idea(symbol="AMD", start="2020-01-01", end="2024-12-31",
                       indicators=["mean_reversion_zscore"],
                       combiner="trend_filtered_reversion")
```

---

## Testing tips

- **Fix a small date range.** A 1-year slice runs in seconds; iterate fast there
  before doing the 7-year run.
- **Eyeball signal density.** A combiner that produces signal on 0.5% of days
  has nothing to backtest. Aim for 5–40%.
- **Compare to buy-and-hold.** `validate_idea`'s figure overlays buy-and-hold;
  if your strategy doesn't beat it on Sharpe, the rule is decoration, not edge.
- **Watch for look-ahead.** Your indicator must only use data up to and
  including bar `t`. The backtester shifts signals by 1 bar before trading,
  but that doesn't save you if your indicator itself peeks ahead.

## See also

- [`TUTORIAL_CUSTOM_INDICATOR.md`](TUTORIAL_CUSTOM_INDICATOR.md) — step-by-step walkthrough
- [`TUTORIAL_ALGO_TRADING_WORKSHOP.md`](TUTORIAL_ALGO_TRADING_WORKSHOP.md) — 90-min workshop with exercises
- [`CUSTOM_LOSS.md`](CUSTOM_LOSS.md), [`CUSTOM_SEARCH.md`](CUSTOM_SEARCH.md) — older extension points
