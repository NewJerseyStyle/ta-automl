# Tutorial — building your first custom indicator

**Goal:** in 20 minutes, code a custom indicator, combine it with TA-Lib's RSI
using your own rule, and backtest it on AMD 2020–2024.

You'll need: this repo installed (`uv pip install -e .`) and a terminal.

---

## 1. Scaffold

```bash
ta-automl-dev new-indicator volume_surge
```

This creates `./my_strategies/volume_surge.py` with a working stub and an
`__init__.py` that auto-imports every plugin in the directory.

## 2. Write the indicator

Replace the TODO in `my_strategies/volume_surge.py` with:

```python
import pandas as pd
from ta_automl.sdk import register_indicator

@register_indicator("volume_surge")
def volume_surge(df: pd.DataFrame, lookback: int = 20, multiplier: float = 1.5) -> pd.Series:
    """+1 on days where volume is multiplier× above its rolling mean AND price closes up."""
    avg_vol = df["Volume"].rolling(lookback).mean()
    surge   = df["Volume"] > multiplier * avg_vol
    up_day  = df["Close"] > df["Open"]
    return (surge & up_day).astype(int)        # 0/1; SDK auto-maps to -1/+1
```

**What is this indicator saying?** "When unusually heavy volume coincides with
an up day, that's institutional accumulation — bullish." Whether that's *true*
on AMD 2020–2024 is what we're about to test.

## 3. Validate it on its own

```python
# scratch.py
import my_strategies                                    # populates registries
from ta_automl.sdk import validate_idea

result = validate_idea(
    symbol="AMD",
    start="2020-01-01", end="2024-12-31",
    indicators=["volume_surge"],                        # only our new one
    # no combiner specified → default 'sum_of_signs'
    cash=10_000, commission=0.002, allow_short=False,
)
print(result.summary())
result.figure.show()
```

```bash
python scratch.py
# AMD  Sharpe=0.31  Return=8.2%  MaxDD=-22.4%  Trades=14
```

A weak signal alone — but it might *combine* well with momentum.

## 4. Combine it with RSI using your own rule

```bash
ta-automl-dev new-combiner volume_confirmed_rsi
```

Edit `my_strategies/volume_confirmed_rsi.py`:

```python
import pandas as pd
from ta_automl.sdk import register_combiner

@register_combiner("volume_confirmed_rsi")
def volume_confirmed_rsi(signals: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """BUY when RSI is oversold AND we've seen a recent volume surge.

    'Recent' = at any point in the last 5 days.
    """
    rsi = signals.get("RSI", 0)                    # already binarized to {-1,0,+1}
    surge = signals.get("volume_surge", 0)
    surge_recent = surge.rolling(5).max().fillna(0)

    out = pd.Series(0, index=df.index, dtype=int)
    out[(rsi == 1) & (surge_recent == 1)] = 1      # oversold + recent surge → BUY
    out[rsi == -1] = -1                            # overbought → SELL regardless
    return out
```

## 5. Validate the combination

```python
result = validate_idea(
    symbol="AMD",
    start="2020-01-01", end="2024-12-31",
    indicators=["volume_surge", "RSI"],            # custom + TA-Lib by name
    combiner="volume_confirmed_rsi",
    allow_short=False,
)
print(result.summary())
```

If Sharpe jumped from 0.31 → 1.0+, your gating rule added value. If it
*dropped*, the volume signal was muddying RSI more than helping. Either way,
**you now have an honest read** — not a hunch.

## 6. Run the same combination from the GUI

```bash
ta-automl-gui --plugins my_strategies/
```

In the browser:
1. Open the **Developer** tab.
2. Pick `volume_surge` and `RSI` in the indicator dropdown.
3. Pick `volume_confirmed_rsi` as the combiner.
4. Click **Backtest this idea** — same numbers, same chart, no code.

That's the full loop: idea → file → registry → validation → visualization.

---

## What "good" looks like

| Sharpe (test slice) | Verdict                                                |
| ------------------- | ------------------------------------------------------ |
| > 1.5               | Genuinely interesting, worth deeper study              |
| 1.0 – 1.5           | Solid; sensitive to slippage assumptions               |
| 0.5 – 1.0           | Marginal; could be luck                                |
| 0 – 0.5             | Probably noise                                         |
| < 0                 | Worse than nothing — your rule has it backwards        |

## Common pitfalls

- **Used `df['Close'].shift(-1)` somewhere.** That's tomorrow's close — look-ahead.
- **Computed an indicator on the full series, then split.** The rolling stats
  near the split absorbed test-set info. Always compute on the full series and
  let the backtester only *trade* on the test slice (which is what
  `validate_idea` does — but if you run `signals` outside it, watch for this).
- **Indicator never fires.** Loosen the threshold. A signal active on 0.5% of
  days has nothing to test.
- **Indicator fires every day.** The combiner is doing nothing — your "signal"
  is just always-on.

## Next steps

- Read [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for the full SDK reference.
- Try the workshop in [`TUTORIAL_ALGO_TRADING_WORKSHOP.md`](TUTORIAL_ALGO_TRADING_WORKSHOP.md).
- Write a custom loss in [`CUSTOM_LOSS.md`](CUSTOM_LOSS.md) (e.g., penalize >10 trades/month).
