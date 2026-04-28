# Algo-trading workshop — 90 minutes, ta-automl

A teaching plan for live workshops. Six stations, each ~12–15 minutes. The
arc goes from **"what is a backtest"** to **"my own custom strategy validated
honestly."**

Audience assumed: comfortable with Python basics; no required knowledge of
finance, machine learning, or TA-Lib.

---

## 0. Setup (before students arrive)

```bash
git clone <repo>
cd ta-automl
uv venv .venv --python 3.11
uv pip install -e .

# verify
ta-automl --symbol AMD --trials 5      # quick smoke run, ~30 seconds
ta-automl-gui                          # opens at http://127.0.0.1:8050
```

If TA-Lib's C library fails to install, students can still do stations 1–3
(GUI-only); stations 4–6 require it.

---

## Station 1 — What is a strategy? (10 min)

**Concept:** a strategy is a function from price history to a sequence of
buy / sell / hold decisions. The decision sequence + a trade simulator =
performance numbers.

**Live demo:**

1. Open `ta-automl-gui` in the browser.
2. Click **🎓 Tutorial mode** — six bite-sized panels walk through the same
   ideas this station covers.
3. Run with defaults (AMD, 2018–2024, weighted, 50 trials). While it runs,
   explain: TA-Lib gives ~158 indicators (RSI, MACD, BBANDS…), the optimizer
   tries combinations, the backtest scores each one on **data the optimizer
   never saw**.
4. When it finishes, discuss the verdict (🟢/🟡/🔴), the equity curve, and the
   signal heatmap.

**Checkpoint:** every student can answer "Why do we hold out a test slice?"

---

## Station 2 — Indicators are just functions on price (10 min)

**Concept:** RSI, MACD, BBANDS — these aren't magic. They're 5–20 lines of
pandas / numpy. The reason there are ~150 of them is that traders have spent
50 years writing slightly different statistics on the same OHLCV series.

**Exercise:**

```python
import pandas as pd
from ta_automl.data.fetcher import fetch_ohlcv

df = fetch_ohlcv("AMD", "2023-01-01", "2024-12-31")

# A 1-line indicator
sma20 = df["Close"].rolling(20).mean()
above = (df["Close"] > sma20).astype(int)

print("Days price > 20-day SMA:", above.mean())   # ≈ 0.5 in a flat trend
```

**Checkpoint:** students see that a "technical indicator" is just a feature
function. They could have written one themselves.

---

## Station 3 — Validating an idea, no AutoML (15 min)

**Concept:** before letting AutoML decide for us, can we just test our own
idea? Yes — that's `validate_idea`.

**Exercise:**

```python
from ta_automl.sdk import validate_idea

# Hypothesis: simple trend-following — RSI oscillator + ADX trend strength
result = validate_idea(
    symbol="NVDA", start="2020-01-01", end="2024-12-31",
    indicators=["RSI", "ADX"],
    # no combiner → default majority vote
)
print(result.summary())
result.figure.show()
```

Discuss: did it beat buy-and-hold? On NVDA 2020–2024, buy-and-hold returned
~700% — *most* strategies look bad against that. What's the right benchmark?

**Variation to try in pairs:** swap the symbol to a *flat* stock (XOM, KO).
Strategies often look much better when buy-and-hold is mediocre.

**Checkpoint:** students articulate the difference between absolute return
and risk-adjusted return (Sharpe). They have an opinion on whether their
two-indicator strategy "worked."

---

## Station 4 — Build your own indicator (15 min)

**Hands-on:**

```bash
ta-automl-dev new-indicator gap_fade
```

Replace the TODO in `my_strategies/gap_fade.py`:

```python
@register_indicator("gap_fade")
def gap_fade(df, threshold: float = 0.02) -> pd.Series:
    """Bet on overnight gaps closing back to yesterday's close."""
    gap = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[gap >  threshold] = -1   # gapped up → fade short
    sig[gap < -threshold] =  1   # gapped down → fade long
    return sig
```

Validate it:

```python
import my_strategies
from ta_automl.sdk import validate_idea
print(validate_idea(symbol="SPY", start="2020-01-01", end="2024-12-31",
                    indicators=["gap_fade"], allow_short=True).summary())
```

**Checkpoint:** every student has a `my_strategies/` directory with at least
one custom indicator that runs and produces a backtest.

---

## Station 5 — Build your own combiner (rules, not search) (15 min)

**Concept:** AutoML is one way to combine indicators. **Your rule** is another.
On a small set of trusted indicators, hand-crafted rules often *beat* AutoML
because they encode domain knowledge AutoML would have to discover.

**Hands-on:**

```bash
ta-automl-dev new-combiner gap_with_trend_filter
```

```python
@register_combiner("gap_with_trend_filter")
def gap_with_trend_filter(signals, df, ema_period=200) -> pd.Series:
    """Only fade gaps when the long-term trend agrees with the fade direction."""
    long_uptrend = df["Close"] > df["Close"].ewm(span=ema_period).mean()
    gap = signals.get("gap_fade", 0)

    out = pd.Series(0, index=df.index, dtype=int)
    out[(gap ==  1) & long_uptrend]      = 1    # gap-down in uptrend → buy
    out[(gap == -1) & (~long_uptrend)]   = -1   # gap-up in downtrend → short
    return out
```

Validate:

```python
print(validate_idea(symbol="SPY", start="2020-01-01", end="2024-12-31",
                    indicators=["gap_fade"],
                    combiner="gap_with_trend_filter",
                    allow_short=True).summary())
```

Compare to station 4. Did filtering by trend help?

**Checkpoint:** students can articulate when to add complexity (and when
adding the trend filter *hurts* — ask why that's a useful finding too).

---

## Station 6 — Compare your rule to AutoML (15 min)

**Concept:** AutoML (the *weighted* and *AutoML+SHAP* strategies) is "let the
optimizer figure out the rule." Your combiner is "I tell the optimizer the
rule." Same data — which wins?

**Live demo:**

```bash
# AutoML over the SAME indicators
ta-automl --symbol SPY --start 2020-01-01 --end 2024-12-31 \
          --search-strategy weighted --trials 60 \
          --plugins my_strategies/
```

Compare the resulting Sharpe to your hand-built combiner from station 5.

**Discussion:**

- If your combiner won: your domain knowledge was real edge. Common when
  indicators are few and you understand them deeply.
- If AutoML won: weight-tuning across many indicators captured something you
  missed. Common when indicators are many and noisy.
- If they tied: the data isn't strong enough to distinguish. **This is the
  most common outcome and is itself a finding** — most "edges" don't survive
  honest testing.

**Closing point.** This is not a money-printing tool; it's a **disciplined
hypothesis-testing tool**. Most strategies don't work. The ones that do, work
because the trader had an actual insight — and ta-automl let them validate
it without lying to themselves with overfit numbers.

---

## Workshop checklist for instructors

- [ ] Before the workshop, run station 0 setup on the same Python version
      students will use; commit the resulting lockfile if needed.
- [ ] Pre-cache OHLCV: `python -c "from ta_automl.data.fetcher import
      fetch_ohlcv; [fetch_ohlcv(s,'2018-01-01','2024-12-31') for s in
      ['AMD','NVDA','SPY','XOM','KO']]"` — saves 5 minutes of waiting per student.
- [ ] Open the GUI in tutorial mode for station 1. Project the screen.
- [ ] Have a fallback plan if TA-Lib install fails (let those students pair).
- [ ] Send the **DEVELOPER_GUIDE.md** as a take-home reference.

## Optional follow-ups

- **Custom loss:** add a "Sharpe minus 0.05× drawdown" loss
  ([`CUSTOM_LOSS.md`](CUSTOM_LOSS.md)) and re-run station 6.
- **Walk-forward analysis:** chain three rolling 1-year tests instead of one
  big train/test split. Where does the strategy first fail?
- **Paper trading:** export the combined signal to CSV and feed it to a paper
  broker. Watch for 1–3 months. The gap between backtest Sharpe and live
  Sharpe is the most honest education in algo-trading there is.
