# Customizing the Search: Search Strategies

A **search strategy** decides how `ta-automl` finds the best combination of
indicators. It receives the screened survivor list, the OHLCV split, and a
`SearchContext`, and must return a `SearchResult`.

The default strategy (`weighted`) optimizes a weighted sum over all surviving
indicators with Vizier or FLAML. Some signals matter for **special events**
(volatility spikes, regime changes, gap downs) but contribute weakly to
average revenue — those tend to get near-zero weights in the default search.
The `shap` strategy is built for that case: it trains a CatBoost classifier
over **all** indicator features, then attributes per-feature SHAP scores.
Indicators that drive specific decisions surface even if they don't move the
revenue needle on average.

---

## Built-in strategies

| Name | What it does | `--optimizer` / `--trials` apply? | Needs |
|------|-------------|-----------------------------------|-------|
| `weighted` (default) | Vizier or FLAML BlendSearch over per-indicator weights + binarize methods + threshold. Revenue-driven; you can read winning indicator weights directly from the result. | **yes** — `weighted` is the only strategy that uses these flags | core deps |
| `automl` | FLAML AutoML picks & tunes a tree classifier (lgbm/xgb/rf/...) over all indicator features. Reports the model's built-in `feature_importances_` for ranking. Black-box, no SHAP needed. | no — uses FLAML's internal time budget (`extra={"automl_time_budget_s": N}`) | core deps (FLAML AutoML extra) |
| `shap` | Same training as `automl`, plus SHAP attribution on the held-out test set. Reveals which indicators drive specific predictions, including event-day signals that don't move average revenue. | no — same as `automl` | `pip install -e '.[shap]'` |

**Why three?** `weighted` and `automl`/`shap` answer different questions:

- **`weighted`** finds *the global mixture* that maximizes Sharpe on the test set. The Vizier-selected per-indicator weights ARE the interpretation — no SHAP needed. Best when you believe a fixed linear combination is the right model.
- **`automl`** uses a non-linear tree model (interaction effects, conditional rules). Its `feature_importances_` is a coarse global ranking — useful but doesn't show *when* an indicator mattered.
- **`shap`** is `automl` + per-sample, per-class attributions. Required when an indicator only matters on tail days / regime shifts — those signals appear in SHAP but get washed out of global importance and out of `weighted`'s linear weights.

```bash
# Default revenue-search
ta-automl --symbol AMD --trials 100

# SHAP-based search (captures event-driven signals)
ta-automl --symbol AMD --search-strategy shap --trials 60

# List all registered search strategies
ta-automl --list-searches
```

The two strategies can give qualitatively different "top indicators" — e.g.
`weighted` may surface `RSI`/`EMA` while `shap` may surface `CDLENGULFING`,
`NATR`, `BBANDS__lowerband` (event-flag indicators that matter on tail days).
Run both and compare.

---

## Writing a custom search strategy

The signature is:

```python
def my_search(ctx: SearchContext) -> SearchResult: ...
```

`SearchContext` carries:
- `df`         — full OHLCV DataFrame
- `df_test`    — held-out slice for backtesting
- `survivors`  — list[str] of indicator keys from the screener
- `config`     — full `StudyConfig`
- `loss_fn`    — name or callable for the loss
- `loss_extra` — dict[str, Any] for loss-config plumbing
- `extra`      — dict[str, Any] for arbitrary state you want to receive

`SearchResult` must contain:
- `best_params` (dict)            — your trial parameters; keys ending `__weight` or `__importance` rank in the traffic light
- `best_metrics` (dict)           — at minimum `sharpe_ratio, total_return, win_rate, num_trades, max_drawdown, objective`
- `signals_df` (DataFrame)        — one column per indicator, values in {-1, 0, +1}, index aligned to `ctx.df`
- `combined` (Series)             — final combined buy/sell/hold signal in {-1, 0, +1}
- `importance` (dict[str, float]) — optional per-key importance for downstream display
- `extra` (dict)                  — anything else you want to surface

### Example — register via decorator

```python
# my_search.py
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from ta_automl.optimization.search import (
    SearchContext, SearchResult, build_signals_df, register_search,
)
from ta_automl.backtest.strategy import run_backtest
from ta_automl.optimization.loss import LossContext, get_loss


@register_search("logistic_l1")
def logistic_l1(ctx: SearchContext) -> SearchResult:
    """Sparse linear search: L1-regularized logistic regression picks the
    smallest indicator subset that explains next-day return direction."""
    cfg = ctx.config
    X = build_signals_df(ctx.df, ctx.survivors)
    next_ret = ctx.df["Close"].pct_change().shift(-1)
    y = (next_ret > 0).astype(int).iloc[:-1]
    X = X.iloc[:-1]

    split = int(len(X) * cfg.train_ratio)
    model = LogisticRegressionCV(penalty="l1", solver="saga", cv=3, max_iter=2000)
    model.fit(X.iloc[:split].values, y.iloc[:split].values)

    # signal on the test slice
    edge = model.predict_proba(X.iloc[split:].values)[:, 1] - 0.5
    sig = pd.Series(0, index=X.iloc[split:].index, dtype=int)
    sig[edge >  0.05] = 1
    sig[edge < -0.05] = -1
    combined = pd.Series(0, index=ctx.df.index, dtype=int)
    combined.loc[sig.index] = sig.values

    df_test = ctx.df_test
    bt = run_backtest(df_test, combined.reindex(df_test.index).fillna(0).astype(int),
                      cash=cfg.cash, commission=cfg.commission,
                      allow_short=cfg.allow_short)
    metrics = {k: v for k, v in bt.items() if not k.startswith("_")}
    metrics["objective"] = float(get_loss(ctx.loss_fn or cfg.loss)(metrics, LossContext()))

    coefs = dict(zip(X.columns, model.coef_.ravel()))
    importance = {k: abs(v) for k, v in coefs.items()}
    best_params = {f"{k}__weight": coefs[k] for k in X.columns}
    best_params["combination_threshold"] = 0.05

    return SearchResult(
        best_params=best_params,
        best_metrics=metrics,
        signals_df=X,
        combined=combined,
        importance=importance,
    )
```

Then run via either:

```bash
# After importing my_search somewhere on PYTHONPATH so the decorator fires:
python -c "import my_search; from ta_automl.main import cli; cli()" \
    --symbol AMD --search-strategy logistic_l1

# Or skip the registry with module:fn syntax (no import side-effect needed):
ta-automl --symbol AMD --search-strategy my_search:logistic_l1
```

### Example — Python API (no CLI)

```python
from ta_automl.config import StudyConfig
from ta_automl.data.fetcher import fetch_ohlcv
from ta_automl.signals.screener import screen_indicators
from ta_automl.optimization.search import SearchContext, get_search

cfg = StudyConfig(symbol="AMD", start="2020-01-01", end="2024-12-31", trials=80)
df = fetch_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.cache_dir)
split = int(len(df) * cfg.train_ratio)
df_test = df.iloc[split:]
survivors = screen_indicators(df, cfg.screen, verbose=False)

ctx = SearchContext(
    df=df, df_test=df_test, survivors=survivors, config=cfg,
    loss_fn="min_drawdown",
    extra={"shap_time_budget_s": 90, "shap_threshold": 0.15},
)
result = get_search("shap")(ctx)
print(result.best_metrics)
print("Top SHAP features:", sorted(result.importance, key=result.importance.get, reverse=True)[:8])
```

---

## Combining loss & search

Loss and search are independent dimensions:

```bash
# weighted-search optimizing for low drawdown
ta-automl --search-strategy weighted --loss min_drawdown

# SHAP-search optimizing for Calmar
ta-automl --search-strategy shap --loss calmar

# user search + user loss
ta-automl --search-strategy my_search:my_fn --loss my_losses:my_loss
```

Within a search strategy, call `get_loss(ctx.loss_fn or ctx.config.loss)` to
honour the caller's loss choice.

---

## Where to look in the code

| File | Purpose |
|------|---------|
| `ta_automl/optimization/search.py` | Registry, `SearchContext`, `SearchResult`, built-ins (`weighted`, `shap`) |
| `ta_automl/optimization/loss.py`   | Loss registry (consumed by every strategy) |
| `ta_automl/main.py`                | `--search-strategy` and `--list-searches` flags |
| `docs/CUSTOM_LOSS.md`              | Companion guide for loss customization |

---

## Tips

- **Honour the train/test split**. The screener uses full data (it's testing
  signal quality, not return prediction), but a search strategy that fits a
  model MUST train on `[:split]` and evaluate on `[split:]`. The default
  `cfg.train_ratio` is 0.70.
- **Always set `metrics["objective"]`** — the framework treats it as the value
  the optimizer maximized, so downstream code can compare strategies. Use the
  loss registry to compute it.
- **Keep `signals_df` columns in `{-1, 0, 1}`** so the traffic-light display
  stays readable.
- **Surface a meaningful `importance` dict** — that's what the UI uses to
  pick top-N indicators to display.
