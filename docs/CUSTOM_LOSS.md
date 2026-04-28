# Customizing the Search: Loss Functions

`ta-automl`'s optimizer maximizes a **loss function** (named `objective`) over
the hyperparameter space. By default it maximizes Sharpe ratio, but you can
plug in any function `(metrics, context) -> float`.

This doc shows how to:
1. Use a built-in loss
2. Write your own and register it
3. Use it from the CLI or from Python

---

## 1. Built-in losses

Run `ta-automl --list-losses` to see the registry. As of writing:

| Name | What it maximizes |
|------|-------------------|
| `sharpe` (default) | Sharpe ratio |
| `return` | Total return % |
| `winrate` | Win rate % |
| `min_drawdown` | `-|max_drawdown|`  (i.e. minimizes drawdown) + tiny return tie-breaker |
| `calmar` | Calmar ratio = `total_return / |max_drawdown|` |
| `sharpe_dd_penalty` | `sharpe - 0.05 * |max_drawdown|` (tunable via `dd_weight`) |

```bash
# Find combinations that minimize drawdown
ta-automl --symbol AMD --loss min_drawdown --trials 100

# Calmar — balance return vs drawdown
ta-automl --symbol AMD --loss calmar --trials 100
```

---

## 2. Writing a custom loss

A loss is a function with this signature:

```python
def my_loss(metrics: dict[str, float], context: LossContext) -> float:
    ...
```

`metrics` is a dict produced by `run_backtest`, with keys:
- `sharpe_ratio`  — annualized Sharpe
- `total_return`  — % return over the test window
- `win_rate`      — % winning trades
- `num_trades`    — integer count
- `max_drawdown`  — non-positive % (e.g. -25.4 means 25.4% peak-to-trough drop)

`context` (a `LossContext` dataclass) carries:
- `params`     — the trial hyperparameter dict (read it for advanced losses)
- `min_trades` — soft minimum trade count (5 by default)
- `extra`      — any user-supplied dict you can plumb through

The optimizer **maximizes** the returned value. To minimize, return the
negative.

### Example A — register via decorator

Create a file anywhere on `PYTHONPATH`, e.g. `my_losses.py`:

```python
# my_losses.py
from ta_automl.optimization.loss import register_loss

@register_loss("low_dd_high_winrate")
def low_dd_high_winrate(metrics, context):
    """Reward win-rate, penalize drawdown, require at least 10 trades."""
    if metrics["num_trades"] < 10:
        return -999.0
    return metrics["win_rate"] - 0.5 * abs(metrics["max_drawdown"])
```

Then register it before invoking the CLI by importing the module:

```bash
# Easiest: import as a side-effect via PYTHONSTARTUP, or wrap a small launcher:
python -c "import my_losses; from ta_automl.main import cli; cli()" \
    --symbol AMD --loss low_dd_high_winrate --trials 100
```

### Example B — pass `module:fn` syntax to the CLI

```python
# my_losses.py
def risk_adjusted(metrics, context):
    sharpe = metrics["sharpe_ratio"]
    dd     = abs(metrics["max_drawdown"])
    trades = metrics["num_trades"]
    return sharpe * min(trades / 20, 1.0) - 0.02 * dd
```

```bash
ta-automl --symbol AMD --loss my_losses:risk_adjusted --trials 100
```

The `module:function` form bypasses the registry — `ta-automl` imports the
module and resolves the attribute directly. The module must be importable
from `sys.path`.

### Example C — Python API (no CLI)

```python
from ta_automl.config import StudyConfig
from ta_automl.data.fetcher import fetch_ohlcv
from ta_automl.signals.screener import screen_indicators
from ta_automl.optimization.evaluator import evaluate_trial
from ta_automl.optimization.flaml_search import run_flaml_study

def downside_pain(metrics, context):
    """Maximize return per unit of drawdown, penalty if too few trades."""
    ret = metrics["total_return"]
    dd  = abs(metrics["max_drawdown"]) + 1e-3
    pen = 1.0 if metrics["num_trades"] >= context.min_trades else 0.2
    return pen * ret / dd

cfg = StudyConfig(symbol="AMD", start="2020-01-01", end="2024-12-31", trials=80)
df = fetch_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.cache_dir)
split = int(len(df) * cfg.train_ratio)
df_test = df.iloc[split:]
survivors = screen_indicators(df, cfg.screen, verbose=True)

def eval_fn(params):
    return evaluate_trial(
        params, df, df_test, survivors, cfg,
        loss_fn=downside_pain,            # ← plug in your callable
        loss_extra={"my_setting": True},  # ← read it via context.extra
    )

best_params, best_metrics = run_flaml_study(
    survivors, eval_fn, n_trials=cfg.trials, metric="objective",
)
print(best_metrics)
```

---

## 3. Multi-objective tip

True multi-objective optimization requires returning a tuple to Vizier. For a
simpler approach, use a **scalarized** loss that combines several metrics:

```python
@register_loss("scalarized")
def scalarized(metrics, context):
    w_ret    = context.extra.get("w_return",   1.0)
    w_sharpe = context.extra.get("w_sharpe",   2.0)
    w_dd     = context.extra.get("w_dd",       0.10)
    return (
        w_ret    * (metrics["total_return"] / 100.0)
        + w_sharpe * metrics["sharpe_ratio"]
        - w_dd     * abs(metrics["max_drawdown"])
    )
```

Pass weights via the Python API's `loss_extra={...}` argument.

---

## 4. Anti-patterns to avoid

- **Don't return NaN or Inf** — the optimizer may treat them as best-so-far. Clamp
  pathological cases to a large negative finite value (e.g. `-999.0`).
- **Don't reward zero-trade configs** — they look great on every metric but
  aren't strategies. Use `metrics["num_trades"]` as a gate.
- **Don't depend on absolute scale** — both Vizier (GP-Bandit) and FLAML
  (BlendSearch) work better when the loss has roughly unit variance. Divide
  drawdown percentages by 100 if you mix them with Sharpe.
- **Read `context.params` carefully** if you must — the dict has the raw
  trial parameters, but mutating it has no effect on the optimizer.

---

## 5. Where to look in the code

| File | Purpose |
|------|---------|
| `ta_automl/optimization/loss.py` | Registry, `LossContext`, all built-in losses |
| `ta_automl/optimization/evaluator.py` | Where the loss is applied per trial |
| `ta_automl/optimization/study.py` | Vizier — reads `metrics["objective"]` |
| `ta_automl/optimization/flaml_search.py` | FLAML — reads `metrics["objective"]` |
| `ta_automl/main.py` | `--loss` and `--list-losses` CLI flags |
