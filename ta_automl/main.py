"""CLI entry point for ta-automl."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@click.command()
@click.option("--symbol",      default="AMD",        show_default=True, help="Ticker symbol")
@click.option("--start",       default="2018-01-01", show_default=True, help="Start date YYYY-MM-DD")
@click.option("--end",         default="2024-12-31", show_default=True, help="End date YYYY-MM-DD")
@click.option("--trials",      default=100,           show_default=True, help="Optimizer trial count")
@click.option("--optimizer",   default="vizier",      show_default=True,
              type=click.Choice(["vizier", "flaml"]), help="Optimizer backend")
@click.option("--metric",      default="sharpe",      show_default=True,
              type=click.Choice(["sharpe", "return", "winrate"]), help="Objective metric")
@click.option("--p-threshold", default=0.05,          show_default=True, help="Stage-1 p-value cutoff")
@click.option("--min-sharpe",  default=0.10,          show_default=True, help="Stage-1 min |Sharpe|")
@click.option("--no-bonferroni", is_flag=True, default=False, help="Disable Bonferroni correction")
@click.option("--top-n",       default=8,             show_default=True, help="Indicators in traffic light")
@click.option("--lookback",    default=30,            show_default=True, help="Days shown in traffic light")
@click.option("--cash",        default=10_000.0,      show_default=True, help="Starting cash")
@click.option("--commission",  default=0.002,         show_default=True, help="Commission per trade")
@click.option("--train-ratio", default=0.70,          show_default=True, help="Train/test split ratio")
@click.option("--no-short",    is_flag=True, default=False, help="Long-only strategy")
@click.option("--save-html",   is_flag=True, default=False, help="Save backtesting HTML chart")
@click.option("--output-dir",  default="results",    show_default=True, help="Output directory")
@click.option("--cache-dir",   default=".cache",     show_default=True, help="Data cache directory")
def cli(
    symbol, start, end, trials, optimizer, metric,
    p_threshold, min_sharpe, no_bonferroni,
    top_n, lookback, cash, commission, train_ratio,
    no_short, save_html, output_dir, cache_dir,
):
    """AutoML hyperparameter optimizer for TA-Lib technical analysis signals."""
    import ta_automl  # triggers compat patches

    from ta_automl.config import ScreenConfig, StudyConfig
    from ta_automl.data.fetcher import fetch_ohlcv
    from ta_automl.display.traffic_light import render_traffic_light
    from ta_automl.optimization.evaluator import evaluate_trial, build_vizier_param_space
    from ta_automl.signals.auto_discover import compute_raw, default_params
    from ta_automl.signals.binarizer import INT_TO_METHOD, binarize
    from ta_automl.signals.screener import screen_indicators

    metric_key = {"sharpe": "sharpe_ratio", "return": "total_return", "winrate": "win_rate"}[metric]

    console = Console()
    console.rule(f"[bold cyan]ta-automl  ·  {symbol}  ·  {start} → {end}[/bold cyan]")

    config = StudyConfig(
        symbol=symbol, start=start, end=end,
        trials=trials, metric=metric_key,
        cash=cash, commission=commission,
        train_ratio=train_ratio,
        allow_short=not no_short,
        optimizer=optimizer,
        top_n=top_n, lookback=lookback,
        save_html=save_html,
        output_dir=Path(output_dir),
        cache_dir=Path(cache_dir),
        screen=ScreenConfig(
            p_threshold=p_threshold,
            min_sharpe=min_sharpe,
            bonferroni=not no_bonferroni,
        ),
    )

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    console.print("\n[bold]Step 1:[/bold] Fetching OHLCV data …")
    df = fetch_ohlcv(symbol, start, end, cache_dir=config.cache_dir)
    console.print(f"  {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")

    # Train / test split
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx]
    df_test  = df.iloc[split_idx:]
    console.print(f"  train: {len(df_train)} days  |  test: {len(df_test)} days")

    # ── 2. Stage 1: screen indicators ─────────────────────────────────────────
    console.print("\n[bold]Step 2:[/bold] Stage-1 indicator screening …")
    survivors = screen_indicators(df, config.screen, verbose=True)

    if not survivors:
        console.print("[bold red]No indicators passed screening. "
                      "Try lowering --p-threshold or --min-sharpe.[/bold red]")
        raise SystemExit(1)

    console.print(f"  → {len(survivors)} indicators survived")

    # ── 3. Stage 2: Vizier / FLAML optimization ────────────────────────────────
    console.print(f"\n[bold]Step 3:[/bold] Stage-2 optimization ({optimizer}, {trials} trials) …")

    def eval_fn(params: dict) -> dict:
        return evaluate_trial(params, df, df_test, survivors, config)

    if optimizer == "vizier":
        from ta_automl.optimization.study import run_vizier_study
        best_params, best_metrics = run_vizier_study(
            survivors, eval_fn, trials,
            study_name=f"{symbol}_{start}_{end}",
            metric=metric_key,
        )
    else:
        from ta_automl.optimization.flaml_search import run_flaml_study
        best_params, best_metrics = run_flaml_study(
            survivors, eval_fn, trials, metric=metric_key,
        )

    console.print(f"\n  Best {metric_key}: [bold green]{best_metrics.get(metric_key, 'N/A'):.4f}[/bold green]")

    # ── 4. Reconstruct full signal history with best params ────────────────────
    console.print("\n[bold]Step 4:[/bold] Reconstructing signals with optimal parameters …")

    threshold = float(best_params.get("combination_threshold", 0.3))
    signals_dict: dict[str, pd.Series] = {}
    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for key in survivors:
        weight = float(best_params.get(f"{key}__weight", 0.0))
        if weight < 0.05:
            continue
        base = key.split("__")[0]
        from ta_automl.optimization.evaluator import _indicator_base
        from ta_automl.signals.auto_discover import param_search_space
        space = param_search_space(base)
        ind_params = {}
        for p_name, (lo, hi, ptype) in space.items():
            full = f"{key}__{p_name}"
            if full in best_params:
                ind_params[p_name] = ptype(best_params[full])

        try:
            raw_dict = compute_raw(base, df, ind_params or None)
        except Exception:
            continue

        raw_series = raw_dict.get(key)
        if raw_series is None:
            raw_series = next(iter(raw_dict.values()))
        method_idx = int(best_params.get(f"{key}__binarize", 0))
        method = INT_TO_METHOD.get(method_idx, "percentile")
        sig = binarize(key, raw_series, df, method=method)

        signals_dict[key] = sig
        weighted_sum += weight * sig
        total_weight += weight

    if total_weight > 0:
        weighted_sum /= total_weight

    combined = pd.Series(0, index=df.index, dtype=int)
    combined[weighted_sum >  threshold] = 1
    combined[weighted_sum < -threshold] = -1

    signals_df = pd.DataFrame(signals_dict, index=df.index)

    # ── 5. Display traffic light ───────────────────────────────────────────────
    console.print("\n[bold]Step 5:[/bold] Rendering traffic light …")
    render_traffic_light(
        df=df,
        signals_df=signals_df,
        combined=combined,
        best_params=best_params,
        best_metrics=best_metrics,
        surviving_keys=survivors,
        top_n=top_n,
        lookback=lookback,
        console=console,
    )

    # ── 6. Save results ────────────────────────────────────────────────────────
    out_dir = config.output_dir / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "optimizer": optimizer,
        "trials": trials,
        "survivors": survivors,
        "best_params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                        for k, v in best_params.items()},
        "best_metrics": best_metrics,
    }
    results_file = out_dir / f"results_{start}_{end}.json"
    results_file.write_text(json.dumps(results, indent=2))
    console.print(f"[dim]Results saved to {results_file}[/dim]")

    # Optional: save backtesting HTML chart (re-run backtest on test set)
    if save_html:
        from ta_automl.backtest.strategy import run_backtest, make_strategy_class
        combined_test = combined.reindex(df_test.index).fillna(0).astype(int)
        bt_result = run_backtest(df_test, combined_test, cash=cash,
                                  commission=commission, allow_short=not no_short)
        html_file = out_dir / f"chart_{start}_{end}.html"
        bt_result["_bt"].plot(filename=str(html_file), open_browser=False)
        console.print(f"[dim]Chart saved to {html_file}[/dim]")

    console.rule("[bold cyan]Done[/bold cyan]")


if __name__ == "__main__":
    cli()
