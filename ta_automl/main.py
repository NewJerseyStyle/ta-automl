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
@click.option("--loss",        default="sharpe",      show_default=True,
              help="Loss function name (see --list-losses) or 'module:fn'")
@click.option("--list-losses", is_flag=True, default=False,
              help="Print all registered loss functions and exit")
@click.option("--search-strategy", default="weighted", show_default=True,
              help="Search strategy: 'weighted' (default), 'shap', or 'module:fn'")
@click.option("--list-searches", is_flag=True, default=False,
              help="Print all registered search strategies and exit")
@click.option("--metric",      default=None,
              type=click.Choice(["sharpe", "return", "winrate"]),
              help="[DEPRECATED] use --loss instead; kept for backwards compat")
@click.option("--p-threshold", default=0.05,          show_default=True, help="Stage-1 p-value cutoff")
@click.option("--min-sharpe",  default=0.10,          show_default=True, help="Stage-1 min |Sharpe|")
@click.option("--no-bonferroni", is_flag=True, default=False, help="Disable Bonferroni correction")
@click.option("--tune-screen", is_flag=True, default=False,
              help="Stage-1 parameter-aware screening: search per-indicator hyperparameters before filtering")
@click.option("--tune-trials", default=8, show_default=True,
              help="Trials per indicator during Stage-1 tuning (only with --tune-screen)")
@click.option("--tune-optimizer", default="vizier", show_default=True,
              type=click.Choice(["vizier", "flaml", "random"]),
              help="Optimizer used for per-indicator tuning (only with --tune-screen)")
@click.option("--tune-metric", default="abs_sharpe", show_default=True,
              type=click.Choice(["abs_sharpe", "sharpe", "neg_p_value"]),
              help="Per-indicator screening score (only with --tune-screen)")
@click.option("--tune-method/--no-tune-method", default=True,
              help="Also search over binarization methods during Stage-1 tuning")
@click.option("--top-n",       default=8,             show_default=True, help="Indicators in traffic light")
@click.option("--lookback",    default=30,            show_default=True, help="Days shown in traffic light")
@click.option("--cash",        default=10_000.0,      show_default=True, help="Starting cash")
@click.option("--commission",  default=0.002,         show_default=True, help="Commission per trade")
@click.option("--train-ratio", default=0.70,          show_default=True, help="Train/test split ratio")
@click.option("--no-short",    is_flag=True, default=False, help="Long-only strategy")
@click.option("--save-html",   is_flag=True, default=False, help="Save backtesting HTML chart")
@click.option("--output-dir",  default="results",    show_default=True, help="Output directory")
@click.option("--cache-dir",   default=".cache",     show_default=True, help="Data cache directory")
@click.option("--plugins",     multiple=True,
              help="Python module(s) or .py files to import before running, "
                   "so user-registered indicators / combiners / losses / searches "
                   "are visible. Can be repeated.")
def cli(
    symbol, start, end, trials, optimizer, loss, list_losses,
    search_strategy, list_searches, metric,
    p_threshold, min_sharpe, no_bonferroni,
    tune_screen, tune_trials, tune_optimizer, tune_metric, tune_method,
    top_n, lookback, cash, commission, train_ratio,
    no_short, save_html, output_dir, cache_dir, plugins,
):
    """AutoML hyperparameter optimizer for TA-Lib technical analysis signals."""
    import ta_automl  # triggers compat patches
    if plugins:
        from ta_automl.sdk.plugins import load_plugins
        load_plugins(plugins)

    from ta_automl.config import ScreenConfig, StudyConfig
    from ta_automl.data.fetcher import fetch_ohlcv
    from ta_automl.display.traffic_light import render_traffic_light
    from ta_automl.optimization.loss import LOSS_REGISTRY, get_loss, list_losses as _list_losses
    from ta_automl.optimization.search import (
        SEARCH_REGISTRY, SearchContext, get_search, list_searches as _list_searches,
    )
    from ta_automl.signals.screener import screen_indicators

    if list_losses:
        click.echo("Registered loss functions:")
        for name in _list_losses():
            doc = (LOSS_REGISTRY[name].__doc__ or "").strip().split("\n")[0]
            click.echo(f"  {name:<22}  {doc}")
        return

    if list_searches:
        click.echo("Registered search strategies:")
        for name in _list_searches():
            doc = (SEARCH_REGISTRY[name].__doc__ or "").strip().split("\n")[0]
            click.echo(f"  {name:<22}  {doc}")
        return

    # Resolve loss: legacy --metric overrides if explicitly set; --loss otherwise
    legacy_map = {"sharpe": "sharpe", "return": "return", "winrate": "winrate"}
    loss_name = legacy_map[metric] if metric else loss
    # Allow 'module:fn' syntax for user-supplied losses
    loss_obj: object = loss_name
    if ":" in loss_name:
        import importlib
        mod_name, fn_name = loss_name.rsplit(":", 1)
        loss_obj = getattr(importlib.import_module(mod_name), fn_name)
    else:
        get_loss(loss_name)  # validate it's registered (raises KeyError if not)

    console = Console()
    console.rule(f"[bold cyan]ta-automl  |  {symbol}  |  {start} -> {end}[/bold cyan]")

    config = StudyConfig(
        symbol=symbol, start=start, end=end,
        trials=trials, loss=loss_name if isinstance(loss_obj, str) else "sharpe",
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
            tune_params=tune_screen,
            tune_trials=tune_trials,
            tune_optimizer=tune_optimizer,
            tune_metric=tune_metric,
            tune_method_choice=tune_method,
        ),
    )

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    console.print("\n[bold]Step 1:[/bold] Fetching OHLCV data …")
    df = fetch_ohlcv(symbol, start, end, cache_dir=config.cache_dir)
    console.print(f"  {len(df)} trading days  ({df.index[0].date()} -> {df.index[-1].date()})")

    # Train / test split
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx]
    df_test  = df.iloc[split_idx:]
    console.print(f"  train: {len(df_train)} days  |  test: {len(df_test)} days")

    # ── 2. Stage 1: screen indicators ─────────────────────────────────────────
    if tune_screen:
        console.print(
            f"\n[bold]Step 2:[/bold] Stage-1 parameter-aware screening "
            f"(tuner={tune_optimizer}, {tune_trials} trials/indicator) …"
        )
    else:
        console.print("\n[bold]Step 2:[/bold] Stage-1 indicator screening (default params) …")
    result = screen_indicators(df, config.screen, verbose=True, return_tuned=True)
    survivors, tuned_map = result if isinstance(result, tuple) else (result, {})

    if not survivors:
        console.print("[bold red]No indicators passed screening. "
                      "Try lowering --p-threshold or --min-sharpe.[/bold red]")
        raise SystemExit(1)

    console.print(f"  {len(survivors)} indicators survived")

    # ── 3. Stage 2: Search strategy dispatch ───────────────────────────────────
    # The 'weighted' strategy is the only one that respects --optimizer and --trials.
    # Other strategies (shap, automl, custom) own their own search loop.
    if search_strategy == "weighted":
        console.print(
            f"\n[bold]Step 3:[/bold] Stage-2 search "
            f"(strategy=weighted, optimizer={optimizer}, {trials} trials) …"
        )
    else:
        console.print(
            f"\n[bold]Step 3:[/bold] Stage-2 search (strategy={search_strategy}) …"
        )
        console.print(
            f"  [dim]note: --optimizer/--trials are ignored by '{search_strategy}'; "
            f"that strategy runs its own internal search loop[/dim]"
        )

    search_ctx = SearchContext(
        df=df, df_test=df_test, survivors=survivors, config=config,
        loss_fn=loss_obj, loss_extra={}, tuned=tuned_map,
    )
    search_callable = get_search(search_strategy)
    result = search_callable(search_ctx)
    best_params  = result.best_params
    best_metrics = result.best_metrics
    signals_df   = result.signals_df
    combined     = result.combined

    console.print(
        f"\n  Loss [bold]{loss_name}[/bold] = "
        f"[bold green]{best_metrics.get('objective', float('nan')):.4f}[/bold green]  "
        f"(sharpe={best_metrics.get('sharpe_ratio', 0):.3f}  "
        f"return={best_metrics.get('total_return', 0):.1f}%  "
        f"max_dd={best_metrics.get('max_drawdown', 0):.1f}%)"
    )

    # ── 4. (signals already reconstructed by the search strategy) ──────────────

    # ── 5. Display traffic light ───────────────────────────────────────────────
    console.print("\n[bold]Step 5:[/bold] Rendering traffic light …")
    render_traffic_light(
        df=df,
        signals_df=signals_df,
        combined=combined,
        best_params=best_params,
        best_metrics=best_metrics,
        surviving_keys=list(signals_df.columns),
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
