"""Rich terminal traffic-light display of signal history."""
from __future__ import annotations

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_CELL = {
    1:  (" BUY  ", "bold white on green"),
    0:  (" HOLD ", "bold black on yellow"),
    -1: (" SELL ", "bold white on red"),
}


def render_traffic_light(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    combined: pd.Series,
    best_params: dict,
    best_metrics: dict,
    surviving_keys: list[str],
    top_n: int = 8,
    lookback: int = 30,
    console: Console | None = None,
) -> None:
    """
    Print a color-coded table of the most recent `lookback` trading days.

    signals_df — DataFrame with one column per survivor indicator (values {-1,0,+1})
    combined   — final combined signal Series
    best_params — optimal trial parameters from Vizier/FLAML
    best_metrics — metrics from the best trial
    """
    if console is None:
        console = Console()

    # Pick top-N indicators by weight or SHAP importance from best_params
    def _importance_of(k: str) -> float:
        return float(
            best_params.get(f"{k}__weight",
            best_params.get(f"{k}__importance", 0.0))
        )
    weights = {
        key: _importance_of(key)
        for key in surviving_keys
        if key in signals_df.columns
    }
    top_keys = sorted(weights, key=lambda k: weights[k], reverse=True)[:top_n]

    # Slice to lookback window, newest first
    recent = signals_df.loc[signals_df.index.isin(df.index)].tail(lookback).iloc[::-1]
    combined_recent = combined.reindex(recent.index)

    # Build Rich table
    tbl = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"[bold]Signal Traffic Light[/bold]",
        title_style="bold white",
    )
    tbl.add_column("Date", style="dim", width=12, no_wrap=True)
    for key in top_keys:
        w = weights.get(key, 0.0)
        short = key.replace("__", "\n")
        tbl.add_column(f"{short}\n[dim]w={w:.2f}[/dim]", justify="center", width=10)
    tbl.add_column("[bold]COMBINED[/bold]", justify="center", width=10, style="bold")

    for date, row in recent.iterrows():
        cells = [str(date.date())]
        for key in top_keys:
            val = int(row.get(key, 0))
            txt, style = _CELL.get(val, (" ???? ", "dim"))
            cells.append(f"[{style}]{txt}[/{style}]")
        cval = int(combined_recent.get(date, 0))
        ctxt, cstyle = _CELL.get(cval, (" ???? ", "dim"))
        cells.append(f"[{cstyle}]{ctxt}[/{cstyle}]")
        tbl.add_row(*cells)

    console.print()
    console.print(tbl)

    # Summary panel
    sharpe = best_metrics.get("sharpe_ratio", 0.0)
    ret    = best_metrics.get("total_return", 0.0)
    wr     = best_metrics.get("win_rate", 0.0)
    trades = int(best_metrics.get("num_trades", 0))
    dd     = best_metrics.get("max_drawdown", 0.0)

    summary_lines = [
        f"[bold green]Sharpe Ratio :[/bold green]  {sharpe:.3f}",
        f"[bold green]Total Return :[/bold green]  {ret:.1f}%",
        f"[bold yellow]Win Rate     :[/bold yellow]  {wr:.1f}%",
        f"[bold yellow]# Trades     :[/bold yellow]  {trades}",
        f"[bold red]Max Drawdown :[/bold red]  {dd:.1f}%",
        "",
        "[bold cyan]Top indicators by weight:[/bold cyan]",
    ]
    for key in top_keys:
        w = weights.get(key, 0.0)
        method_idx = int(best_params.get(f"{key}__binarize", 0))
        from ta_automl.signals.binarizer import INT_TO_METHOD
        method = INT_TO_METHOD.get(method_idx, "?")
        summary_lines.append(f"  {key:<30} weight={w:.3f}  binarize={method}")

    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold]Optimization Results[/bold]",
        border_style="cyan",
    ))
    console.print()
