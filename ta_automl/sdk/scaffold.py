"""`ta-automl new-indicator` and `new-combiner` template generators.

Drops a working starter file into ./my_strategies/ that the user can edit.
"""
from __future__ import annotations

from pathlib import Path

import click

INDICATOR_TEMPLATE = '''"""Custom indicator: {name}.

Drop this file's directory on PYTHONPATH and import it before running
ta-automl, or pass it via --plugins my_strategies/{name}.py.
"""
from __future__ import annotations

import pandas as pd

from ta_automl.sdk import register_indicator


@register_indicator("{name}")
def {name}(df: pd.DataFrame, period: int = 20, threshold: float = 0.5) -> pd.Series:
    """One-line description of what {name} measures.

    Args:
        df: OHLCV DataFrame (columns Open, High, Low, Close, Volume).
        period: rolling window length.
        threshold: cutoff for the BUY/SELL decision.

    Returns:
        Series aligned to df.index. Acceptable shapes:
          • {{-1, 0, +1}}  – already a signal
          • {{0, 1}}       – boolean state, mapped 0→-1, 1→+1
          • float / any   – auto-binarized (top 30% → +1, bottom 30% → -1)
    """
    # TODO: replace with your real logic.
    rolling_mean = df["Close"].rolling(period).mean()
    distance = (df["Close"] - rolling_mean) / rolling_mean

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[distance > threshold] = 1     # price well above MA → uptrend → BUY
    signal[distance < -threshold] = -1   # price well below MA → downtrend → SELL
    return signal
'''


COMBINER_TEMPLATE = '''"""Custom combiner: {name}.

A combiner takes the per-indicator signal matrix and returns one signal.
This bypasses AutoML — your rule IS the strategy. Backtesting validates it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ta_automl.sdk import register_combiner


@register_combiner("{name}")
def {name}(signals: pd.DataFrame, df: pd.DataFrame, min_agreement: int = 2) -> pd.Series:
    """Combine indicator signals into one BUY/SELL/HOLD series.

    Args:
        signals: DataFrame, one column per indicator, values in {{-1, 0, +1}}.
        df: OHLCV (handy if you want price-based gating).
        min_agreement: how many indicators must agree before we act.

    Returns:
        Series of {{-1, 0, +1}} aligned to df.index.
    """
    buys  = (signals ==  1).sum(axis=1)
    sells = (signals == -1).sum(axis=1)

    out = pd.Series(0, index=signals.index, dtype=int)
    out[buys  - sells >=  min_agreement] =  1
    out[sells - buys  >=  min_agreement] = -1
    return out
'''


@click.group(name="ta-automl-dev")
def dev_cli() -> None:
    """Developer helpers for building indicators / combiners / losses."""


@dev_cli.command("new-indicator")
@click.argument("name")
@click.option("--out-dir", default="my_strategies", show_default=True,
              help="Directory to write the file into.")
@click.option("--force", is_flag=True, help="Overwrite if exists.")
def new_indicator(name: str, out_dir: str, force: bool) -> None:
    """Create a starter custom-indicator file."""
    _scaffold(name, out_dir, force, INDICATOR_TEMPLATE, suffix="indicator")


@dev_cli.command("new-combiner")
@click.argument("name")
@click.option("--out-dir", default="my_strategies", show_default=True)
@click.option("--force", is_flag=True)
def new_combiner(name: str, out_dir: str, force: bool) -> None:
    """Create a starter custom-combiner file (no AutoML — your rule is the strategy)."""
    _scaffold(name, out_dir, force, COMBINER_TEMPLATE, suffix="combiner")


@dev_cli.command("list")
def list_extensions() -> None:
    """Show every registered indicator / combiner / loss / search."""
    # Force-load any user plugins on PYTHONPATH first
    from ta_automl.sdk import (
        list_combiners, list_indicators, list_losses, list_searches,
    )
    click.echo(click.style("Indicators:", bold=True))
    for n in list_indicators() or ["(none — register with @register_indicator)"]:
        click.echo(f"  • {n}")
    click.echo(click.style("\nCombiners:", bold=True))
    for n in list_combiners() or ["(none)"]:
        click.echo(f"  • {n}")
    click.echo(click.style("\nLosses:", bold=True))
    for n in list_losses():
        click.echo(f"  • {n}")
    click.echo(click.style("\nSearch strategies:", bold=True))
    for n in list_searches():
        click.echo(f"  • {n}")


def _scaffold(name: str, out_dir: str, force: bool, template: str, *, suffix: str) -> None:
    if not name.isidentifier():
        raise click.UsageError(f"{name!r} is not a valid Python identifier.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.py"
    if target.exists() and not force:
        raise click.UsageError(f"{target} exists. Pass --force to overwrite.")
    target.write_text(template.format(name=name), encoding="utf-8")

    init = out / "__init__.py"
    if not init.exists():
        init.write_text(
            "# Auto-import every plugin in this directory so registries populate.\n"
            "from pathlib import Path as _P\n"
            "import importlib as _il\n"
            "for _f in _P(__file__).parent.glob('*.py'):\n"
            "    if _f.stem not in {'__init__'}:\n"
            "        _il.import_module(f'{__name__}.{_f.stem}')\n",
            encoding="utf-8",
        )
    click.secho(f"✓ Created {target}", fg="green")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Open {target} and edit the TODO.")
    click.echo(f"  2. Import the package once before running ta-automl, e.g.:")
    click.echo(f"       python -c 'import {out_dir}; "
               f"import ta_automl.main as m; m.cli()' --search-strategy {name}")
    click.echo(f"     …or set PYTHONPATH={out_dir.split('/')[0]} and import "
               f"in your own driver script.")


if __name__ == "__main__":
    dev_cli()
