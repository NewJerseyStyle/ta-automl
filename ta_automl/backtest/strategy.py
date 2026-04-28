"""Dynamic backtesting.py Strategy factory and backtest runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta_automl.compat  # ensure np.bool8 shim is applied before bokeh/backtesting loads
from backtesting import Backtest, Strategy


def make_strategy_class(signal_array: np.ndarray, allow_short: bool = True) -> type:
    """
    Create a backtesting.py Strategy subclass with a pre-computed signal baked in
    via closure (required because backtesting.py reads params as class attributes).

    signal_array: int array of {-1, 0, +1} aligned to the df passed to Backtest().
    """
    _sig = signal_array  # captured by closure

    def init(self):
        self._signal = self.I(lambda: _sig, name="combined_signal", overlay=False)

    if allow_short:
        def next(self):
            sig = int(self._signal[-1])
            pos = self.position.size
            if sig == 1 and pos <= 0:
                if pos < 0:
                    self.position.close()
                self.buy()
            elif sig == -1 and pos >= 0:
                if pos > 0:
                    self.position.close()
                self.sell()
    else:
        def next(self):
            sig = int(self._signal[-1])
            pos = self.position.size
            if sig == 1 and pos == 0:
                self.buy()
            elif sig != 1 and pos > 0:
                self.position.close()

    return type("DynamicTAStrategy", (Strategy,), {"init": init, "next": next})


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cash: float = 10_000.0,
    commission: float = 0.002,
    allow_short: bool = True,
) -> dict:
    """
    Run backtesting.py on df using the pre-computed signal series.

    df and signal must share the same DatetimeIndex (same rows).
    Returns a dict with sharpe_ratio, total_return, win_rate, num_trades, max_drawdown.
    """
    # Align signal to df index
    sig_aligned = signal.reindex(df.index).fillna(0).astype(int)
    sig_arr = sig_aligned.values

    StratClass = make_strategy_class(sig_arr, allow_short=allow_short)

    bt = Backtest(
        df,
        StratClass,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = bt.run()

    return {
        "sharpe_ratio":  float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        "total_return":  float(stats.get("Return [%]", 0.0) or 0.0),
        "win_rate":      float(stats.get("Win Rate [%]", 0.0) or 0.0),
        "num_trades":    int(stats.get("# Trades", 0) or 0),
        "max_drawdown":  float(stats.get("Max. Drawdown [%]", 0.0) or 0.0),
        "_bt":           bt,
        "_stats":        stats,
    }
