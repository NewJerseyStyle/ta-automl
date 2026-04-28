"""Runtime compatibility patches — imported first by ta_automl/__init__.py."""
import logging
import os
import warnings

# numpy 2.x removed np.bool8; bokeh 2.x / backtesting.py 0.3.3 still uses it.
# Patch before any bokeh/backtesting import.
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Vizier's in-process servicer emits an unconditional absl warning
# ("Python 3.8+ is required in this case.") on every study creation —
# the check is misleading (Python version is fine) and we run hundreds
# of studies during --tune-screen. Silence it.
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "0")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.use_absl_handler()
except Exception:
    pass

# Also clamp the standard-library logger that absl proxies through
for _name in ("absl", "vizier", "vizier._src.service.vizier_client"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import pandas as pd

# pandas-ta 0.3.14b0 calls DataFrame.append() which was removed in pandas 2.0
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, sort=False, **kw):
        other_df = pd.DataFrame([other]) if isinstance(other, dict) else other
        return pd.concat([self, other_df], ignore_index=ignore_index, sort=sort)
    pd.DataFrame.append = _df_append

# Suppress noisy DeprecationWarnings from pandas-ta's numpy usage
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas_ta")
warnings.filterwarnings("ignore", message=".*DataFrame.groupby.*", category=FutureWarning)


def get_ta_backend() -> str:
    """Return 'talib' if the C wrapper is available, else 'pandas_ta'."""
    try:
        import talib  # noqa: F401
        return "talib"
    except ImportError:
        return "pandas_ta"


TALIB_AVAILABLE = get_ta_backend() == "talib"

if not TALIB_AVAILABLE:
    import sys
    print(
        "[compat] WARNING: TA-Lib C library not found. "
        "Falling back to pandas-ta (fewer indicators). "
        "Install TA-Lib via: pip install TA-Lib  "
        "(Windows pre-built wheel: https://github.com/ta-lib/ta-lib-python)",
        file=sys.stderr,
    )
