from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScreenConfig:
    """Controls Stage-1 indicator quality filter.

    The screener is a pure data-quality pass — it removes degenerate indicators.
    Vizier handles the actual signal selection via weights.
    Individual TA indicators rarely produce significant next-day returns alone
    (efficient market hypothesis), so p-value / Sharpe filters are disabled by default.

    Enable p_filter=True only for exploratory analysis.
    """
    min_signal_frac: float = 0.02    # must be non-zero on at least 2% of days
    max_nan_frac: float = 0.40       # reject if >40% of output is NaN (too much warmup)
    p_filter: bool = False           # enable optional Mann-Whitney p-value filter
    p_threshold: float = 0.20        # p-value cutoff (only used when p_filter=True)
    bonferroni: bool = False         # Bonferroni correction (only used when p_filter=True)
    min_sharpe: float = -2.0         # reject only catastrophically loss-making indicators


@dataclass
class StudyConfig:
    """Full pipeline configuration."""
    symbol: str = "AMD"
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    trials: int = 100
    metric: str = "sharpe"           # legacy alias for `loss`
    loss: str = "sharpe"             # name of registered loss fn (see optimization.loss)
    cash: float = 10_000.0
    commission: float = 0.002
    train_ratio: float = 0.70        # fraction of days for training (screener + no backtest)
    allow_short: bool = True
    optimizer: str = "vizier"        # vizier | flaml
    top_n: int = 8                   # how many indicators to show in traffic light
    lookback: int = 30               # how many recent days in traffic light table
    save_html: bool = False          # save backtesting.py interactive HTML chart
    output_dir: Path = field(default_factory=lambda: Path("results"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
