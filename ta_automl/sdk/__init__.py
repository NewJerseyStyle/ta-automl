"""Public SDK for ta-automl extensions.

This is the single import surface for tutorial users and indicator/strategy
developers. The four extension points are:

    register_indicator(name)   — your own indicator function (df -> Series)
    register_combiner(name)    — your own rule combining indicators -> signal
    register_loss(name)        — your own scoring function (re-exported)
    register_search(name)      — your own AutoML search loop (re-exported)

Plus the one-shot helper:

    validate_idea(...)         — fetch data, compute, backtest, return result

Typical usage (custom indicator + custom combiner, NO AutoML):

    from ta_automl.sdk import (
        register_indicator, register_combiner, validate_idea,
    )

    @register_indicator("price_above_ema")
    def price_above_ema(df, period=50):
        ema = df["Close"].ewm(span=period).mean()
        return (df["Close"] > ema).astype(int)        # boolean -> 0/1

    @register_combiner("my_rules")
    def my_rules(signals, df):
        # signals is a DataFrame with one column per registered indicator
        return signals["price_above_ema"].replace({0: -1, 1: 1})

    result = validate_idea(
        symbol="AMD", start="2020-01-01", end="2024-12-31",
        indicators=["price_above_ema"], combiner="my_rules",
        indicator_params={"price_above_ema": {"period": 50}},
    )
    print(result.metrics)            # sharpe, return, drawdown, ...
    result.figure.show()             # plotly equity curve
"""
from __future__ import annotations

from ta_automl.sdk.indicators import (
    INDICATOR_REGISTRY,
    IndicatorFn,
    get_indicator,
    list_indicators,
    register_indicator,
)
from ta_automl.sdk.combiners import (
    COMBINER_REGISTRY,
    CombinerFn,
    get_combiner,
    list_combiners,
    register_combiner,
)
from ta_automl.sdk.validate import IdeaResult, validate_idea

# Re-export the existing extension points so tutorial users only need one import
from ta_automl.optimization.loss import (  # noqa: F401
    LOSS_REGISTRY,
    LossContext,
    list_losses,
    register_loss,
)
from ta_automl.optimization.search import (  # noqa: F401
    SEARCH_REGISTRY,
    SearchContext,
    SearchResult,
    list_searches,
    register_search,
)

__all__ = [
    # indicator extension
    "register_indicator", "get_indicator", "list_indicators",
    "INDICATOR_REGISTRY", "IndicatorFn",
    # combiner extension
    "register_combiner", "get_combiner", "list_combiners",
    "COMBINER_REGISTRY", "CombinerFn",
    # loss / search re-exports
    "register_loss", "list_losses", "LOSS_REGISTRY", "LossContext",
    "register_search", "list_searches", "SEARCH_REGISTRY",
    "SearchContext", "SearchResult",
    # one-shot helper
    "validate_idea", "IdeaResult",
]
