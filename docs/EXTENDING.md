# Extending ta-automl — index

ta-automl has four extension points. They share a single import surface:

```python
from ta_automl.sdk import (
    register_indicator,   # NEW in v0.2.0
    register_combiner,    # NEW in v0.2.0
    register_loss,
    register_search,
    validate_idea,        # one-shot helper, no AutoML
)
```

## Documentation

| Doc                                                                  | Audience                                                          |
| -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)                           | Reference: every extension point in one place                     |
| [`TUTORIAL_CUSTOM_INDICATOR.md`](TUTORIAL_CUSTOM_INDICATOR.md)       | 20-minute walkthrough: build & validate a custom indicator + rule |
| [`TUTORIAL_ALGO_TRADING_WORKSHOP.md`](TUTORIAL_ALGO_TRADING_WORKSHOP.md) | 90-minute classroom workshop with 6 stations + exercises          |
| [`CUSTOM_LOSS.md`](CUSTOM_LOSS.md)                                   | Deep dive: replacing the optimizer's objective                    |
| [`CUSTOM_SEARCH.md`](CUSTOM_SEARCH.md)                               | Deep dive: replacing the AutoML loop entirely                     |
| [`PARAMETER_AWARE_SCREENING.md`](PARAMETER_AWARE_SCREENING.md)       | Stage-1 hyperparameter tuning                                     |

## Developer CLI

```bash
ta-automl-dev new-indicator <name>     # scaffold a custom indicator
ta-automl-dev new-combiner  <name>     # scaffold a custom combiner (no-AutoML rule)
ta-automl-dev list                     # show all registered extensions
```

## Loading plugins

Both the main CLI and the GUI accept `--plugins`:

```bash
ta-automl --plugins my_strategies/ --search-strategy my_combiner
ta-automl-gui --plugins my_strategies/
```

Or just `import my_strategies` before calling the Python API.
