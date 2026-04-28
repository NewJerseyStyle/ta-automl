"""Developer tab + Tutorial-mode overlay for the GUI.

Lists everything the user has registered (indicators / combiners / losses /
searches), lets them load a plugins directory at runtime, and exposes a
'validate idea' panel that calls sdk.validate_idea() with whatever they
selected — purely a backtest, no AutoML.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ta_automl.sdk import (
    list_combiners,
    list_indicators,
    list_losses,
    list_searches,
)


def developer_panel() -> html.Div:
    return html.Div(
        [
            html.P(
                "This tab is for users who want to build and test their own "
                "indicators or combination rules — without AutoML deciding for "
                "them. Use the validate-idea form below to backtest a specific "
                "list of indicators with a specific rule.",
                className="text-muted",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Load plugins"),
                    dbc.CardBody(
                        [
                            html.P(
                                "Path to a directory or .py file containing your "
                                "@register_indicator / @register_combiner code. "
                                "(You can also pass --plugins on the CLI.)",
                                className="small text-muted mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="dev-plugin-path",
                                              placeholder="./my_strategies",
                                              type="text"),
                                    dbc.Button("Load", id="dev-load-btn",
                                               color="secondary"),
                                ]
                            ),
                            html.Div(id="dev-load-status", className="small mt-2"),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("What's registered"),
                    dbc.CardBody(html.Div(id="dev-registry-list")),
                ],
                className="mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Validate an idea (no AutoML)"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Indicators (multi-select)"),
                                            dcc.Dropdown(
                                                id="dev-ind-pick",
                                                multi=True,
                                                options=[],
                                                placeholder="Pick custom or TA-Lib indicators",
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Combiner"),
                                            dcc.Dropdown(
                                                id="dev-combiner-pick",
                                                options=[],
                                                placeholder="Default: sum_of_signs",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="g-2",
                            ),
                            html.Div(
                                "💡 Click the (?) icons in the main form to see "
                                "analogies for each ML concept while you teach.",
                                className="alert alert-light small mt-3",
                            ),
                            dbc.Button(
                                "▶ Backtest this idea",
                                id="dev-validate-btn",
                                color="success",
                                className="w-100 mt-2",
                            ),
                            html.Div(id="dev-validate-out", className="mt-3"),
                        ]
                    ),
                ]
            ),
        ],
        className="p-3",
    )


def render_registry_list() -> html.Div:
    """Live snapshot of registered extension points, grouped."""

    def block(title: str, items: list[str]) -> html.Div:
        return html.Div(
            [
                html.Div(title, className="fw-semibold small mt-2"),
                html.Div(
                    [dbc.Badge(n, color="secondary", className="me-1 mb-1")
                     for n in items] or
                    [html.Span("(none)", className="text-muted small")],
                ),
            ]
        )

    return html.Div(
        [
            block("Indicators (custom)", list_indicators()),
            block("Combiners", list_combiners()),
            block("Losses", list_losses()),
            block("Search strategies", list_searches()),
        ]
    )


# ── Tutorial-mode overlay ────────────────────────────────────────────────────
TUTORIAL_STEPS: list[dict[str, str]] = [
    {
        "title": "Step 1 — Pick a stock",
        "body": "We'll start with **AMD**. Liquid US large-caps are the easiest to "
                "study because their price history is clean and gap-free.",
    },
    {
        "title": "Step 2 — Pick a date range",
        "body": "Default 2018→2024 covers a full bull/bear/recovery cycle. "
                "Long enough to be statistically meaningful, recent enough that "
                "today's market regime is represented.",
    },
    {
        "title": "Step 3 — Pick a strategy",
        "body": "**Weighted** is the simplest: every indicator gets a weight and "
                "we vote. **AutoML+SHAP** is the most powerful: a tree model "
                "captures non-linear rules, then SHAP explains each prediction. "
                "Start with Weighted.",
    },
    {
        "title": "Step 4 — Loss function",
        "body": "The loss defines what 'good' means. Sharpe is the default. "
                "Switch to **min_drawdown** if you're risk-averse, or **calmar** "
                "for a balance.",
    },
    {
        "title": "Step 5 — Run and read results",
        "body": "Click ▶ Run. You'll get a Sharpe verdict (🟢/🟡/🔴), an equity "
                "curve vs buy-and-hold, a daily signal heatmap, and a list of "
                "which indicators mattered most. **All metrics come from the "
                "held-out test slice — never the training data.**",
    },
    {
        "title": "Step 6 — Bring your own indicator",
        "body": "Open the **Developer** tab. Run `ta-automl-dev new-indicator "
                "my_idea` in a terminal — it drops a starter file in "
                "`./my_strategies/`. Edit, then load the directory in the "
                "Developer tab. Your indicator appears in the dropdown. "
                "Click 'Backtest this idea' to validate without AutoML.",
    },
]


def tutorial_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="tut-title")),
            dbc.ModalBody(dcc.Markdown(id="tut-body")),
            dbc.ModalFooter(
                [
                    dbc.Button("◀ Back", id="tut-prev", color="secondary",
                               outline=True, className="me-2"),
                    html.Span(id="tut-counter", className="text-muted me-auto"),
                    dbc.Button("Skip tutorial", id="tut-close",
                               color="link", className="me-2"),
                    dbc.Button("Next ▶", id="tut-next", color="primary"),
                ]
            ),
        ],
        id="tut-modal",
        size="lg",
        is_open=False,
        backdrop="static",
    )
