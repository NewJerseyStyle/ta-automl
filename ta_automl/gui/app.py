"""ta-automl GUI (v0.2.0) — Plotly Dash front end.

Run:
    ta-automl-gui                  # launches at http://127.0.0.1:8050
    ta-automl-gui --port 9000      # custom port
"""
from __future__ import annotations

import argparse
import webbrowser

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from ta_automl.gui import developer, help_text, runner
from ta_automl.sdk import (
    list_combiners,
    list_indicators,
    validate_idea,
)
from ta_automl.sdk.plugins import load_plugins

# ── App init ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="ta-automl",
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server  # for `gunicorn ta_automl.gui.app:server` if desired


# ── Reusable widgets ─────────────────────────────────────────────────────────
def help_icon(key: str) -> dbc.Button:
    """A small (?) button next to a label that opens an explanation modal."""
    return dbc.Button(
        html.I(className="bi bi-question-circle"),
        id={"type": "help-btn", "key": key},
        color="link",
        size="sm",
        className="p-0 ms-1 align-baseline",
        style={"textDecoration": "none", "fontSize": "0.9rem"},
    )


def labeled(label: str, key: str, control) -> dbc.Row:
    """Form row: label + (?) icon + control + short hint."""
    short = help_text.get(key).get("short", "")
    return dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [html.Label(label, className="fw-semibold"), help_icon(key)],
                    className="d-flex align-items-center",
                ),
                width=4,
            ),
            dbc.Col(
                [
                    control,
                    html.Small(short, className="text-muted d-block mt-1"),
                ],
                width=8,
            ),
        ],
        className="mb-3",
    )


def section(title: str, children, n: int) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(f"{n}. {title}", className="mb-0 fw-bold")),
            dbc.CardBody(children),
        ],
        className="mb-4 shadow-sm",
    )


# ── Layout ───────────────────────────────────────────────────────────────────
controls_panel = html.Div(
    [
        section(
            "Pick a stock and time range",
            [
                labeled("Ticker symbol", "symbol",
                        dbc.Input(id="in-symbol", value="AMD", type="text",
                                  placeholder="e.g. AAPL, NVDA, SPY")),
                labeled("Date range", "date_range",
                        dcc.DatePickerRange(
                            id="in-dates",
                            start_date="2018-01-01",
                            end_date="2024-12-31",
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"},
                        )),
                labeled("Train ratio", "train_ratio",
                        dcc.Slider(id="in-train", min=0.5, max=0.9, step=0.05, value=0.7,
                                   marks={0.5: "50%", 0.7: "70%", 0.9: "90%"})),
            ],
            n=1,
        ),
        section(
            "Choose the machine-learning strategy",
            [
                labeled("Combination strategy", "search_strategy",
                        dbc.Select(
                            id="in-search",
                            value="weighted",
                            options=[
                                {"label": "Weighted vote (simple, fast, transparent)",
                                 "value": "weighted"},
                                {"label": "AutoML black-box (flexible, less interpretable)",
                                 "value": "automl"},
                                {"label": "AutoML + SHAP (flexible + per-day explanations)",
                                 "value": "shap"},
                            ],
                        )),
                labeled("Optimizer", "optimizer",
                        dbc.Select(
                            id="in-optimizer",
                            value="flaml",
                            options=[
                                {"label": "FLAML (no extra setup)", "value": "flaml"},
                                {"label": "Vizier (Google GP-Bandit; needs JAX)",
                                 "value": "vizier"},
                            ],
                        )),
                labeled("Aggregator (weighted strategy)", "aggregator",
                        dbc.Select(
                            id="in-aggregator",
                            value="weighted_sum",
                            options=[
                                {"label": "Weighted sum (smooth, default)",
                                 "value": "weighted_sum"},
                                {"label": "Clamped sum (conviction floor; "
                                          "BUY needs real buy-side agreement)",
                                 "value": "clamped_sum"},
                            ],
                        )),
                labeled("Number of trials", "trials",
                        dcc.Slider(id="in-trials", min=10, max=300, step=10, value=50,
                                   marks={10: "10", 50: "50", 100: "100",
                                          200: "200", 300: "300"})),
                labeled("What to maximize (loss function)", "loss",
                        dbc.Select(
                            id="in-loss", value="sharpe",
                            options=[
                                {"label": "Sharpe ratio (return ÷ volatility) — default",
                                 "value": "sharpe"},
                                {"label": "Total return (raw profit; risky)", "value": "return"},
                                {"label": "Calmar ratio (return ÷ worst loss)", "value": "calmar"},
                                {"label": "Min drawdown (most defensive)",
                                 "value": "min_drawdown"},
                                {"label": "Win rate (% of winning trades)", "value": "winrate"},
                            ],
                        )),
            ],
            n=2,
        ),
        section(
            "Fine-tune the simulation (optional)",
            [
                labeled("Starting cash ($)", "cash",
                        dbc.Input(id="in-cash", value=10000, type="number",
                                  min=100, step=100)),
                labeled("Commission per trade", "commission",
                        dbc.Input(id="in-commission", value=0.002, type="number",
                                  step=0.0005, min=0)),
                labeled("Allow short selling", "allow_short",
                        dbc.Switch(id="in-short", value=True, label="Yes, bet on declines too")),
                labeled("Tune each indicator first", "tune_screen",
                        dbc.Switch(id="in-tune", value=False,
                                   label="Slower; usually finds better strategies")),
                labeled("Per-indicator tune trials", "tune_trials",
                        dcc.Slider(id="in-tune-trials", min=2, max=20, step=2, value=8,
                                   marks={2: "2", 8: "8", 20: "20"})),
                labeled("Top indicators to display", "top_n",
                        dcc.Slider(id="in-topn", min=3, max=20, step=1, value=8,
                                   marks={3: "3", 8: "8", 20: "20"})),
                labeled("Days shown in signal table", "lookback",
                        dcc.Slider(id="in-lookback", min=10, max=120, step=10, value=30,
                                   marks={10: "10", 30: "30", 60: "60", 120: "120"})),
            ],
            n=3,
        ),
        dbc.Button("▶ Run analysis", id="btn-run", color="primary", size="lg",
                   className="w-100 mb-3"),
        html.Div(id="run-warnings"),
    ],
    style={"position": "sticky", "top": "0"},
)


main_results_panel = html.Div(id="results-panel", children=[
    dcc.Markdown(help_text.INTRO_MARKDOWN, className="p-3"),
])

results_panel = dbc.Tabs(
    [
        dbc.Tab(main_results_panel, label="AutoML run", tab_id="tab-main"),
        dbc.Tab(developer.developer_panel(), label="Developer", tab_id="tab-dev"),
    ],
    id="results-tabs",
    active_tab="tab-main",
)


app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(html.Div(
                [
                    html.H2("📈 ta-automl", className="d-inline-block me-3 mb-0"),
                    dbc.Badge("v0.2.0 GUI", color="info"),
                    dbc.Button("🎓 Tutorial mode", id="tut-open",
                               color="link", className="ms-3"),
                    html.Div("AutoML for technical-analysis trading strategies — for everyone.",
                             className="text-muted small mt-1"),
                ],
                className="py-3 border-bottom mb-4",
            )),
        ),
        dbc.Row(
            [
                dbc.Col(controls_panel, lg=5, md=12),
                dbc.Col(results_panel, lg=7, md=12),
            ],
        ),
        # Modal for help text
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="help-title")),
                dbc.ModalBody(id="help-body"),
            ],
            id="help-modal",
            size="lg",
            scrollable=True,
            is_open=False,
        ),
        developer.tutorial_modal(),
        # Background state
        dcc.Store(id="run-id", data=None),
        dcc.Store(id="tut-step", data=0),
        dcc.Interval(id="poll", interval=1500, disabled=True),
    ],
    fluid=True,
    className="pb-5",
)


# ── Help modal ───────────────────────────────────────────────────────────────
@app.callback(
    Output("help-modal", "is_open"),
    Output("help-title", "children"),
    Output("help-body", "children"),
    Input({"type": "help-btn", "key": ALL}, "n_clicks"),
    State("help-modal", "is_open"),
    prevent_initial_call=True,
)
def show_help(_clicks, is_open):
    trigger = ctx.triggered_id
    if not trigger or not any(_clicks or []):
        return no_update, no_update, no_update
    key = trigger["key"] if isinstance(trigger, dict) else trigger
    info = help_text.get(key)
    body = [
        dcc.Markdown(info["long"] or "_No long description._"),
        html.Hr(),
        html.H6("🔍 Analogy"),
        html.P(info["analogy"] or "_No analogy._", className="fst-italic"),
        html.H6("💡 Suggestion"),
        html.P(info["tip"] or "_No tip._", className="text-success"),
    ]
    return True, key.replace("_", " ").title(), body


# ── Run pipeline ─────────────────────────────────────────────────────────────
@app.callback(
    Output("run-id", "data"),
    Output("poll", "disabled"),
    Output("btn-run", "disabled"),
    Output("run-warnings", "children"),
    Input("btn-run", "n_clicks"),
    State("in-symbol", "value"),
    State("in-dates", "start_date"),
    State("in-dates", "end_date"),
    State("in-train", "value"),
    State("in-search", "value"),
    State("in-optimizer", "value"),
    State("in-aggregator", "value"),
    State("in-trials", "value"),
    State("in-loss", "value"),
    State("in-cash", "value"),
    State("in-commission", "value"),
    State("in-short", "value"),
    State("in-tune", "value"),
    State("in-tune-trials", "value"),
    State("in-topn", "value"),
    State("in-lookback", "value"),
    prevent_initial_call=True,
)
def kick_off(n, symbol, start, end, train_ratio, search, optimizer, aggregator,
             trials, loss, cash, commission, allow_short, tune, tune_trials,
             top_n, lookback):
    if not symbol or not start or not end:
        return no_update, no_update, False, dbc.Alert(
            "Please fill in symbol and date range.", color="warning")

    params = dict(
        symbol=symbol, start=str(start)[:10], end=str(end)[:10],
        train_ratio=train_ratio, search_strategy=search, optimizer=optimizer,
        aggregator=aggregator,
        trials=trials, loss=loss, cash=cash, commission=commission,
        allow_short=allow_short, tune_screen=tune, tune_trials=tune_trials,
        tune_optimizer="random", top_n=top_n, lookback=lookback,
    )
    run_id = runner.start_run(params)
    return run_id, False, True, dbc.Alert(
        f"Started run {run_id}. This usually takes 1–10 minutes.",
        color="info", className="py-2",
    )


# ── Poll for progress and render results ─────────────────────────────────────
@app.callback(
    Output("results-panel", "children"),
    Output("poll", "disabled", allow_duplicate=True),
    Output("btn-run", "disabled", allow_duplicate=True),
    Input("poll", "n_intervals"),
    State("run-id", "data"),
    prevent_initial_call=True,
)
def poll(_n, run_id):
    if not run_id:
        return no_update, True, False
    st = runner.get_run(run_id)
    if st is None:
        return dbc.Alert("Run not found.", color="danger"), True, False

    if st.status in ("pending", "running"):
        return _render_progress(st), False, True
    if st.status == "error":
        return _render_error(st), True, False
    return _render_results(st), True, False


# ── Renderers ────────────────────────────────────────────────────────────────
def _render_progress(st: runner.RunState):
    return html.Div(
        [
            html.H5(f"⏳ {st.step or 'Working…'}", className="mb-3"),
            dbc.Progress(value=int(st.progress * 100), striped=True, animated=True,
                         className="mb-3", style={"height": "18px"}),
            html.Div(
                [html.Div(line, className="font-monospace small text-muted")
                 for line in st.log[-25:]],
                style={"maxHeight": "400px", "overflowY": "auto"},
                className="border rounded p-2 bg-light",
            ),
        ],
        className="p-3",
    )


def _render_error(st: runner.RunState):
    return html.Div(
        [
            dbc.Alert([html.H5("Run failed"), html.P(st.error)], color="danger"),
            html.Pre("\n".join(st.log[-40:]),
                     className="small bg-light p-2 border rounded",
                     style={"maxHeight": "400px", "overflowY": "auto"}),
        ],
        className="p-3",
    )


def _render_results(st: runner.RunState):
    r = st.result
    m = r["metrics"]

    metric_cards = dbc.Row(
        [
            _kpi("Sharpe (test)", f"{m['sharpe']:.2f}",
                 _color_for(m["sharpe"], good=1.0, ok=0.5)),
            _kpi("Total return", f"{m['total_return']:.1f}%",
                 _color_for(m["total_return"], good=20, ok=0)),
            _kpi("Max drawdown", f"{m['max_drawdown']:.1f}%",
                 _color_for(-abs(m["max_drawdown"]), good=-15, ok=-30)),
            _kpi("Win rate", f"{m['win_rate']*100:.1f}%" if m["win_rate"] <= 1
                 else f"{m['win_rate']:.1f}%", "secondary"),
            _kpi("Trades", f"{m['n_trades']}", "secondary"),
            _kpi("Indicators kept", f"{r['n_survivors']}", "secondary"),
        ],
        className="mb-4 g-2",
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(_equity_chart(r), label="Equity curve",
                    tab_id="tab-equity"),
            dbc.Tab(_signal_heatmap(r), label="Signal table",
                    tab_id="tab-signals"),
            dbc.Tab(_importance_chart(r), label="Top indicators",
                    tab_id="tab-importance"),
            dbc.Tab(_price_chart(r), label="Price + signals",
                    tab_id="tab-price"),
            dbc.Tab(_params_view(r), label="Best parameters",
                    tab_id="tab-params"),
        ],
        active_tab="tab-equity",
    )

    return html.Div(
        [
            html.H4(f"Results — {r['symbol']}", className="mb-3"),
            dbc.Alert(_interpretation(m), color="light",
                      className="border small"),
            metric_cards,
            tabs,
        ],
        className="p-3",
    )


def _kpi(label, value, color):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, className="text-muted small"),
                    html.Div(value, className=f"h4 mb-0 text-{color}"),
                ]
            ),
            className="text-center shadow-sm",
        ),
        xs=6, md=4, lg=2,
    )


def _color_for(val, good, ok):
    if val >= good:
        return "success"
    if val >= ok:
        return "warning"
    return "danger"


def _interpretation(m):
    s = m["sharpe"]
    if s >= 1.5:
        verdict = "🟢 **Strong result.** A Sharpe above 1.5 on held-out data is rare."
    elif s >= 1.0:
        verdict = "🟢 **Solid.** Sharpe > 1.0 is the usual bar for 'this could be real'."
    elif s >= 0.5:
        verdict = "🟡 **Marginal.** Could be signal, could be lucky. Try a wider date range."
    elif s >= 0:
        verdict = "🟠 **Weak.** Slight edge, probably not worth trading after taxes & slippage."
    else:
        verdict = "🔴 **Negative.** This combination lost money on held-out data."
    return dcc.Markdown(
        f"{verdict}\n\n"
        f"_Reminder: these are backtest numbers on the **held-out test slice**. "
        f"They are honest in the sense that the optimizer never saw this data, "
        f"but they are still historical. Markets adapt._"
    )


def _equity_chart(r):
    eq = r["equity"]
    bh = r["buy_hold"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["x"], y=eq["y"], name="Strategy",
                             line=dict(color="#2e86de", width=2.5)))
    fig.add_trace(go.Scatter(x=bh["x"], y=bh["y"], name="Buy & Hold",
                             line=dict(color="#aaa", width=2, dash="dot")))
    fig.update_layout(
        title="Strategy vs. Buy & Hold (test set, simulated)",
        yaxis_title="Account value ($)",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return dcc.Graph(figure=fig)


def _signal_heatmap(r):
    sig = r["signals_recent"]
    if not sig["columns"] or not sig["index"]:
        return html.Div("No signals available.", className="p-3 text-muted")
    fig = go.Figure(
        data=go.Heatmap(
            z=sig["values"],
            x=sig["columns"],
            y=sig["index"],
            colorscale=[[0.0, "#e74c3c"], [0.5, "#ecf0f1"], [1.0, "#27ae60"]],
            zmin=-1, zmax=1,
            showscale=True,
            colorbar=dict(tickvals=[-1, 0, 1], ticktext=["SELL", "HOLD", "BUY"]),
        )
    )
    fig.update_layout(
        title="Daily signal per top indicator (red = SELL, green = BUY)",
        margin=dict(l=10, r=10, t=50, b=80),
        xaxis=dict(tickangle=-45),
        height=520,
    )
    return dcc.Graph(figure=fig)


def _importance_chart(r):
    items = sorted(r["importance"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
    if not items:
        return html.Div("No importance data.", className="p-3 text-muted")
    keys = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=keys, orientation="h",
                           marker_color=colors))
    fig.update_layout(
        title="Indicator influence on the combined signal",
        xaxis_title="Weight / importance",
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(360, 22 * len(keys) + 80),
    )
    return dcc.Graph(figure=fig)


def _price_chart(r):
    px = r["price"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px["x"], y=px["y"], name="Close",
                             line=dict(color="#34495e", width=1.5)))
    # Overlay BUY/SELL markers from combined_recent
    sig = r["combined_recent"]
    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    if sig["x"]:
        # Index price by date string for marker placement
        px_lookup = dict(zip(px["x"], px["y"]))
        for t, v in zip(sig["x"], sig["y"]):
            y = px_lookup.get(t)
            if y is None or v is None:
                continue
            if v > 0:
                buys_x.append(t); buys_y.append(y)
            elif v < 0:
                sells_x.append(t); sells_y.append(y)
    fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode="markers",
                             name="BUY", marker=dict(color="#27ae60", size=10,
                                                     symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells_x, y=sells_y, mode="markers",
                             name="SELL", marker=dict(color="#e74c3c", size=10,
                                                      symbol="triangle-down")))
    fig.update_layout(
        title="Price with recent strategy signals",
        yaxis_title="Close price",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return dcc.Graph(figure=fig)


def _params_view(r):
    rows = [html.Tr([html.Td(k, className="font-monospace small"),
                     html.Td(str(v), className="font-monospace small")])
            for k, v in sorted(r["best_params"].items())]
    return html.Div(
        [
            html.P("These are the exact parameter values the optimizer chose. "
                   "You can re-run the CLI with the same settings to reproduce.",
                   className="text-muted small"),
            dbc.Table([html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")])),
                       html.Tbody(rows)],
                      bordered=True, striped=True, size="sm"),
        ],
        className="p-2",
    )


# ── Developer tab callbacks ─────────────────────────────────────────────────
@app.callback(
    Output("dev-load-status", "children"),
    Output("dev-registry-list", "children"),
    Output("dev-ind-pick", "options"),
    Output("dev-combiner-pick", "options"),
    Input("dev-load-btn", "n_clicks"),
    Input("results-tabs", "active_tab"),
    State("dev-plugin-path", "value"),
    prevent_initial_call=False,
)
def refresh_registry(_clicks, active_tab, path):
    msg = ""
    if ctx.triggered_id == "dev-load-btn" and path:
        try:
            loaded = load_plugins([path])
            msg = dbc.Alert(f"Loaded: {', '.join(loaded)}",
                            color="success", className="py-2 small")
        except Exception as exc:  # noqa: BLE001
            msg = dbc.Alert(f"Load failed: {type(exc).__name__}: {exc}",
                            color="danger", className="py-2 small")
    ind_opts = [{"label": f"[custom] {n}", "value": n} for n in list_indicators()]
    # A handful of friendly TA-Lib examples beginners recognize
    ind_opts += [{"label": f"[TA-Lib] {n}", "value": n}
                 for n in ("RSI", "MACD__macdhist", "ADX", "ATR", "OBV",
                           "EMA", "BBANDS__upperband", "STOCH__slowk")]
    com_opts = [{"label": n, "value": n} for n in list_combiners()]
    return msg, developer.render_registry_list(), ind_opts, com_opts


@app.callback(
    Output("dev-validate-out", "children"),
    Input("dev-validate-btn", "n_clicks"),
    State("dev-ind-pick", "value"),
    State("dev-combiner-pick", "value"),
    State("in-symbol", "value"),
    State("in-dates", "start_date"),
    State("in-dates", "end_date"),
    State("in-cash", "value"),
    State("in-commission", "value"),
    State("in-short", "value"),
    State("in-train", "value"),
    prevent_initial_call=True,
)
def run_validate_idea(_n, indicators, combiner, symbol, start, end,
                      cash, commission, allow_short, train_ratio):
    if not indicators:
        return dbc.Alert("Pick at least one indicator.", color="warning")
    try:
        res = validate_idea(
            symbol=symbol,
            start=str(start)[:10], end=str(end)[:10],
            indicators=indicators,
            combiner=combiner or None,
            cash=float(cash or 10000),
            commission=float(commission or 0.002),
            allow_short=bool(allow_short),
            train_ratio=float(train_ratio or 0.7),
        )
    except Exception as exc:  # noqa: BLE001
        return dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")
    m = res.metrics
    return html.Div(
        [
            dbc.Alert(res.summary(), color="info", className="py-2"),
            dcc.Graph(figure=res.figure) if res.figure is not None else html.Div(),
            html.Details(
                [
                    html.Summary("Raw metrics", className="text-muted small"),
                    html.Pre(str(m), className="small bg-light p-2 border rounded"),
                ],
                className="mt-2",
            ),
        ]
    )


# ── Tutorial modal callbacks ────────────────────────────────────────────────
@app.callback(
    Output("tut-modal", "is_open"),
    Output("tut-step", "data"),
    Output("tut-title", "children"),
    Output("tut-body", "children"),
    Output("tut-counter", "children"),
    Input("tut-open", "n_clicks"),
    Input("tut-next", "n_clicks"),
    Input("tut-prev", "n_clicks"),
    Input("tut-close", "n_clicks"),
    State("tut-step", "data"),
    State("tut-modal", "is_open"),
    prevent_initial_call=True,
)
def tutorial_nav(_o, _nx, _pv, _cl, step, is_open):
    trig = ctx.triggered_id
    steps = developer.TUTORIAL_STEPS
    n = len(steps)
    step = step or 0
    if trig == "tut-open":
        step = 0
        is_open = True
    elif trig == "tut-close":
        return False, 0, no_update, no_update, no_update
    elif trig == "tut-next":
        if step >= n - 1:
            return False, 0, no_update, no_update, no_update
        step += 1
    elif trig == "tut-prev":
        step = max(0, step - 1)
    s = steps[step]
    return is_open, step, s["title"], s["body"], f"{step + 1} / {n}"


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ta-automl GUI (v0.2.0)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser.")
    parser.add_argument("--plugins", action="append", default=[],
                        help="Plugin module / file / dir to import on launch. "
                             "Can be repeated.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.plugins:
        load_plugins(args.plugins)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"\nta-automl GUI running at {url}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
