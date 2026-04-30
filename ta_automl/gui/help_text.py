"""Plain-English explanations + analogies for every option in the GUI.

The goal: a curious user who has never heard of Sharpe ratio, Vizier, or SHAP
should still feel confident clicking buttons. Every entry has:
  - short:   one-line tooltip / hint shown inline
  - long:    multi-paragraph explanation shown when the (?) is clicked
  - analogy: a concrete real-world comparison
  - tip:     a "pick this if ..." recommendation
"""
from __future__ import annotations

HELP: dict[str, dict[str, str]] = {
    # ── Stock & dates ────────────────────────────────────────────────────────
    "symbol": {
        "short": "The company / ETF you want to study (Yahoo Finance ticker).",
        "long": (
            "A *ticker symbol* is the short code stock exchanges use to identify "
            "a security: AAPL = Apple, TSLA = Tesla, SPY = the S&P 500 ETF. "
            "We download daily Open / High / Low / Close / Volume bars for this "
            "symbol from Yahoo Finance.\n\n"
            "Liquid US large-caps (AAPL, MSFT, NVDA, AMD, SPY, QQQ) work best — "
            "they have clean, gap-free history. Avoid penny stocks, recent IPOs, "
            "or symbols with lots of corporate actions; the patterns we fit are "
            "noisier there."
        ),
        "analogy": (
            "Think of it like choosing which patient to study before running a "
            "medical trial. A long, well-documented patient history (a liquid "
            "stock) gives more reliable findings than a brand-new patient with "
            "two months of records (a fresh IPO)."
        ),
        "tip": "Start with AMD, NVDA, or SPY if you're unsure.",
    },
    "date_range": {
        "short": "How far back to look. Longer history = more data, but old patterns may not hold.",
        "long": (
            "We split your history into *train* (older) and *test* (newer). "
            "Indicators are tuned on train, then scored on test — this checks "
            "they actually generalize to data they didn't see during fitting.\n\n"
            "Too short (< 1 year) and the optimizer overfits noise. Too long "
            "(> 10 years) and ancient market regimes (different volatility, "
            "different Fed policy) bias the result. **5–7 years is the sweet "
            "spot** for daily data."
        ),
        "analogy": (
            "Like training a chess engine. Train it on 100 games and it memorizes "
            "those games but can't play new ones. Train it on 10,000 games over "
            "30 years and it has learned styles that no longer exist. A focused "
            "recent corpus is what you actually want."
        ),
        "tip": "Default 2018-01-01 → 2024-12-31 covers two full bull/bear cycles.",
    },
    "train_ratio": {
        "short": "Fraction of days used to train. The rest is held back to honestly score the result.",
        "long": (
            "If train_ratio = 0.70, the first 70% of your history is used to "
            "tune indicators and search for good parameters. The final 30% is "
            "*held back completely* — never seen during training — and is the "
            "data we report Sharpe / return on.\n\n"
            "This is the gold-standard guard against *overfitting*: a strategy "
            "that looks great on training data but fails on new data."
        ),
        "analogy": (
            "Like a final exam. Students study from the textbook (training set) "
            "but the exam draws from questions they haven't seen (test set). "
            "Letting them grade themselves on the textbook would be meaningless."
        ),
        "tip": "Leave at 0.70 unless you have very little data.",
    },

    # ── Strategy / search method ─────────────────────────────────────────────
    "search_strategy": {
        "short": "How indicators are combined into one buy/sell signal.",
        "long": (
            "Stage 1 finds ~150 candidate technical indicators that aren't "
            "garbage. Stage 2 then needs to **combine** them into one final "
            "signal — and there are three philosophies:\n\n"
            "1. **Weighted (default)** — assign each indicator a weight; the "
            "combined signal is a weighted vote. Simple, transparent, fast.\n"
            "2. **AutoML** — let a machine-learning algorithm (gradient-boosted "
            "trees) figure out *non-linear* rules: 'BUY when RSI < 30 AND ADX > "
            "25 BUT NOT during high VIX'. More flexible, harder to interpret.\n"
            "3. **AutoML + SHAP** — same as AutoML, but adds an explanation "
            "layer that tells you *which* indicators drove each prediction.\n\n"
            "Most users should start with **Weighted**."
        ),
        "analogy": (
            "Imagine you're picking a restaurant by polling 10 friends.\n"
            "  • Weighted: each friend gets a score (foodie = 3 votes, picky "
            "eater = 0.5 votes); average their picks.\n"
            "  • AutoML: hire a chef-detective who learns 'when Alice and Bob "
            "agree but Carol disagrees, trust Alice'. Better picks, but you "
            "can't easily explain why.\n"
            "  • AutoML + SHAP: the same chef-detective, but now they hand you "
            "a receipt explaining whose vote mattered for each meal."
        ),
        "tip": "Start with Weighted. Try AutoML+SHAP only after you understand the basics.",
    },
    "optimizer": {
        "short": "The search algorithm that explores parameter combinations.",
        "long": (
            "Once we know what to optimize, we need an algorithm that picks "
            "*which combinations to try* out of trillions of possibilities:\n\n"
            "• **Vizier (Google)** — uses a Gaussian Process to model 'good "
            "regions' of the search space; great when each trial is expensive. "
            "Requires JAX. This is the same engine Google uses internally.\n\n"
            "• **FLAML (Microsoft)** — BlendSearch heuristic, no extra deps, "
            "almost as good. Recommended if Vizier complains about JAX."
        ),
        "analogy": (
            "Looking for the highest peak in a foggy mountain range.\n"
            "  • Vizier is a smart hiker building a 3-D map as they go: every "
            "step refines its guess of where the summit is.\n"
            "  • FLAML is an experienced hiker using rule-of-thumb shortcuts: "
            "less mathematically pure, but already on the mountain in seconds."
        ),
        "tip": "Vizier if installed, otherwise FLAML — quality is similar in practice.",
    },
    "aggregator": {
        "short": "How the weighted strategy turns per-indicator votes into one BUY/SELL decision.",
        "long": (
            "After the optimizer learns a weight for each indicator, those "
            "weighted signals still need to be collapsed into a single trade "
            "decision each day. Two rules are available:\n\n"
            "• **Weighted sum** *(default)* — compute Σ wᵢ·sᵢ, then BUY when "
            "this exceeds a learned threshold (and the mirror for SELL). "
            "Smooth: small weight changes give small Sharpe changes, which is "
            "what the GP-based optimizer (Vizier) likes. Best when you have "
            "many indicators and you trust the optimizer to balance them.\n\n"
            "• **Clamped sum** *(conviction floor)* — tally long-side weight "
            "(only +1 votes) and short-side weight (only −1 votes) separately. "
            "BUY only when the long-side share clears a learned floor AND beats "
            "short-side share. SELL is symmetric. Encodes the rule: 'a BUY "
            "conclusion must come from real buy-side agreement, not just the "
            "absence of sell votes.' The optimizer also tunes the long/short "
            "conviction floors. Surface is steppier (Sharpe can jump as a "
            "threshold tips one indicator in/out), so it converges a little "
            "slower — but is more robust when many indicators sit at 0 most "
            "days and you want to avoid trading on near-zero net consensus."
        ),
        "analogy": (
            "Imagine a panel of advisors voting BUY / HOLD / SELL.\n"
            "  • **Weighted sum** averages their votes by trust level. If the "
            "average leans positive, you BUY. Quiet advisors (HOLD) drag the "
            "average toward zero — that's fine, sometimes you want a soft "
            "consensus.\n"
            "  • **Clamped sum** says: 'I'll only BUY if at least, say, 30% of "
            "advisor weight is *actively saying BUY* — and they outvote the "
            "actively-bearish ones.' Silent advisors don't help. You're "
            "demanding real conviction before pulling the trigger."
        ),
        "tip": (
            "Use weighted sum (default) for most setups. Switch to clamped sum "
            "when (a) most indicators are 0 most days and you don't want a "
            "thin majority of small-weight votes to fire trades, or (b) you "
            "want a strategy whose decisions are easy to explain as "
            "'enough indicators agreed to BUY *and* not many disagreed.'"
        ),
    },
    "trials": {
        "short": "How many parameter combinations to try. More = better result, slower.",
        "long": (
            "Each trial = one full backtest with a different parameter set. "
            "Diminishing returns kick in fast: 30 trials usually finds 80% of "
            "what 300 trials would.\n\n"
            "  • 20–50: quick exploratory run (~1–3 min)\n"
            "  • 100: default, balanced (~3–10 min)\n"
            "  • 300+: serious tuning, only worth it on a real GPU/CPU"
        ),
        "analogy": (
            "Like rolling dice to find the best loaded version. After 20 rolls "
            "you've got a sense; after 100 you're sure; after 1000 you're "
            "wasting your evening."
        ),
        "tip": "Use 30 for a quick test, 100 for serious work.",
    },

    # ── Loss function ────────────────────────────────────────────────────────
    "loss": {
        "short": "The number the optimizer is trying to maximize. Defines what 'good' means.",
        "long": (
            "Different losses encode different definitions of a 'good' "
            "strategy:\n\n"
            "• **sharpe** — return per unit of volatility. The standard metric. "
            "A strategy with Sharpe > 1.0 is solid; > 2.0 is excellent.\n"
            "• **return** — total profit. Dangerous: rewards reckless bets.\n"
            "• **calmar** — return / max drawdown. Punishes big losses harshly.\n"
            "• **min_drawdown** — minimize worst peak-to-trough loss. Good for "
            "the risk-averse.\n"
            "• **winrate** — % of profitable trades (ignores trade size).\n\n"
            "Different losses produce *different strategies* on the same data."
        ),
        "analogy": (
            "Like grading a delivery driver. If you grade only on packages-per-"
            "hour (return), they speed and crash. If you grade on packages-per-"
            "hour ÷ accidents (Sharpe), they balance speed and safety. The grade "
            "you choose changes their behavior."
        ),
        "tip": "Stick with sharpe unless you have a specific reason.",
    },

    # ── Risk / commission ────────────────────────────────────────────────────
    "cash": {
        "short": "Starting account size for the simulated backtest.",
        "long": (
            "We simulate trading $X starting capital. Sharpe and percentage "
            "returns don't depend on this number, so $10,000 is fine for "
            "comparison purposes."
        ),
        "analogy": "Like saying 'imagine you started with $10k' before telling someone how a strategy did.",
        "tip": "Default $10,000 is standard.",
    },
    "commission": {
        "short": "Cost per trade (as a fraction). 0.002 = 0.2% per trade.",
        "long": (
            "Every buy and every sell loses this fraction to fees + slippage. "
            "Set realistically — 0.0 looks great in backtests but is fantasy.\n\n"
            "  • 0.0010 = institutional / very liquid\n"
            "  • 0.0020 = retail brokerage default\n"
            "  • 0.0050 = small cap, wide spreads"
        ),
        "analogy": "The friction that turns a 'looks profitable' backtest into reality.",
        "tip": "0.002 is realistic for retail US equities.",
    },
    "allow_short": {
        "short": "Let the strategy bet that prices will go DOWN, not just up.",
        "long": (
            "*Long* = buy hoping the price rises. *Short* = sell first, buy back "
            "later, hoping the price fell. Allowing both doubles the strategy's "
            "opportunities, but is more complex and not allowed in some retirement "
            "accounts."
        ),
        "analogy": (
            "Long-only is like a sports better who only ever bets on the favorite. "
            "Allowing shorts means betting on either team — more flexible, but "
            "you need to actually be right about the underdog."
        ),
        "tip": "Off (long-only) for simpler, retirement-friendly strategies.",
    },

    # ── Stage-1 screening ────────────────────────────────────────────────────
    "tune_screen": {
        "short": "Spend extra time tuning EACH indicator before combining them.",
        "long": (
            "By default, each indicator (RSI, MACD, etc.) uses its standard "
            "textbook parameters. With this on, we run a small per-indicator "
            "search to find better parameters specific to *your* stock — then "
            "feed only the winners into Stage 2.\n\n"
            "Slower, but the final strategy is usually meaningfully better."
        ),
        "analogy": (
            "Like building a sports team. The default is drafting players "
            "with their generic stats. With tune-screen on, you give each "
            "candidate a personal tryout first."
        ),
        "tip": "Turn on for serious runs. Off for quick exploration.",
    },
    "tune_trials": {
        "short": "How many parameter sets to test per indicator (only matters when tune-screen is on).",
        "long": (
            "Indicators × tune_trials = total Stage-1 work. ~150 indicators × "
            "8 trials = 1,200 quick evaluations. Going higher rarely helps "
            "because each indicator only has 2–4 parameters to tune."
        ),
        "analogy": "How long each candidate gets in their tryout.",
        "tip": "8 is fine. Use 4 for a quick run.",
    },

    # ── Output / display ─────────────────────────────────────────────────────
    "top_n": {
        "short": "How many top indicators to display in the final report.",
        "long": "Out of all surviving indicators, show the N most influential in the chart and table.",
        "analogy": "Like 'top 10 most valuable players' from a 50-player roster.",
        "tip": "8 is readable. 20+ becomes a blur.",
    },
    "lookback": {
        "short": "How many recent trading days to show in the signal table.",
        "long": "The traffic-light table shows 1 row per recent day, marking each indicator's signal. Longer lookback = more context but a busier display.",
        "analogy": "How far back the rear-view mirror sees.",
        "tip": "30 trading days ≈ 6 weeks.",
    },
}


def get(key: str) -> dict[str, str]:
    """Return the help record for `key`, or an empty stub if missing."""
    return HELP.get(key, {"short": "", "long": "", "analogy": "", "tip": ""})


# ── Onboarding intro shown on first load ─────────────────────────────────────
INTRO_MARKDOWN = """
### Welcome to ta-automl

This tool helps you discover, tune, and verify **technical analysis trading
strategies** without writing any code.

**The big idea.** Stock charts have ~150 well-known mathematical indicators
(RSI, MACD, Bollinger Bands, …). Most people just pick one or two by gut feel.
This tool tests *all of them*, throws away the useless ones, and uses machine
learning to find the best **combination** — tuned specifically for the stock
and time period you choose.

**Three steps:**
1. **Pick a stock and time range.** A liquid stock (AAPL, NVDA, SPY) over
   5–7 years is ideal.
2. **Pick a strategy.** "Weighted" is the simple, transparent default. The
   AutoML options are more powerful but harder to interpret.
3. **Click Run.** You'll get an equity curve, a signal table, and a list of
   which indicators mattered most.

> Click any **(?)** button next to an option for a deeper explanation and a
> real-world analogy.

**Important caveat.** This is a *backtesting* tool. A strategy that worked on
historical data is not guaranteed to work on future data. Real markets adapt;
patterns decay. Treat results as hypotheses to investigate, not signals to
trade on.
"""
