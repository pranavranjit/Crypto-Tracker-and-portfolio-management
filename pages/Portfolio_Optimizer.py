"""
Portfolio Optimizer - Streamlit App
Interactive portfolio optimization with Sharpe ratio analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from pages.cores.portfolio_optimizer import PortfolioOptimizer
from pages.cores.yf_session import YF_SESSION
from pages.cores.ui import apply_theme, footer, hero, section, stat_cards

# Configuration
st.set_page_config(
    page_title="Portfolio Optimizer · Mutual Fund Portfolio Lab",
    page_icon="🎯",
    layout="wide",
)
apply_theme()

hero(
    "Portfolio Optimizer",
    "Enter what you hold (or start from the sample mutual-fund portfolio), pull live "
    "prices, and see where your allocation sits against the Sharpe-optimal and "
    "minimum-variance portfolios on the efficient frontier, with a rebalancing plan in dollars.",
    badges=["Markowitz mean-variance", "Long-only, fully invested", "SciPy SLSQP"],
)

# A diversified mix of mutual funds and ETFs; every visitor sees results
# immediately and can edit from there.
SAMPLE_PORTFOLIO = pd.DataFrame(
    {
        "Symbol": ["VFIAX", "VTIAX", "VBTLX", "QQQ", "GLD"],
        "Weight %": [35.0, 15.0, 25.0, 15.0, 10.0],
    }
)
COLORS = {"Current": "#F5A524", "Max Sharpe": "#3DD68C", "Min Variance": "#B388FF"}

if "holdings_df" not in st.session_state:
    st.session_state.holdings_df = SAMPLE_PORTFOLIO.copy()


def _clean_holdings(df: pd.DataFrame) -> Tuple[Dict[str, float], List[str]]:
    """Validate the editor table -> {SYMBOL: fraction}, plus any problems found."""
    problems: List[str] = []
    weights: Dict[str, float] = {}
    for _, row in df.iterrows():
        sym = str(row.get("Symbol") or "").strip().upper()
        if not sym or sym == "NAN":
            continue
        try:
            w = float(row.get("Weight %"))
        except (TypeError, ValueError):
            problems.append(f"{sym}: weight is not a number")
            continue
        if not np.isfinite(w) or w < 0:
            problems.append(f"{sym}: weight must be zero or positive")
            continue
        weights[sym] = weights.get(sym, 0.0) + w / 100.0
    return weights, problems


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_returns(symbols: Tuple[str, ...], lookback_days: int) -> Tuple[pd.DataFrame, List[str]]:
    """Daily returns for the last ``lookback_days`` trading days all symbols share."""
    # Trading days -> calendar days, plus a buffer for holidays.
    start_date = datetime.now() - timedelta(days=int(lookback_days * 365 / 252) + 45)
    data = yf.download(
        list(symbols),
        start=start_date,
        auto_adjust=True,
        progress=False,
        session=YF_SESSION,
    )
    if data is None or data.empty:
        return pd.DataFrame(), list(symbols)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    close = close.dropna(how="all")
    missing = [s for s in symbols if s not in close.columns or close[s].dropna().empty]
    close = close.drop(columns=missing, errors="ignore")
    # Common trading days only (crypto trades at weekends, funds do not).
    close = close.dropna(how="any")
    returns_df = close.pct_change().dropna().tail(lookback_days)
    return returns_df, missing


# ==================== Step 1: holdings + settings ====================
section("1 · Your portfolio")
col_hold, col_set = st.columns([1.35, 1], gap="large")

with col_hold:
    edited = st.data_editor(
        st.session_state.holdings_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="holdings_editor",
        column_config={
            "Symbol": st.column_config.TextColumn(
                "Symbol",
                help="Yahoo Finance ticker: funds (VFIAX), ETFs (SPY), stocks (AAPL) or crypto (BTC-USD).",
                required=True,
            ),
            "Weight %": st.column_config.NumberColumn(
                "Weight %", min_value=0.0, max_value=100.0, step=1.0, format="%.1f%%", required=True
            ),
        },
    )
    b1, b2, b3 = st.columns([1, 1, 1.4])
    if b1.button("↺ Load sample", use_container_width=True):
        st.session_state.holdings_df = SAMPLE_PORTFOLIO.copy()
        st.session_state.pop("holdings_editor", None)
        st.session_state.pop("optimizer", None)
        st.rerun()
    if b2.button("🗑️ Clear", use_container_width=True):
        st.session_state.holdings_df = pd.DataFrame({"Symbol": pd.Series(dtype=str), "Weight %": pd.Series(dtype=float)})
        st.session_state.pop("holdings_editor", None)
        st.session_state.pop("optimizer", None)
        st.rerun()
    with b3.popover("📄 Import CSV", use_container_width=True):
        uploaded_file = st.file_uploader(
            "CSV with columns Symbol, Weight (in %)", type=["csv"], label_visibility="visible"
        )
        if uploaded_file:
            try:
                csv_df = pd.read_csv(uploaded_file)
                cols = {c.lower().strip(): c for c in csv_df.columns}
                if "symbol" in cols and "weight" in cols:
                    st.session_state.holdings_df = pd.DataFrame(
                        {
                            "Symbol": csv_df[cols["symbol"]].astype(str).str.strip().str.upper(),
                            "Weight %": pd.to_numeric(csv_df[cols["weight"]], errors="coerce"),
                        }
                    )
                    st.session_state.pop("holdings_editor", None)
                    st.session_state.pop("optimizer", None)
                    st.rerun()
                else:
                    st.error("CSV must have 'Symbol' and 'Weight' columns")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

weights_in, problems = _clean_holdings(edited)
total_weight = sum(weights_in.values()) * 100

with col_set:
    investment_amount = st.number_input(
        "Portfolio value ($)",
        min_value=1000,
        value=100000,
        step=1000,
        help="Used to turn weight changes into buy/sell amounts.",
    )
    risk_free_rate = st.slider(
        "Risk-free rate (% a year)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1,
        help="Subtracted from returns in the Sharpe ratio, e.g. the 3-month T-bill yield.",
    ) / 100
    lookback_days = st.slider(
        "Lookback (trading days)",
        min_value=63,
        max_value=756,
        value=252,
        step=21,
        help="History used to estimate returns and covariances. 252 ≈ 1 year.",
    )
    m1, m2 = st.columns(2)
    m1.metric("Holdings", len(weights_in))
    m2.metric("Weights sum", f"{total_weight:.0f}%")
    if weights_in and abs(total_weight - 100) > 1:
        st.caption("Weights don't add to 100%, so they will be scaled proportionally.")

for p in problems:
    st.warning(p)

# Initialize optimizer
optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate, lookback_periods=lookback_days)

run = st.button("🚀 Analyze & optimize", type="primary")
settings_key = (tuple(sorted(weights_in.items())), round(risk_free_rate, 6), lookback_days)
if run or ("optimizer" not in st.session_state and weights_in):
    if not weights_in or total_weight <= 0:
        st.error("Add at least one holding with a positive weight.")
    else:
        with st.spinner("Fetching prices and solving the optimization…"):
            try:
                symbols = tuple(weights_in)
                returns_df, missing = fetch_returns(symbols, lookback_days)
                if returns_df.empty or len(returns_df) < 20:
                    raise ValueError(
                        "Not enough price history for these symbols. Check the tickers "
                        "(e.g. BTC-USD for crypto, SPY for ETFs)."
                    )
                available = list(returns_df.columns)
                current_weights = {s: weights_in.get(s, 0.0) for s in available}
                current = optimizer.current_portfolio_performance(current_weights, returns_df)
                optimal = optimizer.optimize_portfolio(returns_df) if len(available) > 1 else None
                min_var = optimizer.min_variance_portfolio(returns_df) if len(available) > 1 else None
                rand_r, rand_v, rand_s = optimizer.efficient_frontier(returns_df, num_portfolios=4000)
                front_v, front_r = optimizer.efficient_frontier_curve(returns_df)
                st.session_state["optimizer"] = {
                    "returns_df": returns_df,
                    "missing": missing,
                    "symbols": available,
                    "current": current,
                    "optimal": optimal,
                    "min_var": min_var,
                    "random": (rand_r, rand_v, rand_s),
                    "frontier": (front_v, front_r),
                    "settings_key": settings_key,
                    "risk_free_rate": risk_free_rate,
                    "lookback_days": lookback_days,
                }
            except Exception as e:
                st.session_state.pop("optimizer", None)
                st.error(f"Could not analyze the portfolio: {e}")

res = st.session_state.get("optimizer")
if res:
    if res["settings_key"] != settings_key:
        st.info("Holdings or settings changed. Click **Analyze & optimize** to update the results below.")
    if res["missing"]:
        st.warning(f"No price data found for: {', '.join(res['missing'])}. They were left out.")

    returns_df = res["returns_df"]
    current, optimal, min_var = res["current"], res["optimal"], res["min_var"]
    rf = res["risk_free_rate"]

    # ==================== Step 2: results ====================
    section("2 · Results")
    st.caption(
        f"{len(returns_df)} trading days to {returns_df.index[-1]:%d %b %Y} · "
        f"annualized · risk-free rate {rf*100:.1f}%"
    )

    def _card(title, perf, accent, base=None):
        stats = []
        for label, key, scale, fmt, unit, higher_is_good in (
            ("Return", "return", 100, ".1f", "%", True),
            ("Volatility", "volatility", 100, ".1f", "%", False),
            ("Sharpe", "sharpe_ratio", 1, ".2f", "", True),
        ):
            value = perf[key] * scale
            delta, good = None, None
            if base is not None:
                diff = value - base[key] * scale
                delta = f"{diff:+{fmt}}{' pts' if unit else ''} vs yours"
                good = (diff >= 0) == higher_is_good
            stats.append((label, f"{value:{fmt}}{unit}", delta, good))
        return {"title": title, "accent": accent, "stats": stats}

    cards = [_card("📊 Your portfolio", current, COLORS["Current"])]
    if optimal is not None:
        cards.append(_card("🎯 Max-Sharpe portfolio", optimal, COLORS["Max Sharpe"], current))
        cards.append(_card("🛡️ Minimum-variance portfolio", min_var, COLORS["Min Variance"], current))
    stat_cards(cards)
    if optimal is None:
        st.info("Add a second holding to optimize the mix.")

    stats = optimizer.calculate_metrics(returns_df)
    tab_frontier, tab_alloc, tab_plan, tab_corr = st.tabs(
        ["📈 Efficient frontier", "🥧 Allocation", "📋 Rebalancing plan", "🔗 Correlations"]
    )

    with tab_frontier:
        rand_r, rand_v, rand_s = res["random"]
        front_v, front_r = res["frontier"]
        fig = go.Figure()
        # SVG, not WebGL: WebGL points always paint above SVG ones and would
        # hide the Current / Max Sharpe markers.
        fig.add_trace(
            go.Scatter(
                x=rand_v * 100,
                y=rand_r * 100,
                mode="markers",
                name="Random portfolios",
                marker=dict(
                    color=rand_s,
                    colorscale="Viridis",
                    size=4,
                    opacity=0.55,
                    colorbar=dict(title="Sharpe", thickness=12),
                ),
                hovertemplate="Risk %{x:.1f}% · return %{y:.1f}%<extra></extra>",
            )
        )
        if len(front_v):
            fig.add_trace(
                go.Scatter(
                    x=front_v * 100,
                    y=front_r * 100,
                    mode="lines",
                    name="Efficient frontier",
                    line=dict(color="#E6E9F0", width=2.5),
                    hovertemplate="Frontier: risk %{x:.1f}% · return %{y:.1f}%<extra></extra>",
                )
            )
        if optimal is not None and optimal["volatility"] > 0:
            # Capital market line: every mix of the risk-free asset and the tangency portfolio.
            x_end = max(float(np.max(rand_v)), optimal["volatility"]) * 100 * 1.05
            slope = (optimal["return"] - rf) / optimal["volatility"]
            fig.add_trace(
                go.Scatter(
                    x=[0, x_end],
                    y=[rf * 100, (rf + slope * x_end / 100) * 100],
                    mode="lines",
                    name="Capital market line",
                    line=dict(color="#3DD68C", width=1.4, dash="dash"),
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=stats["std_returns"].values * 100,
                y=stats["mean_returns"].values * 100,
                mode="markers+text",
                name="Individual holdings",
                text=list(returns_df.columns),
                textposition="top center",
                textfont=dict(size=11, color="#C9D1E2"),
                marker=dict(size=9, color="#8A93A6", line=dict(width=1, color="#0E1117")),
                hovertemplate="%{text}: risk %{x:.1f}% · return %{y:.1f}%<extra></extra>",
            )
        )
        points = [("Current", current, "star", 18)]
        if optimal is not None:
            points += [("Max Sharpe", optimal, "star", 18), ("Min Variance", min_var, "diamond", 13)]
        for label, perf, symbol, size in points:
            fig.add_trace(
                go.Scatter(
                    x=[perf["volatility"] * 100],
                    y=[perf["return"] * 100],
                    mode="markers",
                    name=label,
                    marker=dict(size=size, color=COLORS[label], symbol=symbol, line=dict(width=1.5, color="#0E1117")),
                    hovertemplate=f"<b>{label}</b><br>Risk %{{x:.1f}}% · return %{{y:.1f}}%<extra></extra>",
                )
            )
        fig.update_layout(
            height=560,
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis_title="Risk: annual volatility (%)",
            yaxis_title="Expected annual return (%)",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each dot is a random long-only mix of your holdings. The white curve is the best "
            "achievable return for each level of risk; the max-Sharpe portfolio is where the "
            "capital market line touches it."
        )

    with tab_alloc:
        if optimal is None:
            st.info("Add a second holding to compare allocations.")
        else:
            cur_w = current["weights"]
            opt_w = dict(zip(optimal["symbols"], optimal["weights"]))
            mv_w = dict(zip(min_var["symbols"], min_var["weights"]))
            syms = list(optimal["symbols"])
            fig_a = go.Figure()
            for label, w in (("Current", cur_w), ("Max Sharpe", opt_w), ("Min Variance", mv_w)):
                fig_a.add_trace(
                    go.Bar(
                        name=label,
                        x=syms,
                        y=[w.get(s, 0) * 100 for s in syms],
                        marker_color=COLORS[label],
                        hovertemplate=f"{label}: %{{x}} %{{y:.1f}}%<extra></extra>",
                    )
                )
            fig_a.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=8, r=8, t=36, b=8),
                yaxis_title="Weight (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_a, use_container_width=True)
            alloc_df = pd.DataFrame(
                {
                    "Symbol": syms,
                    "Current %": [cur_w.get(s, 0) * 100 for s in syms],
                    "Max Sharpe %": [opt_w.get(s, 0) * 100 for s in syms],
                    "Min Variance %": [mv_w.get(s, 0) * 100 for s in syms],
                    "Annual return %": [stats["mean_returns"][s] * 100 for s in syms],
                    "Annual volatility %": [stats["std_returns"][s] * 100 for s in syms],
                }
            )
            pct = st.column_config.NumberColumn(format="%.1f%%")
            st.dataframe(
                alloc_df,
                hide_index=True,
                use_container_width=True,
                column_config={c: pct for c in alloc_df.columns if c != "Symbol"},
            )

    with tab_plan:
        if optimal is None:
            st.info("Add a second holding to get a rebalancing plan.")
        else:
            suggestions_df = optimizer.generate_suggestions(
                current["weights"], dict(zip(optimal["symbols"], optimal["weights"])), investment_amount
            )
            if suggestions_df.empty:
                st.success("✅ Your portfolio is already at the max-Sharpe mix. No trades needed.")
            else:
                buys = suggestions_df[suggestions_df["Amount Change"] > 0]
                sells = suggestions_df[suggestions_df["Amount Change"] < 0]
                p1, p2, p3 = st.columns(3)
                p1.metric("Buy / increase", f"{len(buys)} holdings", f"+${buys['Amount Change'].sum():,.0f}",
                          delta_color="off")
                p2.metric("Sell / reduce", f"{len(sells)} holdings", f"-${abs(sells['Amount Change'].sum()):,.0f}",
                          delta_color="off")
                p3.metric("Total turnover", f"${suggestions_df['Amount Change'].abs().sum() / 2:,.0f}",
                          help="Half the sum of all trades: the amount that actually changes hands.")
                show = suggestions_df[
                    ["Action", "Symbol", "Current Weight %", "Optimal Weight %", "Current Amount", "Optimal Amount", "Amount Change"]
                ]
                money = st.column_config.NumberColumn(format="dollar")
                pct = st.column_config.NumberColumn(format="%.1f%%")
                st.dataframe(
                    show,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Current Weight %": pct,
                        "Optimal Weight %": pct,
                        "Current Amount": money,
                        "Optimal Amount": money,
                        "Amount Change": money,
                    },
                )
                st.caption(
                    "Before trading: selling can trigger capital-gains tax, trades cost money, and "
                    "optimal weights are estimated from the past, so many investors move part of the way."
                )
                summary_text = f"""PORTFOLIO OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PORTFOLIO VALUE: ${investment_amount:,.0f}
RISK-FREE RATE: {rf*100:.1f}%
LOOKBACK PERIOD: {res['lookback_days']} trading days

CURRENT PORTFOLIO
  Return: {current['return']*100:.2f}%
  Volatility: {current['volatility']*100:.2f}%
  Sharpe Ratio: {current['sharpe_ratio']:.3f}

MAX-SHARPE PORTFOLIO
  Return: {optimal['return']*100:.2f}%
  Volatility: {optimal['volatility']*100:.2f}%
  Sharpe Ratio: {optimal['sharpe_ratio']:.3f}

REBALANCING
  {len(buys)} holdings to buy/increase
  {len(sells)} holdings to sell/reduce
  Turnover: ${suggestions_df['Amount Change'].abs().sum() / 2:,.0f}

Educational output, not investment advice.
"""
                d1, d2, _ = st.columns([1, 1, 2])
                d1.download_button(
                    "📥 Plan (CSV)",
                    suggestions_df.to_csv(index=False),
                    file_name="portfolio_suggestions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                d2.download_button(
                    "📄 Report (TXT)",
                    summary_text,
                    file_name="portfolio_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    with tab_corr:
        corr = returns_df.corr()
        fig_c = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                colorbar=dict(title="ρ", thickness=12),
                xgap=2,
                ygap=2,
            )
        )
        fig_c.update_layout(
            height=120 + 48 * len(corr),
            margin=dict(l=8, r=8, t=8, b=8),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
        st.caption("Daily-return correlations. Low or negative values are what make diversification work.")

footer()
