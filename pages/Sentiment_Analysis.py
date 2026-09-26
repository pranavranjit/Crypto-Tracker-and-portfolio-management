# pages/Sentiment_Analysis.py
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pages.cores.sentiment_pipeline import run_sentiment_pipeline_live
from pages.cores.ui import apply_theme, footer, hero, section

st.set_page_config(
    page_title="Market Sentiment · Mutual Fund Portfolio Lab",
    page_icon="📰",
    layout="wide",
)
apply_theme()

REGIMES = [  # (upper bound, label, colour)
    (25, "Extreme Fear", "#E5484D"),
    (45, "Fear", "#F5A524"),
    (55, "Neutral", "#9AA3B5"),
    (75, "Greed", "#7FD6A4"),
    (100, "Extreme Greed", "#3DD68C"),
]


def map_to_gauge(v: float) -> str:
    if v < 25:
        return "Extreme Fear"
    if v < 45:
        return "Fear"
    if v <= 55:
        return "Neutral"
    if v <= 75:
        return "Greed"
    return "Extreme Greed"


def regime_color(v: float) -> str:
    label = map_to_gauge(v)
    return next(color for _, name, color in REGIMES if name == label)


def _coindesk_key():
    try:
        key = st.secrets.get("COINDESK_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("COINDESK_API_KEY") or None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_sentiment_df(api_key: str | None = None) -> tuple[pd.DataFrame, str]:
    """Fetch news live and score every article with VADER, fully in-memory.

    Scored with no neutral-band cut (thr=0) so the slider below filters the
    cached rows instead of re-downloading the news on every change. Cached for
    one hour; 'Refresh' busts the cache.
    """
    result = run_sentiment_pipeline_live(api_key=api_key, thr=0.0)
    df_final = result.get("df_final")
    if df_final is None or df_final.empty:
        raise RuntimeError("No news could be downloaded right now. Please try again shortly.")
    return df_final, result.get("source") or "news feed"


hero_slot = st.empty()


def _hero(badges):
    with hero_slot.container():
        hero(
            "Market Sentiment",
            "Every headline is scored with VADER, using a lexicon tuned with market "
            "terms like 'bullish', 'rekt' and 'all-time high', then averaged per day into a "
            "0–100 fear/greed index that feeds the allocation view.",
            badges=badges,
        )


_hero(["Loading headlines…"])

colA, colB, colC = st.columns([1.2, 1.2, 0.6], vertical_alignment="bottom")
with colA:
    thres = st.slider(
        "Neutral band (drop |compound| below)",
        0.0,
        0.2,
        0.05,
        0.01,
        help="Headlines scoring close to 0 carry little sentiment; this drops them before averaging.",
    )
with colB:
    smoothing_days = st.slider("Smoothing (EMA days)", 1, 14, 7)
with colC:
    refresh = st.button("🔄 Refresh", use_container_width=True, help="Pull the latest headlines.")

if refresh:
    fetch_sentiment_df.clear()
    st.rerun()

try:
    with st.spinner("Downloading headlines and scoring sentiment…"):
        df_all, source = fetch_sentiment_df(_coindesk_key())
except Exception as e:
    _hero(["News source unavailable"])
    st.error(f"Sentiment pipeline failed: {e}")
    st.stop()

df = df_all.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["compound", "pos", "neu", "neg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "compound"]).sort_values("date")
df = df.loc[df["compound"].abs() >= thres].copy()
if df.empty:
    st.error("No headlines left after the neutral-band filter. Lower the threshold.")
    st.stop()


daily = (
    df.set_index("date")
    .groupby(pd.Grouper(freq="D"))
    .agg(
        avg_compound=("compound", "mean"),
        n_articles=("compound", "size"),
    )
    .reset_index()
)
daily = daily.dropna(subset=["avg_compound"])
daily["avg_pct"] = (daily["avg_compound"] + 1.0) * 50.0
daily["ema"] = daily["avg_pct"].ewm(span=smoothing_days, adjust=False).mean()

avg_last7 = float(daily.tail(7)["avg_pct"].mean()) if len(daily) else 50.0
ema_now = float(daily["ema"].iloc[-1]) if len(daily) else 50.0
ema_prev = (
    float(daily["ema"].iloc[-8])
    if len(daily) >= 8
    else (float(daily["ema"].iloc[0]) if len(daily) else 50.0)
)
ema_slope = ema_now - ema_prev
current_regime = map_to_gauge(avg_last7)

_hero(
    [
        f"Source: {source}",
        f"{len(df):,} headlines · {len(daily)} days",
        f"Latest: {daily['date'].iloc[-1]:%d %b %Y}",
    ]
)

st.session_state.setdefault("sentiment", {})
st.session_state["sentiment"].update(
    {
        "df": df,
        "daily": daily,
        "source": source,
        "params": {"thres": float(thres), "smoothing_days": int(smoothing_days)},
        "metrics": {
            "days": int(len(daily)),
            "articles": int(df.shape[0]),
            "ema_now": ema_now,
            "last7_avg_pct": avg_last7,
        },
        "signals": {
            "last7_avg_pct": avg_last7,
            "ema_now": ema_now,
            "latest_date": str(daily["date"].iloc[-1]) if len(daily) else None,
        },
    }
)

# ---- Headline numbers: gauge + KPIs ----
g_col, k_col = st.columns([1.1, 1.4], gap="large")
with g_col:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_last7,
            number=dict(valueformat=".0f", font=dict(size=54)),
            title=dict(text=f"<b>{current_regime}</b><br><span style='font-size:0.8em;color:#8A93A6'>7-day average, 0–100</span>"),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickvals=[0, 25, 45, 55, 75, 100]),
                bar=dict(color="#E6E9F0", thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[lo, hi], color=c)
                    for (lo, (hi, _, c)) in zip([0, 25, 45, 55, 75], REGIMES)
                ],
            ),
        )
    )
    gauge.update_layout(height=290, margin=dict(l=36, r=44, t=70, b=10))
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
with k_col:
    st.write("")
    k1, k2 = st.columns(2)
    k1.metric("Sentiment, last 7 days", f"{avg_last7:.1f}", help="Average daily score, 0 = max fear, 100 = max greed.")
    k2.metric(
        f"EMA-{smoothing_days} trend",
        f"{ema_now:.1f}",
        delta=f"{ema_slope:+.1f} over 7 days",
        help="Smoothed index and its change over the last week.",
    )
    k3, k4 = st.columns(2)
    k3.metric("Headlines scored", f"{len(df):,}")
    k4.metric("Days covered", f"{len(daily):,}")

tab_overvi, tab_headlines, tab_calendar, tab_distrib, tab_invest = st.tabs(
    ["📈 Trend", "📰 Headlines", "🗓️ Calendar", "📊 Distributions", "🧭 Investment view"]
)

with tab_overvi:
    trend_fig = go.Figure()
    lo = 0
    for hi, label, color in REGIMES:
        trend_fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.07, line_width=0)
        lo = hi
    trend_fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["avg_pct"],
            name="Daily average",
            marker_color=[regime_color(v) for v in daily["avg_pct"]],
            opacity=0.55,
            hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra></extra>",
        )
    )
    trend_fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["ema"],
            name=f"EMA {smoothing_days}",
            mode="lines",
            line=dict(color="#E6E9F0", width=2.5),
            hovertemplate="EMA: %{y:.1f}<extra></extra>",
        )
    )
    trend_fig.add_hline(y=50, line_dash="dot", line_color="#8A93A6", annotation_text="Neutral 50")
    ymin = max(0.0, float(daily["avg_pct"].min()) - 5)
    ymax = min(100.0, float(daily["avg_pct"].max()) + 5)
    trend_fig.update_layout(
        height=440,
        margin=dict(l=8, r=8, t=36, b=8),
        yaxis=dict(title="Sentiment index (0–100)", range=[ymin, ymax]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.15,
    )
    st.plotly_chart(trend_fig, use_container_width=True)
    st.caption("Bars: each day's average (coloured by regime). Line: exponential moving average.")

with tab_headlines:
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=6)].copy()
    has_url = "url" in recent.columns
    cols = ["date", "title", "compound"] + (["url"] if has_url else [])
    cfg = {
        "date": st.column_config.DatetimeColumn("Date", format="D MMM", width="small"),
        "title": st.column_config.TextColumn("Headline", width="large"),
        "compound": st.column_config.NumberColumn("Score", format="%+.2f", width="small"),
    }
    if has_url:
        cfg["url"] = st.column_config.LinkColumn("Link", display_text="Open ↗", width="small")
    st.markdown("**😊 Most positive this week**")
    st.dataframe(
        recent.nlargest(6, "compound")[cols], hide_index=True, use_container_width=True, column_config=cfg
    )
    st.markdown("**😟 Most negative this week**")
    st.dataframe(
        recent.nsmallest(6, "compound")[cols], hide_index=True, use_container_width=True, column_config=cfg
    )
    st.caption("VADER compound score from −1 (most negative) to +1 (most positive).")

with tab_calendar:
    cal = daily.set_index("date")["avg_pct"].asfreq("D")
    frame = pd.DataFrame(
        {
            "week": cal.index.to_period("W-SUN").start_time,
            "weekday": cal.index.dayofweek,
            "value": cal.values,
        }
    )
    grid = frame.pivot(index="weekday", columns="week", values="value").reindex(range(7))
    fig_cal = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=list(grid.columns),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=[[0, "#E5484D"], [0.45, "#F5A524"], [0.5, "#9AA3B5"], [0.55, "#7FD6A4"], [1, "#3DD68C"]],
            zmin=max(0.0, 50 - max(10.0, float(np.nanmax(np.abs(grid.values - 50))))),
            zmax=min(100.0, 50 + max(10.0, float(np.nanmax(np.abs(grid.values - 50))))),
            xgap=3,
            ygap=3,
            hovertemplate="Week of %{x|%d %b %Y}, %{y}: %{z:.1f}<extra></extra>",
            colorbar=dict(title="Index", thickness=12),
        )
    )
    fig_cal.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=16, b=8),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickformat="%d %b"),
    )
    st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})
    st.caption("One square per day, one column per week. Grey days had no headlines past the neutral band.")

with tab_distrib:
    st.caption("How strongly positive or negative the scored headlines are.")
    cA, cB = st.columns(2)
    hist = px.histogram(df, x="compound", nbins=50, color_discrete_sequence=["#4C8DFF"])
    hist.update_layout(height=320, margin=dict(l=8, r=8, t=36, b=8), title="Compound score", bargap=0.05)
    cA.plotly_chart(hist, use_container_width=True)
    if {"pos", "neu", "neg"}.issubset(df.columns):
        comp = df[["pos", "neu", "neg"]].mean().rename({"pos": "Positive", "neu": "Neutral", "neg": "Negative"})
        bar = go.Figure(
            go.Bar(
                x=comp.index,
                y=comp.values * 100,
                marker_color=["#3DD68C", "#9AA3B5", "#E5484D"],
                text=[f"{v*100:.0f}%" for v in comp.values],
                textposition="outside",
            )
        )
        bar.update_layout(height=320, margin=dict(l=8, r=8, t=36, b=8), title="Average word-share by tone", yaxis_title="%")
        cB.plotly_chart(bar, use_container_width=True)

with tab_invest:
    ema_component = float(np.clip((ema_now - 50) * 1.2 + 50, 0, 100))
    slope_component = float(np.clip(50 + 4 * ema_slope, 0, 100))
    risk_metric = 0.6 * avg_last7 + 0.3 * ema_component + 0.1 * slope_component

    def risk_to_allocation(score: float) -> dict:
        if score < 35:
            return {"Defensive": 0.70, "Core": 0.25, "Risk-On": 0.05}
        if score < 50:
            return {"Defensive": 0.50, "Core": 0.40, "Risk-On": 0.10}
        if score < 65:
            return {"Defensive": 0.30, "Core": 0.55, "Risk-On": 0.15}
        if score < 80:
            return {"Defensive": 0.15, "Core": 0.55, "Risk-On": 0.30}
        return {"Defensive": 0.10, "Core": 0.45, "Risk-On": 0.45}

    allocation = risk_to_allocation(risk_metric)
    st.session_state["sentiment"]["allocation"] = allocation

    i1, i2 = st.columns([1, 1.6], gap="large")
    with i1:
        donut = go.Figure(
            go.Pie(
                labels=list(allocation),
                values=[v * 100 for v in allocation.values()],
                hole=0.62,
                marker=dict(colors=["#4C8DFF", "#B388FF", "#3DD68C"]),
                textinfo="label+percent",
                sort=False,
            )
        )
        donut.update_layout(
            height=300,
            margin=dict(l=8, r=8, t=8, b=8),
            showlegend=False,
            annotations=[dict(text=f"<b>{risk_metric:.0f}</b><br>risk score", showarrow=False, font=dict(size=16))],
        )
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
    with i2:
        st.markdown(
            f"""
**Regime: {current_regime}** · risk score **{risk_metric:.1f}** / 100

The risk score blends the 7-day level (60%), the smoothed trend (30%) and its
momentum (10%). Higher scores move money from the **Defensive** bucket (bonds,
cash) towards **Core** (broad index funds) and **Risk-On** (the momentum
strategy's picks).

| Bucket | Weight | What goes in it |
|---|---|---|
| Defensive | {allocation['Defensive']*100:.0f}% | Bond funds, T-bills, cash |
| Core | {allocation['Core']*100:.0f}% | Broad index funds (e.g. VFIAX, VTSAX) |
| Risk-On | {allocation['Risk-On']*100:.0f}% | The top momentum model below |
            """
        )

    section("Risk-On sleeve (from the Momentum Explorer)")
    momo = st.session_state.get("momentum", {}) or {}
    portfolio_outputs = momo.get("portfolio_outputs", {}) or {}
    metrics = momo.get("metrics", {}) or {}

    if not portfolio_outputs:
        st.info("Open the Momentum Explorer first; its best model becomes the Risk-On sleeve here.")
        try:
            st.page_link("Momentum_Explorer.py", label="Go to the Momentum Explorer", icon="📈")
        except Exception:  # page registry unavailable (e.g. page run on its own)
            pass
    else:
        ranked = sorted(
            metrics.items(),
            key=lambda kv: kv[1].get("Sharpe Ratio", float("-inf"))
            if isinstance(kv[1], dict)
            else float("-inf"),
            reverse=True,
        )
        top_model = ranked[0][0] if ranked else next(iter(portfolio_outputs))
        top_sharpe = (
            ranked[0][1].get("Sharpe Ratio") if ranked and isinstance(ranked[0][1], dict) else None
        )
        ro_curve = portfolio_outputs.get(top_model)

        if ro_curve is None or getattr(ro_curve, "empty", True):
            st.info("Top-Sharpe curve is empty.")
        else:
            ro_curve = ro_curve.copy()
            ro_curve["date"] = pd.to_datetime(ro_curve["date"], errors="coerce")
            ro_curve = ro_curve.dropna(subset=["date"])

            risk_on_wt = float(allocation.get("Risk-On", 0.0))
            fig_ro = go.Figure()
            fig_ro.add_trace(
                go.Scatter(
                    x=ro_curve["date"],
                    y=ro_curve["portfolio_value"],
                    mode="lines",
                    name=f"{top_model} (Sharpe {top_sharpe:.2f})" if top_sharpe is not None else top_model,
                    line=dict(color="#4C8DFF", width=2),
                    hovertemplate="%{x|%d %b %Y}: $%{y:,.0f}<extra></extra>",
                )
            )
            if risk_on_wt > 0:
                fig_ro.add_trace(
                    go.Scatter(
                        x=ro_curve["date"],
                        y=ro_curve["portfolio_value"] * risk_on_wt,
                        mode="lines",
                        line=dict(dash="dash", color="#3DD68C", width=1.6),
                        name=f"Sized to the Risk-On weight ({risk_on_wt:.0%})",
                        hovertemplate="%{x|%d %b %Y}: $%{y:,.0f}<extra></extra>",
                    )
                )
            fig_ro.update_layout(
                height=400,
                margin=dict(l=8, r=8, t=36, b=8),
                yaxis=dict(title="Portfolio value ($)", tickprefix="$", tickformat=",.0f"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_ro, use_container_width=True)
            st.caption(f"Regime: {current_regime} · Risk-On weight {risk_on_wt:.0%} · model: {top_model}")

footer()
