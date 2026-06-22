# pages/sentiment_analysis.py
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pages.cores.sentiment_pipeline import run_sentiment_pipeline_live

st.set_page_config(page_title="Sentiment", layout="wide")
st.title("Market Sentiment")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    thres = st.slider("Neutral band (|compound| <=)", 0.0, 0.2, 0.05, 0.01)
with colB:
    smoothing_days = st.slider("Daily smoothing (EMA days)", 1, 14, 7)
with colC:
    refresh = st.button("Refresh Live Data")


@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_sentiment_df(thr: float = 0.05, api_key: str | None = None) -> pd.DataFrame:
    """Fetch news live from CoinDesk + score with VADER, fully in-memory.

    No local files are read or written. Cached for one hour; click
    'Refresh Live Data' to bust the cache.
    """
    api_key = api_key or os.environ.get("COINDESK_API_KEY") or None
    result = run_sentiment_pipeline_live(api_key=api_key, thr=thr)
    df_final = result.get("df_final")
    if df_final is None or df_final.empty:
        raise RuntimeError("Sentiment pipeline returned no rows.")
    return df_final


if refresh:
    fetch_sentiment_df.clear()
    st.rerun()

try:
    df = fetch_sentiment_df(thr=float(thres))
except Exception as e:
    st.error(f"Sentiment pipeline failed: {e}")
    st.stop()

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["compound", "pos", "neu", "neg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "compound"]).sort_values("date")
if df.empty:
    st.error("No rows with valid dates/compound found after pipeline run.")
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


st.session_state.setdefault("sentiment", {})
st.session_state["sentiment"].update(
    {
        "df": df,
        "daily": daily,
        "params": {"thres": float(thres), "smoothing_days": int(smoothing_days)},
        "metrics": {
            "days": int(len(daily)),
            "articles": int(df.shape[0]),
            "ema_now": float(daily["ema"].iloc[-1]) if len(daily) else None,
            "last7_avg_pct": float(daily.tail(7)["avg_pct"].mean()) if len(daily) else None,
        },
        "signals": {
            "last7_avg_pct": float(daily.tail(7)["avg_pct"].mean()) if len(daily) else None,
            "ema_now": float(daily["ema"].iloc[-1]) if len(daily) else None,
            "latest_date": str(daily["date"].iloc[-1]) if len(daily) else None,
        },
    }
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Days covered", f"{len(daily):,}")
k2.metric("Articles", f"{int(df.shape[0]):,}")
k3.metric("Latest EMA", f"{daily['ema'].iloc[-1]:.1f}" if len(daily) else "—")
k4.metric("Neutral band", f"±{thres:.2f}")

tab_overvi, tab_heatmap, tab_distrib, tab_invest = st.tabs(
    ["Overview", "Heatmap", "Distributions", "Investment View"]
)

with tab_overvi:
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["avg_pct"],
            name="Daily",
            mode="lines",
            line=dict(width=1),
            opacity=0.45,
        )
    )
    trend_fig.add_trace(
        go.Scatter(x=daily["date"], y=daily["ema"], name=f"EMA {smoothing_days}", mode="lines")
    )
    trend_fig.add_hline(y=50, line_dash="dash", annotation_text="Neutral (50)")
    trend_fig.update_layout(
        title="Daily Average Sentiment (0-100)",
        xaxis_title="Date",
        yaxis_title="Index",
        hovermode="x unified",
    )
    st.plotly_chart(trend_fig, use_container_width=True)

with tab_heatmap:
    m = daily.copy()
    m["Year"] = m["date"].dt.year
    m["Month"] = m["date"].dt.strftime("%b")
    heat = m.pivot_table(
        index="Year", columns="Month", values="avg_pct", aggfunc="mean"
    ).reindex(
        columns=[
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    )

    fig_hm = go.Figure(
        go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale="RdYlGn",
            zmid=50,
            colorbar=dict(title="%"),
            hovertemplate="Year=%{y}<br>Month=%{x}<br>Avg=%{z:.1f}%<extra></extra>",
        )
    )
    fig_hm.update_layout(
        title="Monthly Average Sentiment (0-100)", xaxis_title="Month", yaxis_title="Year"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with tab_distrib:
    st.caption("Use the slider to see the effect of thresholding.")
    thres_local = st.slider(
        "Histogram threshold |compound| >=", 0.00, 0.20, float(thres), 0.01, key="thres_hist"
    )
    df_thres = df.loc[df["compound"].abs() >= thres_local].copy()

    cA, cB = st.columns(2)
    cA.plotly_chart(
        px.histogram(df, x="compound", nbins=60, title="Compound (all)"),
        use_container_width=True,
    )
    cB.plotly_chart(
        px.histogram(
            df_thres,
            x="compound",
            nbins=60,
            title=f"Compound (|compound| >= {thres_local:.2f})",
        ),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    for name, holder in zip(["pos", "neu", "neg"], [c1, c2, c3]):
        if name in df.columns:
            holder.plotly_chart(
                px.histogram(df_thres, x=name, nbins=50, title=f"Distribution: {name}"),
                use_container_width=True,
            )

with tab_invest:
    avg_last7 = float(daily.tail(7)["avg_pct"].mean()) if len(daily) else 50.0
    ema_now = float(daily["ema"].iloc[-1]) if len(daily) else 50.0
    ema_prev = (
        float(daily["ema"].iloc[-8])
        if len(daily) >= 8
        else (float(daily["ema"].iloc[0]) if len(daily) else 50.0)
    )
    ema_slope = ema_now - ema_prev

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

    current_regime = map_to_gauge(avg_last7)

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

    st.markdown(
        f"**Regime:** `{current_regime}` • **Risk score:** `{risk_metric:.1f}` -> "
        f"Defensive `{allocation['Defensive']*100:.0f}%`, "
        f"Core `{allocation['Core']*100:.0f}%`, "
        f"Risk-On `{allocation['Risk-On']*100:.0f}%`"
    )

    st.divider()
    st.subheader("Risk-On Sleeve (from Momentum)")

    momo = st.session_state.get("momentum", {}) or {}
    portfolio_outputs = momo.get("portfolio_outputs", {}) or {}
    metrics = momo.get("metrics", {}) or {}

    if not portfolio_outputs:
        st.info(
            "No Momentum results in session. Run a backtest on the Momentum Explorer page "
            "to populate the Risk-On sleeve."
        )
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
                    name=f"{top_model} (Sharpe={top_sharpe:.2f})" if top_sharpe is not None else top_model,
                    hovertemplate="%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
                )
            )
            if risk_on_wt > 0:
                fig_ro.add_trace(
                    go.Scatter(
                        x=ro_curve["date"],
                        y=ro_curve["portfolio_value"] * risk_on_wt,
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"{top_model} × Risk-On weight ({risk_on_wt:.0%})",
                    )
                )
            fig_ro.update_layout(
                title=f"Risk-On Model Performance — {top_model}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                hovermode="x unified",
                legend_title="Models",
            )
            st.plotly_chart(fig_ro, use_container_width=True)
            st.caption(
                f"Regime: {current_regime} • Risk-On weight = {risk_on_wt:.0%} • Baseline: {top_model}"
            )
