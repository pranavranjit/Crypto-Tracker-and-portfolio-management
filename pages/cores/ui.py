"""Shared UI theming for all Streamlit pages.

Call ``apply_theme()`` once, right after ``st.set_page_config(...)`` on every
page, to get a consistent look: comfortable spacing between the sidebar and the
main content, softer cards, and tidier headers. ``hero()`` renders the page
header and ``sidebar_about()`` the author block, so every page reads as one app.
"""
from __future__ import annotations

import html
from typing import Iterable, Optional

import streamlit as st

APP_NAME = "Mutual Fund Portfolio Lab"
REPO_URL = "https://github.com/pranavranjit/Crypto-Tracker-and-portfolio-management"
AUTHOR_URL = "https://www.linkedin.com/in/pranav2ranjit"

_CSS = """
<style>
/* ---- Main content: breathing room, especially next to the sidebar ---- */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    padding-left: 3.5rem;
    padding-right: 3.5rem;
    max-width: 1400px;
}
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
}

/* ---- Sidebar: padding + a soft divider so it doesn't crowd the page ---- */
section[data-testid="stSidebar"] {
    background: #131722;
    border-right: 1px solid #262C3A;
}
section[data-testid="stSidebar"] > div {
    padding: 1.4rem 1.2rem;
}
section[data-testid="stSidebar"] .block-container {
    padding-left: 0;
    padding-right: 0;
}

/* ---- Headings ---- */
h1 {
    font-weight: 700;
    letter-spacing: -0.01em;
    padding-bottom: 0.2rem;
}
h2, h3 { font-weight: 600; }

/* ---- Page header ---- */
.pl-hero { margin: 0 0 1.4rem 0; }
.pl-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8FB3FF;
    margin-bottom: 0.35rem;
}
.pl-hero h1 {
    font-size: 2.1rem;
    line-height: 1.15;
    margin: 0 0 0.45rem 0;
    padding: 0;
}
.pl-hero p.pl-lede {
    color: #AEB5C4;
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 780px;
    margin: 0 0 0.8rem 0;
}
.pl-badges { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.pl-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    color: #C9D1E2;
    background: #171B26;
    border: 1px solid #262C3A;
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
}
.pl-badge .pl-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #3DD68C; box-shadow: 0 0 6px rgba(61, 214, 140, 0.6);
}

/* ---- Section label ---- */
.pl-section {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8A93A6;
    margin: 1.6rem 0 0.6rem 0;
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: #171B26;
    border: 1px solid #262C3A;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
div[data-testid="stMetricLabel"] { opacity: 0.7; }

/* ---- Expander: card-like surface ---- */
div[data-testid="stExpander"] {
    border: 1px solid #262C3A;
    border-radius: 12px;
    background: #171B26;
}

/* ---- Bordered containers / forms ---- */
div[data-testid="stForm"] {
    border: 1px solid #262C3A;
    border-radius: 14px;
    background: #141824;
    padding: 1.2rem 1.3rem 0.6rem 1.3rem;
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px;
    padding: 0.5rem 1.1rem;
    font-weight: 600;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] { font-weight: 600; }

/* ---- Dataframes / tables: rounded corners ---- */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #262C3A;
}

/* ---- Chat ---- */
div[data-testid="stChatMessage"] {
    background: #151A25;
    border: 1px solid #232938;
    border-radius: 12px;
}

/* ---- Comparison cards (stat_cards) ---- */
.pl-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0.9rem;
    margin: 0.2rem 0 1.2rem 0;
}
.pl-card {
    background: #171B26;
    border: 1px solid #262C3A;
    border-top: 3px solid var(--accent, #4C8DFF);
    border-radius: 12px;
    padding: 0.9rem 1.1rem 1rem 1.1rem;
}
.pl-card-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.7rem; color: #E6E9F0; }
.pl-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }
.pl-stat-label { display: block; font-size: 0.74rem; color: #8A93A6; margin-bottom: 0.15rem; }
.pl-stat-value { display: block; font-size: 1.45rem; font-weight: 600; color: #E6E9F0; line-height: 1.2; }
.pl-stat-delta { display: inline-block; font-size: 0.74rem; margin-top: 0.2rem; }
.pl-stat-delta.good { color: #3DD68C; }
.pl-stat-delta.bad { color: #FF6B6B; }

/* ---- Sidebar about block ---- */
.pl-about { font-size: 0.82rem; color: #9AA3B5; line-height: 1.5; }
.pl-about strong { color: #E6E9F0; }
.pl-about a { color: #8FB3FF; text-decoration: none; }
.pl-about a:hover { text-decoration: underline; }

/* ---- Footer ---- */
.pl-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #262C3A;
    color: #7C8599;
    font-size: 0.8rem;
    line-height: 1.5;
}
.pl-footer a { color: #8FB3FF; text-decoration: none; }

/* ---- Thin divider spacing ---- */
hr { margin: 1.4rem 0; }
</style>
"""


def apply_theme() -> None:
    """Inject the shared CSS and the sidebar author block. Safe on every page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    sidebar_about()


def hero(
    title: str,
    lede: str,
    eyebrow: str = APP_NAME,
    badges: Optional[Iterable[str]] = None,
    live_badge: Optional[str] = None,
) -> None:
    """Page header: small eyebrow, title, one-paragraph description and badges.

    ``live_badge`` gets a green status dot (e.g. "Live data as of 24 Sep").
    """
    chips = []
    if live_badge:
        chips.append(
            f'<span class="pl-badge"><span class="pl-dot"></span>{html.escape(live_badge)}</span>'
        )
    for b in badges or []:
        chips.append(f'<span class="pl-badge">{html.escape(b)}</span>')
    st.markdown(
        f"""
        <div class="pl-hero">
            <span class="pl-eyebrow">{html.escape(eyebrow)}</span>
            <h1>{html.escape(title)}</h1>
            <p class="pl-lede">{html.escape(lede)}</p>
            <div class="pl-badges">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    """Small uppercase section label."""
    st.markdown(f'<div class="pl-section">{html.escape(label)}</div>', unsafe_allow_html=True)


def stat_cards(cards: Iterable[dict]) -> None:
    """Side-by-side comparison cards that wrap on narrow screens.

    Each card: {"title": str, "accent": "#hex", "stats": [(label, value,
    delta or None, good: bool or None), ...]}. st.metric inside nested columns
    truncates values ("1…") at a third of the page width; this does not.
    """
    parts = []
    for card in cards:
        stats = []
        for label, value, delta, good in card["stats"]:
            delta_html = ""
            if delta:
                cls = "" if good is None else ("good" if good else "bad")
                delta_html = f'<span class="pl-stat-delta {cls}">{html.escape(delta)}</span>'
            stats.append(
                f'<div><span class="pl-stat-label">{html.escape(label)}</span>'
                f'<span class="pl-stat-value">{html.escape(value)}</span>{delta_html}</div>'
            )
        parts.append(
            f'<div class="pl-card" style="--accent:{card.get("accent", "#4C8DFF")}">'
            f'<div class="pl-card-title">{html.escape(card["title"])}</div>'
            f'<div class="pl-stats">{"".join(stats)}</div></div>'
        )
    st.markdown(f'<div class="pl-cards">{"".join(parts)}</div>', unsafe_allow_html=True)


def sidebar_about() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="pl-about">
                <strong>{APP_NAME}</strong><br>
                ML momentum backtests, Sharpe-optimal allocation and news
                sentiment, on live market data.<br><br>
                Built by <a href="{AUTHOR_URL}" target="_blank">Pranav</a> &middot;
                <a href="{REPO_URL}" target="_blank">Source on GitHub</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def footer() -> None:
    st.markdown(
        f"""
        <div class="pl-footer">
            Educational project, not investment advice. Backtests use historical
            data and past performance does not guarantee future results.
            &middot; <a href="{REPO_URL}" target="_blank">Source code</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
