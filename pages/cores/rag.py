"""Tiny in-memory RAG retriever for portfolio / financial-advice context.

Uses sklearn TF-IDF + cosine similarity over a curated knowledge base of
short, self-contained passages. No external vector DB, no network calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_BASE: List[Tuple[str, str]] = [
    (
        "Sharpe ratio",
        "The Sharpe ratio measures excess return per unit of total risk: "
        "(annualized return - risk-free rate) / annualized volatility. "
        "Rule of thumb: <0.5 weak, 0.5-1.0 acceptable, 1.0-2.0 good, >2.0 excellent. "
        "It penalizes both upside and downside volatility equally; for skewed return "
        "profiles consider Sortino (downside-only) or Calmar (return / max drawdown)."
    ),
    (
        "Maximum drawdown",
        "Maximum drawdown is the largest peak-to-trough loss before a new high. "
        "A -20% drawdown means $100 fell to $80 at the worst point. Drawdowns matter "
        "more than volatility for investors with short horizons or who may need to "
        "withdraw during a downturn. A strategy with high Sharpe but >40% drawdown "
        "may still be unsuitable for conservative investors."
    ),
    (
        "CAGR",
        "Compound Annual Growth Rate normalizes total return to a per-year figure: "
        "(ending/starting)^(1/years) - 1. CAGR smooths volatility; pair it with "
        "drawdown and Sharpe to judge the path, not just the destination. Equity "
        "long-run CAGR is roughly 8-10% nominal; anything claiming sustained 30%+ "
        "deserves scrutiny."
    ),
    (
        "Asset allocation buckets",
        "Three buckets simplify portfolio construction: Defensive (bonds, cash, "
        "T-bills) for capital preservation; Core (broad equity index funds like "
        "SPY/VTI) for long-run growth; Risk-On (factor tilts, single stocks, "
        "crypto, leveraged plays) for outsized upside. Typical weightings: "
        "Conservative 70/25/5, Balanced 30/55/15, Aggressive 10/45/45 "
        "(Defensive/Core/Risk-On)."
    ),
    (
        "Diversification",
        "Diversification reduces unsystematic (idiosyncratic) risk but cannot "
        "eliminate systematic (market) risk. Benefits flatten after ~20-30 "
        "uncorrelated names. Correlation matters more than count - 30 tech stocks "
        "is less diversified than 5 assets across equity/bonds/commodities. Watch "
        "for hidden correlation that spikes in crises (when you need diversification "
        "most)."
    ),
    (
        "Mean-Variance optimization",
        "Markowitz mean-variance optimization picks weights to maximize "
        "expected return for a given variance, or minimize variance for a "
        "given return. Inputs are noisy: small changes in expected-return "
        "estimates produce large weight swings. Mitigate with shrinkage "
        "estimators, position caps, or robust alternatives (Black-Litterman, "
        "resampling)."
    ),
    (
        "Minimum-variance portfolio",
        "The minimum-variance portfolio sits at the leftmost tip of the efficient "
        "frontier - lowest possible volatility for the chosen asset set. It does "
        "not require return forecasts (only the covariance matrix), making it "
        "more robust than mean-variance. Often dominated by low-volatility assets; "
        "good defensive allocation but may lag in strong bull markets."
    ),
    (
        "Risk parity",
        "Risk parity assigns weights so each asset contributes equal risk (not "
        "equal capital). Typically inverse-volatility weighted, often with "
        "leverage applied to lower-vol assets like bonds. Popularized by "
        "Bridgewater's All Weather. Performs best when assets are uncorrelated; "
        "vulnerable when bonds and equities sell off together (e.g., 2022)."
    ),
    (
        "Momentum strategy",
        "Momentum exploits the empirical tendency for recent winners to keep "
        "winning over 3-12 month horizons. Strong long-run Sharpe but brutal "
        "drawdowns during regime shifts (-40% to -50% in 2009 momentum crash). "
        "Works best with cross-sectional ranking, frequent rebalancing, and "
        "volatility targeting overlays. Avoid pure 1-month momentum (short-term "
        "reversal dominates)."
    ),
    (
        "Rebalancing cadence",
        "Rebalancing frequency trades off drift control against transaction "
        "costs and taxes. Quarterly or annual rebalancing is standard for "
        "long-term portfolios. Threshold-based rebalancing (rebalance when a "
        "sleeve drifts >5% from target) is more efficient than calendar-based. "
        "Tax-loss harvesting can be combined with rebalancing in taxable accounts."
    ),
    (
        "Fear / Greed regime",
        "Sentiment indicators (Fear & Greed Index, VIX, put/call ratio) act as "
        "contrarian signals at extremes. Extreme Fear (<25) historically marks "
        "favorable entry points; Extreme Greed (>75) signals froth and elevated "
        "drawdown risk. Sentiment alone is not a timing tool - combine with "
        "trend or valuation filters. Position sizing should fade as sentiment "
        "reaches extremes."
    ),
    (
        "Position sizing",
        "Position size should scale with conviction, volatility, and portfolio "
        "risk budget. The Kelly criterion gives optimal sizing for known edge: "
        "f = (bp - q) / b. In practice use fractional Kelly (1/4 to 1/2 Kelly) "
        "to handle estimation error. A simple rule: risk no more than 1-2% of "
        "capital on any single position's stop-loss distance."
    ),
    (
        "Risk tolerance and horizon",
        "Risk tolerance combines capacity (financial ability to absorb losses) "
        "and willingness (psychological comfort). Longer horizons can absorb "
        "more equity volatility - 100% equities is reasonable at 25 with no "
        "near-term liabilities, inappropriate at 65 living off the portfolio. "
        "Rule of thumb: equity % = 110 - age, adjusted for income stability "
        "and other assets."
    ),
    (
        "Volatility interpretation",
        "Annualized volatility is the standard deviation of daily returns "
        "scaled by sqrt(252). 15% annual vol means a 1-sigma year ranges "
        "roughly +/-15% around expected return. Equity indices: 12-20% in "
        "normal regimes, 25-40% in crises. Single stocks: 30-60%. Bonds: "
        "3-8%. Crypto: 60-100%+. Vol is mean-reverting but can spike for "
        "extended periods."
    ),
    (
        "Common pitfalls",
        "Common retail errors: chasing past winners, overconcentration in "
        "employer stock, over-trading driven by news cycle, ignoring fees "
        "and tax drag, using leverage without volatility budgeting, and "
        "panic-selling at drawdown bottoms. The single biggest predictor of "
        "long-term returns is staying invested through volatility - missing "
        "the 10 best days per decade can halve total return."
    ),
    (
        "Bonds and duration",
        "Bond prices move inversely to yields; duration measures sensitivity "
        "(a 7-year duration bond loses ~7% if yields rise 1%). Short-duration "
        "(<3y) preserves capital with modest yield; long-duration (>10y) "
        "amplifies rate bets. In a rising-rate regime favor short duration "
        "and floating-rate; in a falling-rate regime extend duration. "
        "Investment-grade for stability, high-yield carries equity-like risk."
    ),
    (
        "Crypto allocation",
        "Crypto is a high-volatility, high-correlation-in-crisis Risk-On "
        "asset. Reasonable allocation for risk-tolerant investors is 1-5% "
        "of total portfolio; aggressive growth posture 5-15%. Concentrate "
        "in BTC/ETH for the core sleeve; reserve smaller positions for "
        "alts. Expect 70-90% drawdowns within multi-year cycles. Cold-storage "
        "or qualified custody is mandatory for sizable holdings."
    ),
]


@dataclass
class RetrievedChunk:
    title: str
    text: str
    score: float


class RagRetriever:
    def __init__(self, knowledge_base: List[Tuple[str, str]] = KNOWLEDGE_BASE):
        self.titles = [t for t, _ in knowledge_base]
        self.texts = [x for _, x in knowledge_base]
        self.docs = [f"{t}. {x}" for t, x in knowledge_base]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query: str, k: int = 4, min_score: float = 0.05) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        order = np.argsort(-sims)[:k]
        out: List[RetrievedChunk] = []
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                continue
            out.append(RetrievedChunk(title=self.titles[idx], text=self.texts[idx], score=score))
        return out


_RETRIEVER: RagRetriever | None = None


def get_retriever() -> RagRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = RagRetriever()
    return _RETRIEVER


def format_for_prompt(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    lines = []
    for c in chunks:
        lines.append(f"- [{c.title}] {c.text}")
    return "\n".join(lines)
