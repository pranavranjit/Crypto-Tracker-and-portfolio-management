import os
from typing import Any, Dict, List

import google.generativeai as genai
import pandas as pd
import streamlit as st

from pages.cores.rag import RetrievedChunk, format_for_prompt, get_retriever

st.set_page_config(page_title="Helper Bot", layout="centered")
st.title("Helper Bot")
st.subheader("Your AI copilot for market momentum and smart trading decisions")


API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if API_KEY:
    genai.configure(api_key=API_KEY)


def summarize_df(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "None"
    head_csv = df.head(min(max_rows, len(df))).to_csv(index=False)
    return f"[HEAD CSV rows={min(max_rows, len(df))}]\n{head_csv}"


def build_context(ss: Dict[str, Any]) -> str:
    momentum = ss.get("momentum", {}) if isinstance(ss.get("momentum"), dict) else {}
    metrics = momentum.get("metrics", {}) or {}
    params = momentum.get("params", {}) or {}
    symbols = params.get("symbols")
    features = params.get("features")
    models_used = params.get("models")
    initial_capital = params.get("initial_capital")
    rebalance_freq = params.get("rebalance_freq")

    portfolio_outputs = momentum.get("portfolio_outputs", {}) or {}
    metrics_table = "None"
    if metrics:
        metrics_table = pd.DataFrame(metrics).T.to_csv()

    top_model = None
    top_sharpe = None
    if metrics:
        ranked = sorted(
            metrics.items(),
            key=lambda kv: kv[1].get("Sharpe Ratio", float("-inf")) if isinstance(kv[1], dict) else float("-inf"),
            reverse=True,
        )
        if ranked:
            top_model, top_metrics = ranked[0]
            top_sharpe = top_metrics.get("Sharpe Ratio") if isinstance(top_metrics, dict) else None

    top_curve_summary = "None"
    if top_model and top_model in portfolio_outputs:
        pf = portfolio_outputs[top_model]
        if isinstance(pf, pd.DataFrame) and not pf.empty and {"date", "portfolio_value"}.issubset(pf.columns):
            preview = pf[["date", "portfolio_value"]].tail(30).copy()
            preview["date"] = pd.to_datetime(preview["date"]).dt.strftime("%Y-%m-%d")
            top_curve_summary = summarize_df(preview, max_rows=30)

    sentiment = ss.get("sentiment", {}) if isinstance(ss.get("sentiment"), dict) else {}
    sent_signals = sentiment.get("signals", {}) or {}
    sent_metrics = sentiment.get("metrics", {}) or {}
    sent_params = sentiment.get("params", {}) or {}
    allocation = sentiment.get("allocation") or sentiment.get("alloc")

    last7 = sent_signals.get("avg_last7_avg_pct") or sent_signals.get("last7_avg_pct")
    ema_now = sent_signals.get("ema_now")
    regime = _regime_label(last7)

    return f"""
[MOMENTUM]
symbols: {symbols}
models: {models_used}
features: {features}
initial_capital: {initial_capital}
rebalance_freq: {rebalance_freq}
top_model: {top_model}
top_sharpe: {top_sharpe}
metrics_table:
{metrics_table}
top_model_curve_tail:
{top_curve_summary}

[SENTIMENT]
last7_avg_pct: {last7}
ema_now: {ema_now}
regime: {regime}
params: {sent_params}
metrics: {sent_metrics}
suggested_allocation: {allocation}
""".strip()


def _regime_label(score) -> str:
    if score is None:
        return "unknown"
    try:
        v = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if v < 25:
        return "Extreme Fear"
    if v < 45:
        return "Fear"
    if v <= 55:
        return "Neutral"
    if v <= 75:
        return "Greed"
    return "Extreme Greed"


SYSTEM_INSTRUCTION = (
    "You are a cautious, plain-spoken financial analysis assistant. "
    "Use the [KNOWLEDGE] block as authoritative background on portfolio theory, "
    "and the [MOMENTUM] / [SENTIMENT] blocks as the user's live session data. "
    "When making suggestions, cite which knowledge passage or which metric you are using. "
    "Prefer allocation buckets (Defensive / Core / Risk-On) over exact percentages unless "
    "the user asks. Always remind that this is educational information, not personalised "
    "financial advice, and that past performance does not guarantee future results."
)


def build_model() -> "genai.GenerativeModel":
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
    )


def start_chat():
    if "gemini_chat" not in st.session_state:
        model = build_model()
        hist = []
        for m in st.session_state.get("messages", []):
            role = "user" if m.get("role") == "user" else "model"
            content = m.get("content", "")
            if content:
                hist.append({"role": role, "parts": [content]})
        st.session_state.gemini_chat = model.start_chat(history=hist)
    return st.session_state.gemini_chat


def friendly_wrap(raw_text: str, sources: List[RetrievedChunk]) -> str:
    body = raw_text.strip()
    src = ""
    if sources:
        names = ", ".join(c.title for c in sources)
        src = f"\n\n_Knowledge used: {names}_"
    return (
        f"{body}{src}\n\n"
        "_Note: educational information, not financial advice. Backtest before deploying._"
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your fintech copilot. Ask me about your portfolio, the momentum results, or general allocation/risk concepts."}
    ]

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])

prompt = st.chat_input("Ask about allocations, Sharpe, drawdowns, the live signals…")

if prompt:
    if not API_KEY:
        st.error("GEMINI_API_KEY not configured. Set it via st.secrets or env var.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Thinking…")

        try:
            retriever = get_retriever()
            chunks = retriever.retrieve(prompt, k=4)
            knowledge_block = format_for_prompt(chunks) or "(no relevant passages)"

            ctx = build_context(dict(st.session_state))
            chat = start_chat()

            response = chat.send_message([
                {"text": f"[KNOWLEDGE]\n{knowledge_block}"},
                {"text": f"[CONTEXT]\n{ctx}"},
                {"text": f"[USER QUESTION]\n{prompt}"},
            ])

            answer = response.text or "(No answer returned.)"
            friendly_answer = friendly_wrap(answer, chunks)

        except Exception as e:
            friendly_answer = f"Sorry, I hit an error: {e}"

        placeholder.write(friendly_answer)
        st.session_state.messages.append({"role": "assistant", "content": friendly_answer})

    st.rerun()
