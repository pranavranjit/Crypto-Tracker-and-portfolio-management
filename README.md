# Crypto Tracker & Portfolio Management

A multi-page [Streamlit](https://streamlit.io) app for momentum backtesting,
portfolio optimization, and market-sentiment analysis. All market data is
fetched **live** (Yahoo Finance / CoinDesk) — there are no local datasets to
maintain, so it deploys cleanly to Streamlit Community Cloud.

## Features

| Page | What it does |
|------|--------------|
| **Momentum Explorer** (home) | Downloads OHLCV from yfinance and backtests a rotational strategy that, at each step, picks the highest predicted-return symbol. Models: Ridge / OLS / ElasticNet. Also compares Min-Variance, Mean-Variance, and Risk-Parity curves with Sharpe, CAGR, and max-drawdown metrics. |
| **Portfolio Optimizer** | Sharpe-ratio optimization (`scipy.optimize`), efficient frontier, and rebalancing suggestions. Optionally accepts an uploaded holdings CSV. |
| **Market Sentiment** | Pulls crypto news from CoinDesk and scores it with a market-tuned VADER lexicon to produce a 0–100 sentiment / fear-greed signal. Runs fully in-memory. |
| **Helper Bot** | A Gemini-powered (`google-generativeai`) assistant with lightweight TF-IDF retrieval over the page's data. |

## Tech stack

Python · Streamlit · pandas / numpy · scikit-learn · scipy · plotly / matplotlib ·
yfinance · NLTK (VADER) · google-generativeai

## Run locally

```bash
# 1. Install dependencies (Python 3.11 or 3.12 recommended)
pip install -r requirements.txt

# 2. Provide secrets (see below): create .streamlit/secrets.toml

# 3. Launch
streamlit run Momentum_Explorer.py
```

The app opens at http://localhost:8501.

## Secrets / configuration

Create `.streamlit/secrets.toml` (already git-ignored — never commit it):

```toml
GEMINI_API_KEY   = "your-google-gemini-key"   # required for the Helper Bot page
COINDESK_API_KEY = "your-coindesk-key"         # optional; enables the news feed
```

The code reads these via `st.secrets[...]` with an `os.environ` fallback, so
you can also export them as environment variables.

## Deploy to Streamlit Community Cloud

> GitHub Pages cannot host this app — it serves only static files, while
> Streamlit needs a Python server. Use Streamlit Community Cloud instead.

1. Push to GitHub (this repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** →
   **Deploy from GitHub**.
3. Set:
   - **Repository:** `pranavranjit/Crypto-Tracker-and-portfolio-management`
   - **Branch:** `main`
   - **Main file path:** `Momentum_Explorer.py`
4. Under **Advanced settings → Secrets**, paste the same TOML as above.
5. Click **Deploy**. The first build installs `requirements.txt`; subsequent
   loads are cached.

First load may take a moment while live data downloads (results are cached for
one hour; use **Refresh Live Data** to bust the cache). If Yahoo Finance
rate-limits the shared cloud IP, wait a minute and refresh.

## Project structure

```
Momentum_Explorer.py      # main page + entry point
pages/
  Portfolio_Optimizer.py  # Sharpe optimization page
  Sentiment_Analysis.py   # market sentiment page
  Gemini.py               # Helper Bot page
  cores/                  # shared logic
    reader.py             # feature engineering / cleaning
    runners.py            # rotational backtest engine
    ml.py                 # rolling forecast models
    sentiment_pipeline.py # in-memory news + VADER pipeline
    rag.py                # TF-IDF retrieval for the bot
    commons.py            # FEATURES / MODELS constants
requirements.txt
.streamlit/               # config + secrets (secrets git-ignored)
```
