import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from pages.cores.commons import FEATURES, MODELS
from pages.cores.yf_session import YF_SESSION
from pages.cores.reader import (
    add_ohlc_features,
    add_volume_and_technical_features,
    addReturns,
    clean_fund_df,
)

# Project imports
from pages.cores.runners import CASH, multi_symbol_rotation_by_model
from pages.cores.ui import apply_theme, footer, hero, section

st.set_page_config(
    page_title="Momentum Explorer · Mutual Fund Portfolio Lab",
    page_icon="📈",
    layout="wide",
)
apply_theme()


m_labels = {
    "ridge": "Ridge",
    "ols": "OLS",
    "elasticnet": "ElasticNet",
}
f_labels = {
    "momentum": "Momentum (14d)",
    "rsi": "RSI (14d)",
    "volatility": "Volatility (14d)",
    "volume_ratio": "Volume ratio",
    "intraday_range": "Intraday range",
    "close_to_high": "Close-to-high",
}

# Ticker -> (name, vehicle). Mutual funds first: they are what the app is about,
# and ETFs tracking similar indexes sit alongside for comparison.
FUNDS: Dict[str, Tuple[str, str]] = {
    "VFIAX": ("Vanguard 500 Index Admiral", "Mutual fund"),
    "VTSAX": ("Vanguard Total Stock Market Index Admiral", "Mutual fund"),
    "FXAIX": ("Fidelity 500 Index Fund", "Mutual fund"),
    "VIGAX": ("Vanguard Growth Index Admiral", "Mutual fund"),
    "VEMAX": ("Vanguard Emerging Markets Index Admiral", "Mutual fund"),
    "VOO": ("Vanguard S&P 500 ETF", "ETF"),
    "VTI": ("Vanguard Total Stock Market ETF", "ETF"),
    "SPY": ("SPDR S&P 500 ETF Trust", "ETF"),
    "QQQ": ("Invesco QQQ Trust (Nasdaq-100)", "ETF"),
    "IWM": ("iShares Russell 2000 ETF", "ETF"),
    "VIG": ("Vanguard Dividend Appreciation ETF", "ETF"),
    "VXUS": ("Vanguard Total International Stock ETF", "ETF"),
    "BND": ("Vanguard Total Bond Market ETF", "ETF"),
    "BNDX": ("Vanguard Total International Bond ETF", "ETF"),
    "SCHZ": ("Schwab U.S. Aggregate Bond ETF", "ETF"),
}
DEFAULT_TICKERS = list(FUNDS)
DEFAULT_START = "2010-01-01"
# US large-cap, emerging markets, Nasdaq-100 and bonds: different enough that
# rotating between them is a real decision (VOO/VTI/SPY are one index thrice).
DEFAULT_SELECTION = ["VFIAX", "VEMAX", "QQQ", "BND"]

BENCHMARK = "Equal-weight benchmark"
STRATEGY_METHODS = {
    "Min Variance": "MinVar",
    "Mean Variance": "meanvar",
    "Risk Parity": "riskparity",
}
REBALANCE_LABELS = {"ME": "Monthly", "W-FRI": "Weekly (Fridays)", "D": "Daily"}
COLORS = {
    "Ridge": "#4C8DFF",
    "OLS": "#F5A524",
    "ElasticNet": "#3DD68C",
    "Min Variance": "#B388FF",
    "Mean Variance": "#FF6B8A",
    "Risk Parity": "#22D3EE",
    BENCHMARK: "#9AA3B5",
}


# Short descriptions for the selector chips (full names are in the Fund universe tab).
FUND_SHORT = {
    "VFIAX": "S&P 500 MF",
    "VTSAX": "US total market MF",
    "FXAIX": "S&P 500 MF",
    "VIGAX": "US growth MF",
    "VEMAX": "Emerging mkts MF",
    "VOO": "S&P 500",
    "VTI": "US total market",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "US small caps",
    "VIG": "Dividend growth",
    "VXUS": "Intl stocks",
    "BND": "US bonds",
    "BNDX": "Intl bonds",
    "SCHZ": "US agg. bonds",
}


def _fund_label(ticker: str) -> str:
    short = FUND_SHORT.get(ticker)
    return f"{ticker} · {short}" if short else ticker


def _build_symbol_to_df(
    raw_frames: Dict[str, pd.DataFrame], threshold: int = 100
) -> Dict[str, pd.DataFrame]:
    symbol_to_df: Dict[str, pd.DataFrame] = {}
    for symbol, df in raw_frames.items():
        if df is None or df.empty or len(df) < threshold:
            continue
        mini = df.copy()
        # Normalize to date-only (midnight) so dates align across tickers regardless
        # of the timezone/DST offset yfinance attaches.
        mini["date"] = (
            pd.to_datetime(mini["date"], errors="coerce", utc=True)
            .dt.tz_convert("UTC")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        mini = mini.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        mini["symbol"] = symbol

        mini = addReturns(mini)
        mini = add_ohlc_features(mini)
        if "usd_volume" not in mini.columns and "volume" in mini.columns:
            mini = mini.rename(columns={"volume": "usd_volume"})
        mini = add_volume_and_technical_features(mini)

        cleaned = clean_fund_df(mini.reset_index(drop=True))
        if cleaned is None or cleaned.empty or len(cleaned) < threshold:
            continue
        symbol_to_df[symbol] = cleaned
    return symbol_to_df


def _download_history(tickers: Tuple[str, ...], start: str) -> Dict[str, pd.DataFrame]:
    """One batched Yahoo request, then per-ticker retries for anything missing.

    The batch is ~10x faster than fifteen sequential calls; the per-ticker
    fallback keeps the old resilience when Yahoo rate-limits a shared cloud IP.
    """
    frames: Dict[str, pd.DataFrame] = {}
    try:
        data = yf.download(
            list(tickers),
            start=start,
            auto_adjust=True,
            actions=False,
            group_by="ticker",
            threads=True,
            progress=False,
            session=YF_SESSION,
        )
    except Exception:
        data = None
    if data is not None and not data.empty and isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        for t in tickers:
            if t in available:
                frame = data[t].dropna(how="all")
                if not frame.empty:
                    frames[t] = frame

    for t in tickers:
        if t in frames:
            continue
        # Yahoo frequently rate-limits shared cloud IPs; retry a few times
        # with a short backoff before giving up on a ticker.
        for attempt in range(3):
            try:
                hist = yf.Ticker(t, session=YF_SESSION).history(
                    start=start, auto_adjust=True, actions=False
                )
                if hist is not None and not hist.empty:
                    frames[t] = hist
                    break
            except Exception:
                pass
            time.sleep(0.5 * (attempt + 1))
    return frames


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_data(
    tickers: Tuple[str, ...] = tuple(DEFAULT_TICKERS), start: str = DEFAULT_START
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Download OHLCV data live from yfinance and build the symbol_to_df dict.

    Returns (symbol_to_df, last price date). No on-disk caching — Streamlit's
    in-process cache (1 hour) is the only cache.
    """
    raw_frames: Dict[str, pd.DataFrame] = {}
    last_date = None
    for t, hist in _download_history(tickers, start).items():
        hist = hist.reset_index().rename(
            columns={
                "Date": "date",
                "Datetime": "date",
                "index": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
            }
        )
        raw_frames[t] = hist
        d = pd.to_datetime(hist["date"], errors="coerce", utc=True).max()
        if pd.notna(d) and (last_date is None or d > last_date):
            last_date = d

    if not raw_frames:
        raise RuntimeError(
            "Could not download any market data from Yahoo Finance. This is "
            "usually a temporary rate-limit on the server's IP — wait a minute "
            "and click 'Refresh data', or try again shortly."
        )

    as_of = last_date.strftime("%d %b %Y") if last_date is not None else "unknown"
    return _build_symbol_to_df(raw_frames, threshold=100), as_of


def _build_price_panel(symbol_to_df: dict, symbols: List[str]) -> pd.DataFrame:
    """Return wide price panel PX[date x symbol] from symbol_to_df."""
    panel = {}
    for sym in symbols:
        df = symbol_to_df.get(sym)
        if df is None or "date" not in df.columns:
            continue
        dfx = df.copy()
        dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx = dfx.dropna(subset=["date"]).sort_values("date")
        price_colmnmn = _pick_price_colmnmn(dfx)
        if not price_colmnmn:
            continue
        s = dfx.set_index("date")[price_colmnmn].astype(float).rename(sym)
        panel[sym] = s
    if not panel:
        raise ValueError("No usable price columns found for selected symbols.")
    PX = pd.concat(panel.values(), axis=1).dropna(how="any")
    return PX


def _pick_price_colmnmn(df: pd.DataFrame) -> Optional[str]:
    for c in ["close", "price", "Close", "Price", "adj_close", "Adj Close"]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[-1] if num_cols else None


def _pick_adj_close_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "adj_close",
        "Adj Close",
        "Adj_Close",
        "adj close",
        "adjclose",
        "close",
        "Close",
    ]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[-1] if num_cols else None


def compute_sharpe_by_symbol(
    symbol_to_df: dict, annual_factor: float = 252.0, risk_free: float = 0.0
) -> Dict[str, float]:
    """Compute annualized Sharpe ratio per symbol using adj_close (or fallback).

    Returns dict symbol -> sharpe
    """
    out = {}
    for sym, df in symbol_to_df.items():
        try:
            if df is None or df.empty:
                out[sym] = float("nan")
                continue
            col = _pick_adj_close_col(df)
            if col is None:
                out[sym] = float("nan")
                continue
            price = df[col].copy()
            price.index = pd.to_datetime(df.get("date", df.index), errors="coerce")
            price = price.sort_index()
            rets = price.pct_change().dropna()
            if rets.empty or rets.std() == 0:
                out[sym] = float("nan")
                continue
            # excess returns (assume risk_free in same step units is approx 0)
            sharpe = (rets.mean() - 0.0) / (rets.std() + 1e-12) * (annual_factor**0.5)
            out[sym] = float(sharpe)
        except Exception:
            out[sym] = float("nan")
    return out


def _get_features(px: pd.Series) -> pd.DataFrame:
    r1 = px.pct_change(1)
    r5 = px.pct_change(5)
    r10 = px.pct_change(10)
    r20 = px.pct_change(20)
    vol20 = r1.rolling(20).std()
    mom20 = px.pct_change(20)
    return pd.DataFrame(
        {"r1": r1, "r5": r5, "r10": r10, "r20": r20, "vol20": vol20, "mom20": mom20}
    )


def _cs_zscore(df_xs: pd.DataFrame) -> pd.DataFrame:
    return (df_xs - df_xs.mean()) / (df_xs.std(ddof=0) + 1e-9)


def _sentiment_Current_regime_from_session(ss) -> str:
    sig = (
        ss.get("sentiment", {}).get("signals", {})
        if isinstance(ss.get("sentiment", {}), dict)
        else {}
    )
    last7 = sig.get("last7_avg_pct")
    ema = sig.get("ema_now")
    if last7 is None:
        return "neutral"
    if last7 < 45 or (ema is not None and ema < 50):
        return "fear"
    if last7 > 75:
        return "extreme_greed"
    if last7 > 55:
        return "greed"
    return "neutral"


def xsltr_Current_regime_backtest(
    symbol_to_df: dict,
    symbols: list[str],
    pred_horizon: int = 5,
    initial_capital: float = 10_000.0,
    rebalance_days: int = 5,
    rebalance_freq: Optional[str] = None,
    lookbck_days: int = 120,
    buffer: int = 40,
    session_state=None,
) -> pd.DataFrame:

    from sklearn.linear_model import Ridge

    PX = _build_price_panel(symbol_to_df, symbols)
    feat = {sym: _get_features(PX[sym]) for sym in PX.columns}
    F = pd.concat({sym: feat[sym] for sym in PX.columns}, axis=1)

    Current_regime = _sentiment_Current_regime_from_session(session_state or {})
    if Current_regime == "fear":
        topk, exposure = 1, 0.4
    elif Current_regime == "greed":
        topk, exposure = 5, 1.0
    elif Current_regime == "extreme_greed":
        topk, exposure = 6, 1.0
    else:
        topk, exposure = 3, 0.7

    dates = PX.index.to_list()
    if len(PX) < (lookbck_days + buffer + pred_horizon + 2):
        raise ValueError("Not enough history for XSLTR (fetch more symbols).")

    Present_val = initial_capital
    curve = []
    # default ridge alpha (was missing) — keep small regularization
    ridge_alpha = 1.0
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)

    start_idx = lookbck_days + buffer
    end_idx = len(dates) - pred_horizon - 1

    if rebalance_freq:
        sched = pd.date_range(dates[start_idx], dates[end_idx], freq=rebalance_freq)
        rebal_dates = [d for d in sched if d in PX.index]
        rebal_indices = [PX.index.get_loc(d) for d in rebal_dates]
    else:
        rebal_indices = list(range(start_idx, end_idx, rebalance_days))

    for t_idx in rebal_indices:
        if t_idx + pred_horizon >= len(dates):
            continue
        t0 = dates[t_idx]
        t1 = dates[t_idx + pred_horizon]

        # Train panel
        rows_X, rows_y = [], []
        for d_idx in range(t_idx - lookbck_days, t_idx):
            d = dates[d_idx]
            d1_idx = d_idx + pred_horizon
            if d1_idx >= len(dates):
                continue
            d1 = dates[d1_idx]

            xs_rows, syms_ok = [], []
            for sym in PX.columns:
                row_all = F.get(sym)
                if row_all is None or d not in row_all.index:
                    continue
                rowv = row_all.loc[d][["r1", "r5", "r10", "r20", "vol20", "mom20"]]
                if rowv.isna().any():
                    continue
                xs_rows.append(rowv.values.astype(float))
                syms_ok.append(sym)

            if len(xs_rows) < 3:
                continue

            Xd = pd.DataFrame(
                xs_rows,
                index=syms_ok,
                columns=["r1", "r5", "r10", "r20", "vol20", "mom20"],
            )
            Xd = _cs_zscore(Xd)

            keep, yvals = [], []
            for sym in Xd.index:
                p0 = PX.loc[d, sym]
                p1 = PX.loc[d1, sym]
                r = p1 / p0 - 1.0
                if np.isfinite(r):
                    keep.append(sym)
                    yvals.append(r)
            if len(keep) < 3:
                continue

            Xd = Xd.loc[keep]
            rows_X.append(Xd.values)
            rows_y.append(np.array(yvals))

        if not rows_X:
            curve.append((t1, Present_val))
            continue

        X_train = np.vstack(rows_X)
        y_train = np.concatenate(rows_y)
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]
        if len(X_train) < 30:
            curve.append((t1, Present_val))
            continue

        ridge.fit(X_train, y_train)

        # Score at t0
        feat_rows = {}
        for sym in PX.columns:
            row_all = F.get(sym)
            if row_all is None or t0 not in row_all.index:
                continue
            row = row_all.loc[t0][["r1", "r5", "r10", "r20", "vol20", "mom20"]]
            if row.isna().any():
                continue
            feat_rows[sym] = row.values.astype(float)

        if len(feat_rows) < 3:
            curve.append((t1, Present_val))
            continue

        X0 = pd.DataFrame(feat_rows).T
        X0.columns = ["r1", "r5", "r10", "r20", "vol20", "mom20"]
        X0 = _cs_zscore(X0)

        scores = ridge.predict(X0.values)
        rank = pd.Series(scores, index=X0.index).sort_values(ascending=False)
        k = min(topk, len(rank))
        if k == 0:
            curve.append((t1, Present_val))
            continue
        picks = list(rank.index[:k])

        # realising equal-weight period return
        rets = []
        for sym in picks:
            p0 = PX.loc[t0, sym]
            p1 = PX.loc[t1, sym]
            r = p1 / p0 - 1.0
            if np.isfinite(r):
                rets.append(r)
        if not rets:
            curve.append((t1, Present_val))
            continue

        Present_val *= 1.0 + float(np.mean(rets))
        curve.append((t1, Present_val))

    out = pd.DataFrame(curve, columns=["date", "portfolio_value"]).dropna()
    return out


def _rebalance_indices_for_panel(
    PX: pd.DataFrame, lookbck_days: int, rebalance_freq: str
) -> List[int]:
    """Row positions to rebalance on, starting at ``lookbck_days``.

    Calendar dates from the schedule are mapped back to the last trading day
    on or before them, so a month-end that falls on a weekend still
    rebalances (it used to be skipped because Saturday is not in the panel).
    """
    dates = PX.index
    if len(dates) <= lookbck_days + 1:
        return []
    sched = pd.date_range(dates[lookbck_days], dates[-1], freq=rebalance_freq)
    pos = dates.get_indexer(sched, method="pad")
    idxs = sorted({int(p) for p in pos if p >= lookbck_days})
    if not idxs or idxs[0] != lookbck_days:
        idxs = [lookbck_days] + idxs
    return [i for i in idxs if i < len(dates) - 1]


def _safe_pinv(mat: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(mat + 1e-10 * np.eye(mat.shape[0]))


def _weights_MinVar(Sigma: np.ndarray, enforce_nonneg: bool = True) -> np.ndarray:
    n = Sigma.shape[0]
    invS = _safe_pinv(Sigma)
    ones = np.ones(n)
    w = invS @ ones
    denom = float(ones @ w)
    if denom <= 0:
        w = ones / n
    else:
        w = w / denom
    if enforce_nonneg:
        w = np.maximum(w, 0)
        s = w.sum()
        w = (w / s) if s > 0 else np.ones(n) / n
    return w


def _weights_meanvar(
    Sigma: np.ndarray, mu: np.ndarray, enforce_nonneg: bool = True, cap: float = 0.5
) -> np.ndarray:
    invS = _safe_pinv(Sigma)
    w = invS @ mu
    w = np.maximum(w, 0) if enforce_nonneg else w
    if enforce_nonneg:
        w = np.minimum(w, cap)
    s = w.sum()
    w = (w / s) if s > 0 else np.ones_like(w) / len(w)
    return w


def _weights_risk_parity(vol: np.ndarray) -> np.ndarray:
    inv_vol = 1.0 / np.maximum(vol, 1e-12)
    w = inv_vol / inv_vol.sum()
    return w


def _allocation_weights(window: pd.DataFrame, method: str) -> np.ndarray:
    n = window.shape[1]
    if n == 1:
        return np.ones(1)
    if window.shape[0] < 20:
        return np.ones(n) / n
    mu = window.mean().to_numpy()
    Sigma = np.cov(window.to_numpy().T)
    vol = window.std().to_numpy()
    if method == "MinVar":
        return _weights_MinVar(Sigma, enforce_nonneg=True)
    if method == "meanvar":
        return _weights_meanvar(Sigma, mu, enforce_nonneg=True, cap=0.5)
    if method == "riskparity":
        return _weights_risk_parity(vol)
    raise ValueError(f"Unknown method {method}")


def _cumret_backtest(
    symbol_to_df: dict,
    symbols: List[str],
    initial_capital: float,
    rebalance_freq: str,
    lookbck_days: int,
    method: str,
) -> pd.DataFrame:
    """Daily value of a long-only allocation re-weighted on a calendar schedule.

    At each rebalance close the weights come from the trailing ``lookbck_days``
    of returns (all known at that close); between rebalances the positions
    drift with prices, as a real portfolio would.
    """
    PX = _build_price_panel(symbol_to_df, symbols)
    ret = PX.pct_change()

    idxs = _rebalance_indices_for_panel(PX, lookbck_days, rebalance_freq)
    if not idxs:
        raise ValueError(
            "No rebalancing points available. Try shorter frequency or longer history."
        )
    rebalance_days = set(idxs)
    asset_ret = ret.fillna(0.0).to_numpy()

    start = idxs[0]
    holdings = None
    Present_val = float(initial_capital)
    values = []
    for i in range(start, len(PX)):
        if holdings is not None:
            holdings = holdings * (1.0 + asset_ret[i])
            Present_val = float(holdings.sum())
        if i in rebalance_days:
            window = ret.iloc[max(1, i - lookbck_days + 1) : i + 1].dropna(how="any")
            holdings = Present_val * _allocation_weights(window, method)
        values.append(Present_val)

    return pd.DataFrame({"date": PX.index[start:], "portfolio_value": values})


def _equal_weight_curve(
    symbol_to_df: dict, symbols: List[str], initial_capital: float
) -> pd.DataFrame:
    """Equal weight in every selected fund, rebalanced daily: the do-nothing-clever baseline."""
    PX = _build_price_panel(symbol_to_df, symbols)
    rets = PX.pct_change().fillna(0.0).mean(axis=1)
    values = initial_capital * (1.0 + rets).cumprod()
    return pd.DataFrame({"date": PX.index, "portfolio_value": values.to_numpy()})


def _annualize_factor_from_index(idx: pd.DatetimeIndex) -> Tuple[float, float]:
    if len(idx) < 2:
        return (float("nan"), float("nan"))
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    step_days = int(np.median(gaps)) if len(gaps) else 1
    step_days = max(step_days, 1)
    annual_steps = 252 / step_days
    return float(np.sqrt(annual_steps)), float(step_days)


def compute_curve_metrics(pf_df: pd.DataFrame) -> Tuple[float, float, float]:

    MIN_STEPS_FOR_SHARPE = 5
    MIN_DAYS_FOR_CAGR = 90
    MIN_POINTS_FOR_MDD = 3

    if pf_df is None or pf_df.empty or "date" not in pf_df.columns:
        return (np.nan, np.nan, np.nan)

    s = pf_df.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"])
    if s.empty:
        return (np.nan, np.nan, np.nan)

    Present_val = s.set_index("date")["portfolio_value"].dropna()
    if Present_val.size < 2:
        return (np.nan, np.nan, np.nan)

    # Sharpe
    step_ret = Present_val.pct_change().dropna()
    if len(step_ret) >= MIN_STEPS_FOR_SHARPE and step_ret.std() > 0:
        ann_factor, _ = _annualize_factor_from_index(Present_val.index)
        if not np.isfinite(ann_factor):
            ann_factor = np.sqrt(252.0)
        sharpe = (step_ret.mean() / (step_ret.std() + 1e-12)) * ann_factor
    else:
        sharpe = np.nan

    # CAGR
    total_days = (Present_val.index[-1] - Present_val.index[0]).days
    if total_days >= MIN_DAYS_FOR_CAGR and Present_val.iloc[0] > 0:
        cagr = (Present_val.iloc[-1] / Present_val.iloc[0]) ** (
            365.25 / total_days
        ) - 1.0
    else:
        cagr = np.nan

    # Max Drawdown
    if len(Present_val) >= MIN_POINTS_FOR_MDD:
        running_max = Present_val.cummax()
        dd = Present_val / running_max - 1.0
        mdd = float(dd.min()) if not dd.empty else np.nan
    else:
        mdd = np.nan

    return (float(sharpe), float(cagr), mdd)


def _curve_stats(pf_df: pd.DataFrame) -> Dict[str, float]:
    """The metrics table row for one curve (all curves here are daily)."""
    sh, cg, mdd = compute_curve_metrics(pf_df)
    values = pf_df.set_index("date")["portfolio_value"]
    daily = values.pct_change().dropna()
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
    stats = {
        "Sharpe Ratio": float(sh) if np.isfinite(sh) else np.nan,
        "CAGR (%)": float(cg * 100.0) if np.isfinite(cg) else np.nan,
        "Volatility (%)": float(daily.std() * np.sqrt(252) * 100.0) if len(daily) > 1 else np.nan,
        "Max Drawdown (%)": float(mdd * 100.0) if np.isfinite(mdd) else np.nan,
        "Final Value": float(values.iloc[-1]),
    }
    if "holding" in pf_df.columns:
        held = pf_df["holding"].dropna()
        stats["Switches / yr"] = float(held.ne(held.shift()).iloc[1:].sum() / years)
        stats["Time in cash (%)"] = float((held == CASH).mean() * 100.0)
    return stats


def _align_curves(curves: Dict[str, pd.DataFrame], initial_capital: float) -> Dict[str, pd.DataFrame]:
    """Cut every curve to the window they all share and rebase to the same start.

    The models need a training window before their first trade and the
    allocations a covariance lookback, so without this each curve would be
    measured over a different period and CAGRs would not be comparable.
    """
    usable = {k: v for k, v in curves.items() if v is not None and len(v) > 1}
    if not usable:
        return {}
    start = max(pd.Timestamp(v["date"].iloc[0]) for v in usable.values())
    end = min(pd.Timestamp(v["date"].iloc[-1]) for v in usable.values())
    out = {}
    for name, df in usable.items():
        d = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(d) < 2:
            continue
        d = d.copy()
        d["portfolio_value"] = d["portfolio_value"] / d["portfolio_value"].iloc[0] * initial_capital
        out[name] = d
    return out


def _effective_symbols_and_train_size(
    symbol_to_df: dict, symbols: list[str], train_size: int, pred_h: int
):
    ok = []
    min_len = 10**9
    for s in symbols:
        df = symbol_to_df[s]
        if "date" not in df.columns:
            continue
        n = len(df)
        if n <= pred_h + 10:
            continue
        ok.append(s)
        min_len = min(min_len, n)
    if not ok:
        return [], train_size
    max_train = max(30, min_len - pred_h - 5)
    eff_train = int(min(train_size, max_train))
    return ok, eff_train


def _safe_core_backtest(
    symbol_to_df: dict,
    symbols: list[str],
    model_list,
    features,
    train_size: int,
    pred_horizon: int,
    threshold: float,
    initial_capital: float,
    cost_bps: float = 0.0,
    warnings: Optional[List[str]] = None,
):
    portfolio_outputs = {}
    sharpe_ratio = {}
    kwargs = dict(
        symbol_to_df={s: symbol_to_df[s] for s in symbols},
        symbols=symbols,
        features=features,
        train_size=train_size,
        pred_horizon=pred_horizon,
        threshold=threshold,
        initial_capital=initial_capital,
        cost_bps=cost_bps,
    )
    try:
        return multi_symbol_rotation_by_model(model_name_list=model_list, **kwargs)
    except Exception as e:
        msg = str(e)
        if "Length of values" not in msg:
            raise
        for m in model_list:
            try:
                pc, sr = multi_symbol_rotation_by_model(model_name_list=[m], **kwargs)
                portfolio_outputs.update(pc)
                sharpe_ratio.update(sr)
            except Exception as ee:
                if warnings is not None:
                    warnings.append(f"Skipped model '{m}': {ee}")
        if not portfolio_outputs:
            raise ValueError(
                "All selected models failed to produce predictions (check features/train size)."
            )
        return portfolio_outputs, sharpe_ratio


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60, max_entries=32)
def run_backtest(
    _symbol_to_df: dict,
    data_key: str,
    symbols: Tuple[str, ...],
    models: Tuple[str, ...],
    features: Tuple[str, ...],
    train_size: int,
    pred_horizon: int,
    threshold_bps: float,
    cost_bps: float,
    initial_capital: float,
    rebalance_freq: str,
    strategies: Tuple[str, ...],
) -> dict:
    """Run every selected model and allocation, aligned to one common window.

    Cached across sessions for a day, keyed on the parameters and ``data_key``
    (which changes with the latest price date), so the default configuration
    is trained once and then served instantly to every visitor.
    """
    warnings: List[str] = []
    syms, eff_train_size = _effective_symbols_and_train_size(
        _symbol_to_df, list(symbols), train_size, pred_horizon
    )
    if not syms:
        raise ValueError(
            "None of the selected funds has enough history for these settings."
        )

    model_curves, _ = _safe_core_backtest(
        _symbol_to_df,
        syms,
        list(models),
        list(features),
        eff_train_size,
        pred_horizon,
        threshold_bps / 1e4,
        initial_capital,
        cost_bps=cost_bps,
        warnings=warnings,
    )
    curves = {m_labels.get(k, k): v for k, v in model_curves.items()}

    for label in strategies:
        try:
            curves[label] = _cumret_backtest(
                _symbol_to_df,
                syms,
                initial_capital,
                rebalance_freq,
                eff_train_size,
                method=STRATEGY_METHODS[label],
            )
        except Exception as e:
            warnings.append(f"{label} skipped: {e}")

    curves[BENCHMARK] = _equal_weight_curve(_symbol_to_df, syms, initial_capital)
    aligned = _align_curves(curves, initial_capital)
    bench_curve = aligned.pop(BENCHMARK, None)

    metrics = {name: _curve_stats(df) for name, df in aligned.items()}
    return {
        "portfolio_outputs": aligned,
        "metrics": metrics,
        "sharpe_ratio": {k: v["Sharpe Ratio"] for k, v in metrics.items()},
        "benchmark": {
            "name": BENCHMARK,
            "curve": bench_curve,
            "metrics": _curve_stats(bench_curve) if bench_curve is not None else {},
        },
        "warnings": warnings,
        "params": {
            "symbols": syms,
            "models": list(models),
            "features": list(features),
            "train_size": eff_train_size,
            "pred_horizon": pred_horizon,
            "threshold_bps": threshold_bps,
            "cost_bps": cost_bps,
            "initial_capital": initial_capital,
            "rebalance_freq": rebalance_freq,
            "cum_strats": list(strategies),
        },
    }


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fund_universe_table(_symbol_to_df: dict, data_key: str) -> pd.DataFrame:
    sharpe = compute_sharpe_by_symbol(_symbol_to_df)
    rows = []
    for sym, df in _symbol_to_df.items():
        px = df.set_index("date")["close"].astype(float)
        years = max((px.index[-1] - px.index[0]).days / 365.25, 1e-9)
        daily = px.pct_change().dropna()
        name, vehicle = FUNDS.get(sym, (sym, ""))
        rows.append(
            {
                "Ticker": sym,
                "Fund": name,
                "Type": vehicle,
                "Since": px.index[0].strftime("%b %Y"),
                "CAGR (%)": ((px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1) * 100,
                "Volatility (%)": daily.std() * np.sqrt(252) * 100,
                "Sharpe Ratio": sharpe.get(sym, np.nan),
                "Max Drawdown (%)": (px / px.cummax() - 1).min() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("Sharpe Ratio", ascending=False)


def _style_fig(fig: go.Figure, height: int = 460, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        yaxis_title=y_title,
        xaxis_title=None,
    )
    return fig


def _trace_style(name: str) -> dict:
    color = COLORS.get(name, "#C9D1E2")
    if name == BENCHMARK:
        return dict(color=color, width=2, dash="dot")
    if name in STRATEGY_METHODS:
        return dict(color=color, width=1.6, dash="dash")
    return dict(color=color, width=2)


def _fmt(v, fmt: str, suffix: str = "") -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:{fmt}}{suffix}"


def _delta(a, b, fmt: str, suffix: str) -> Optional[str]:
    """Difference for st.metric's delta, or None (no arrow) when undefined."""
    d = (a if a is not None else np.nan) - (b if b is not None else np.nan)
    return f"{d:{fmt}}{suffix}" if np.isfinite(d) else None


def _render_hero(live_badge: str, fund_badge: str) -> None:
    hero(
        "Momentum Explorer",
        "Can a machine-learning model pick which fund to hold tomorrow? Each model is "
        "retrained walk-forward on momentum, RSI, volatility and volume features, "
        "then compared with classic allocations and an equal-weight benchmark on live data.",
        live_badge=live_badge,
        badges=[fund_badge, "Walk-forward · no look-ahead", "Ridge · OLS · ElasticNet"],
    )


# ---------------------------------------------------------------- page body
hero_slot = st.empty()
with hero_slot.container():
    _render_hero("Loading live data…", f"{len(DEFAULT_TICKERS)} funds")

try:
    with st.spinner("Downloading fund prices from Yahoo Finance…"):
        symbol_to_df, data_as_of = load_data()
except Exception as _e:
    st.error(f"Failed to download market data: {_e}")
    st.stop()

available_symbols = [t for t in DEFAULT_TICKERS if t in symbol_to_df]
n_mutual = sum(1 for t in available_symbols if FUNDS.get(t, ("", ""))[1] == "Mutual fund")
data_key = f"{data_as_of}|{','.join(available_symbols)}"

with hero_slot.container():
    _render_hero(
        f"Live Yahoo Finance data · prices to {data_as_of}",
        f"{len(available_symbols)} funds ({n_mutual} mutual funds)",
    )

default_selection = [t for t in DEFAULT_SELECTION if t in symbol_to_df] or available_symbols[:4]

section("Configure the backtest")
with st.form("backtest_config"):
    cfg1, cfg2, cfg3 = st.columns([1.5, 1, 1.3])
    with cfg1:
        symbols = st.multiselect(
            "Funds to rotate between",
            options=available_symbols,
            default=default_selection,
            format_func=_fund_label,
            help="Each day the strategy holds the ONE fund its model expects to do best tomorrow.",
        )
    with cfg2:
        selected_models = st.multiselect(
            "Models",
            options=MODELS,
            default=MODELS,
            format_func=lambda x: m_labels.get(x, x.replace("_", " ").title()),
        )
    with cfg3:
        selected_features = st.multiselect(
            "Features",
            options=FEATURES,
            default=FEATURES,
            format_func=lambda x: f_labels.get(x, x.replace("_", " ").title()),
        )

    with st.expander("⚙️ Advanced settings", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            train_size = st.slider(
                "Initial training window (trading days)",
                30,
                250,
                120,
                help="History the models see before their first forecast. After that the window keeps expanding.",
            )
            pred_horizon = st.slider(
                "Refit models every (days)",
                1,
                20,
                5,
                help="Models are re-trained on everything known so far, then forecast the next day until the next refit.",
            )
        with adv2:
            threshold_bps = st.slider(
                "Min. forecast to invest (bps/day)",
                0.0,
                10.0,
                0.0,
                step=0.5,
                help="Hold cash unless the best forecast next-day return beats this. 1 bp = 0.01%.",
            )
            cost_bps = st.slider(
                "Trading cost per switch (bps)",
                0.0,
                25.0,
                0.0,
                step=0.5,
                help="Charged every time the rotation changes fund (or moves in/out of cash).",
            )
        with adv3:
            initial_capital = st.number_input(
                "Initial capital ($)", 1000, 1_000_000, 10_000, step=1000
            )
            rebalance_freq = st.selectbox(
                "Allocation rebalancing",
                list(REBALANCE_LABELS),
                index=0,
                format_func=REBALANCE_LABELS.get,
                help="How often Min Variance / Mean Variance / Risk Parity reset their weights.",
            )
            cum_strats_to_add = st.multiselect(
                "Allocation strategies",
                list(STRATEGY_METHODS),
                default=list(STRATEGY_METHODS),
            )

    run_button = st.form_submit_button("▶ Run backtest", type="primary")

if run_button or "momentum" not in st.session_state:
    if not symbols or not selected_models or not selected_features:
        st.warning("Pick at least one fund, one model and one feature, then run the backtest.")
    else:
        try:
            with st.spinner("Training walk-forward models and simulating every strategy…"):
                st.session_state["momentum"] = run_backtest(
                    symbol_to_df,
                    data_key,
                    tuple(symbols),
                    tuple(selected_models),
                    tuple(selected_features),
                    int(train_size),
                    int(pred_horizon),
                    float(threshold_bps),
                    float(cost_bps),
                    float(initial_capital),
                    rebalance_freq,
                    tuple(cum_strats_to_add),
                )
        except Exception:
            st.error("Backtest failed.")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

res = st.session_state.get("momentum")
if res and res.get("metrics"):
    outputs: Dict[str, pd.DataFrame] = res["portfolio_outputs"]
    metrics: Dict[str, Dict[str, float]] = res["metrics"]
    bench = res.get("benchmark") or {}
    bench_curve = bench.get("curve")
    bench_m = bench.get("metrics") or {}
    params = res["params"]
    for w in res.get("warnings", []):
        st.warning(w)

    def _sharpe_key(name):
        v = metrics[name].get("Sharpe Ratio", np.nan)
        return v if np.isfinite(v) else -np.inf

    best = max(metrics, key=_sharpe_key)
    bm = metrics[best]
    rotation_names = [n for n in outputs if "holding" in outputs[n].columns]
    signal_model = best if best in rotation_names else (rotation_names[0] if rotation_names else None)
    first_date = min(df["date"].iloc[0] for df in outputs.values())
    last_date = max(df["date"].iloc[-1] for df in outputs.values())

    section("Results")
    cap_col, btn_col = st.columns([5, 1], vertical_alignment="center")
    cap_col.caption(
        f"{', '.join(params['symbols'])} · {first_date:%b %Y} – {last_date:%b %Y} · "
        f"every curve starts with ${params['initial_capital']:,.0f} on the same day"
    )
    if btn_col.button(
        "🔄 Refresh data",
        type="tertiary",
        use_container_width=True,
        help="Clear the cache and pull fresh prices from Yahoo Finance.",
    ):
        load_data.clear()
        run_backtest.clear()
        fund_universe_table.clear()
        st.session_state.pop("momentum", None)
        st.rerun()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Top strategy (Sharpe)", best)
    k2.metric(
        "Sharpe ratio",
        _fmt(bm.get("Sharpe Ratio"), ".2f"),
        delta=_delta(bm.get("Sharpe Ratio"), bench_m.get("Sharpe Ratio"), "+.2f", " vs benchmark"),
    )
    k3.metric(
        "CAGR",
        _fmt(bm.get("CAGR (%)"), ".1f", "%"),
        delta=_delta(bm.get("CAGR (%)"), bench_m.get("CAGR (%)"), "+.1f", " pts vs benchmark"),
    )
    k4.metric(
        "Max drawdown",
        _fmt(bm.get("Max Drawdown (%)"), ".1f", "%"),
        delta=_delta(bm.get("Max Drawdown (%)"), bench_m.get("Max Drawdown (%)"), "+.1f", " pts vs benchmark"),
    )
    if signal_model:
        sig = outputs[signal_model].dropna(subset=["holding"])
        k5.metric(
            "Latest signal",
            str(sig["holding"].iloc[-1]),
            help=f"{signal_model}'s pick at the close of {sig['date'].iloc[-1]:%d %b %Y}, held for the next trading day.",
        )
    else:
        k5.metric("Benchmark CAGR", _fmt(bench_m.get("CAGR (%)"), ".1f", "%"))

    tab_growth, tab_dd, tab_monthly, tab_hold, tab_metrics, tab_universe = st.tabs(
        ["📈 Growth", "📉 Drawdowns", "🗓️ Monthly returns", "🧭 Holdings", "📋 Metrics", "🔎 Fund universe"]
    )

    all_curves = dict(outputs)
    if bench_curve is not None:
        all_curves[BENCHMARK] = bench_curve

    with tab_growth:
        log_scale = st.toggle("Log scale", value=True, help="Log scale makes equal percentage moves look equal.")
        fig = go.Figure()
        for name, df in all_curves.items():
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["portfolio_value"],
                    mode="lines",
                    name=name,
                    line=_trace_style(name),
                    hovertemplate="%{fullData.name}: $%{y:,.0f}<extra></extra>",
                )
            )
        _style_fig(fig, height=500, y_title="Portfolio value ($)")
        fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Solid: ML rotation models (one fund at a time) · dashed: allocation strategies · "
            "dotted: equal weight in every selected fund, rebalanced daily."
        )

    with tab_dd:
        default_dd = [best] + ([BENCHMARK] if BENCHMARK in all_curves else [])
        dd_pick = st.multiselect("Curves", list(all_curves), default=default_dd, key="dd_pick")
        fig_dd = go.Figure()
        for name in dd_pick:
            df = all_curves[name]
            dd = df["portfolio_value"] / df["portfolio_value"].cummax() - 1.0
            fig_dd.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=dd * 100,
                    mode="lines",
                    name=name,
                    line=_trace_style(name),
                    fill="tozeroy" if name != BENCHMARK else None,
                    hovertemplate="%{fullData.name}: %{y:.1f}%<extra></extra>",
                )
            )
        _style_fig(fig_dd, height=420, y_title="Drawdown from peak (%)")
        st.plotly_chart(fig_dd, use_container_width=True)
        st.caption("How far each portfolio sat below its previous high. Shallower and shorter is better.")

    with tab_monthly:
        strat = st.selectbox("Strategy", list(all_curves), index=list(all_curves).index(best), key="heat_pick")
        s = all_curves[strat].set_index("date")["portfolio_value"]
        month_end = s.resample("ME").last()
        prev = month_end.shift(1)
        prev.iloc[0] = s.iloc[0]
        mret = (month_end / prev - 1.0) * 100
        table = (
            pd.DataFrame({"Year": mret.index.year, "Month": mret.index.strftime("%b"), "r": mret.values})
            .pivot(index="Year", columns="Month", values="r")
            .reindex(columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        )
        year_end = s.resample("YE").last()
        prev_y = year_end.shift(1)
        prev_y.iloc[0] = s.iloc[0]
        z_month = table.to_numpy()
        z_year = ((year_end / prev_y - 1.0) * 100).to_numpy().reshape(-1, 1)
        years = [str(y) for y in table.index]

        def _cell_text(a):
            return [["" if not np.isfinite(v) else f"{v:.1f}" for v in row] for row in a]

        def _limit(a, pct):
            a = np.abs(a[np.isfinite(a)])
            return float(np.percentile(a, pct)) if a.size else 5.0

        heat = go.Figure()
        # Months and full years get separate colour scales: a +15% year would
        # otherwise saturate a scale built for +/-4% months.
        lim_m = _limit(z_month, 95)
        heat.add_trace(
            go.Heatmap(
                z=z_month,
                x=list(table.columns),
                y=years,
                colorscale="RdYlGn",
                zmid=0,
                zmin=-lim_m,
                zmax=lim_m,
                text=_cell_text(z_month),
                texttemplate="%{text}",
                textfont=dict(size=10),
                xgap=2,
                ygap=2,
                hovertemplate="%{x} %{y}: %{z:.2f}%<extra></extra>",
                colorbar=dict(title="Month %", thickness=12),
            )
        )
        lim_y = _limit(z_year, 100)
        heat.add_trace(
            go.Heatmap(
                z=z_year,
                x=["Full year"],
                y=years,
                colorscale="RdYlGn",
                zmid=0,
                zmin=-lim_y,
                zmax=lim_y,
                text=_cell_text(z_year),
                texttemplate="<b>%{text}</b>",
                textfont=dict(size=11),
                xgap=2,
                ygap=2,
                showscale=False,
                hovertemplate="%{y} full year: %{z:.2f}%<extra></extra>",
            )
        )
        heat.update_layout(
            height=max(360, 26 * len(years) + 70),
            margin=dict(l=8, r=8, t=8, b=8),
            yaxis=dict(autorange="reversed", type="category"),
            xaxis=dict(side="top", type="category"),
        )
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})
        st.caption("Monthly returns (%), with the full-year return in the last column.")

    with tab_hold:
        if not rotation_names:
            st.info("Select at least one ML model to see what it held.")
        else:
            shares = []
            for name in rotation_names:
                held = outputs[name]["holding"].dropna()
                pct = held.value_counts(normalize=True) * 100
                for fund, v in pct.items():
                    shares.append({"Model": name, "Fund": fund, "Share of days (%)": v})
            share_df = pd.DataFrame(shares)
            fig_h = go.Figure()
            for name in rotation_names:
                d = share_df[share_df["Model"] == name]
                fig_h.add_trace(
                    go.Bar(
                        x=d["Fund"],
                        y=d["Share of days (%)"],
                        name=name,
                        marker_color=COLORS.get(name),
                        hovertemplate="%{fullData.name} held %{x} on %{y:.0f}% of days<extra></extra>",
                    )
                )
            _style_fig(fig_h, height=380, y_title="Share of trading days (%)")
            fig_h.update_layout(barmode="group", hovermode="closest")
            st.plotly_chart(fig_h, use_container_width=True)

            latest_rows = []
            for name in rotation_names:
                h = outputs[name].dropna(subset=["holding"])
                recent = h.tail(21)["holding"]
                latest_rows.append(
                    {
                        "Model": name,
                        "Latest pick": h["holding"].iloc[-1],
                        "As of close": h["date"].iloc[-1].strftime("%d %b %Y"),
                        "Switches in last month": int(recent.ne(recent.shift()).iloc[1:].sum()),
                    }
                )
            st.dataframe(pd.DataFrame(latest_rows), hide_index=True, use_container_width=True)

    with tab_metrics:
        rows = dict(metrics)
        if bench_m:
            rows[BENCHMARK] = bench_m
        metrics_df = pd.DataFrame(rows).T
        metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan).sort_values("Sharpe Ratio", ascending=False)
        # Turnover columns only apply to the rotation models; show a dash, not "None".
        for col, fmt in (("Switches / yr", "{:.0f}"), ("Time in cash (%)", "{:.0f}%")):
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].map(lambda v, f=fmt: "—" if pd.isna(v) else f.format(v))
        col_cfg = {
            "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
            "CAGR (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Volatility (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Max Drawdown (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Final Value": st.column_config.NumberColumn(format="dollar"),
        }
        st.dataframe(metrics_df, use_container_width=True, column_config=col_cfg)
        wide = pd.concat(
            {name: df.set_index("date")["portfolio_value"] for name, df in all_curves.items()}, axis=1
        )
        st.download_button(
            "⬇️ Download equity curves (CSV)",
            wide.to_csv().encode("utf-8"),
            file_name="momentum_equity_curves.csv",
            mime="text/csv",
        )

    with tab_universe:
        uni = fund_universe_table(symbol_to_df, data_key)
        st.dataframe(
            uni,
            hide_index=True,
            use_container_width=True,
            height=36 * (len(uni) + 1) + 4,
            column_config={
                "CAGR (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Volatility (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
                "Max Drawdown (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption(
            "Buy-and-hold statistics for every fund loaded, over each fund's full history "
            "(dividends reinvested). Sharpe assumes a 0% risk-free rate."
        )

with st.expander("How the backtest works"):
    st.markdown(
        """
- **Target.** Every model predicts a fund's *next-day* return from features computed only
  from prices up to today's close, so a forecast made today is traded tomorrow.
- **Walk-forward training.** Models start with the initial training window, forecast the
  next block of days, then are refit on everything known so far. Features are standardised
  with statistics from each training window only.
- **Rotation.** Each day the portfolio holds the single fund with the highest forecast, or
  cash if no forecast beats the threshold. Optional trading costs are charged on every switch.
- **Allocations.** Min Variance, Mean Variance (long-only, 50% cap) and Risk Parity reset
  weights on the chosen schedule from the trailing window and drift with prices in between.
- **Benchmark.** Equal weight in every selected fund, rebalanced daily.
- **Data.** Daily prices adjusted for dividends and splits from Yahoo Finance, refreshed hourly.
        """
    )

footer()
