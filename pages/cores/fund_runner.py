"""
Mutual Fund Portfolio Runner
Multi-symbol rotation strategy optimized for mutual funds
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from .fund_analysis import (
    compute_fund_features,
    cross_sectional_zscore,
    rolling_fund_backtest,
    compute_portfolio_metrics,
)


def build_price_panel(
    symbol_to_df: Dict[str, pd.DataFrame],
    symbols: List[str],
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Build wide price panel from multiple funds.
    
    Parameters:
    -----------
    symbol_to_df : Dict[str, pd.DataFrame]
        Dictionary mapping symbols to DataFrames
    symbols : List[str]
        List of symbols to include
    price_col : str
        Price column name ('close', 'price', 'adj_close', etc.)
    
    Returns:
    --------
    pd.DataFrame
        Wide panel with dates as index and symbols as columns (all NaNs dropped)
    """
    panel = {}
    
    for sym in symbols:
        df = symbol_to_df.get(sym)
        if df is None:
            continue
        
        # Find datecolumn
        date_col = None
        for col in ["date", "Date"]:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            continue
        
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
        dfx = dfx.dropna(subset=[date_col]).sort_values(date_col)
        
        # Find price column
        price_found = None
        for col in [price_col, "close", "price", "Close", "Price", "adj_close", "Adj Close"]:
            if col in dfx.columns:
                price_found = col
                break
        
        if price_found is None:
            numeric_cols = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c])]
            price_found = numeric_cols[-1] if numeric_cols else None
        
        if price_found is None:
            continue
        
        s = dfx.set_index(date_col)[price_found].astype(float).rename(sym)
        panel[sym] = s
    
    if not panel:
        raise ValueError("No usable price data found for selected symbols.")
    
    PX = pd.concat(panel.values(), axis=1).dropna(how="any")
    return PX


def xsltr_regime_backtest(
    symbol_to_df: Dict[str, pd.DataFrame],
    symbols: List[str],
    pred_horizon: int = 5,
    initial_capital: float = 10_000.0,
    rebalance_days: int = 5,
    lookback_days: int = 120,
    buffer: int = 40,
    ridge_alpha: float = 1.0,
    sentiment_regime: str = "neutral",
) -> pd.DataFrame:
    """
    Cross-sectional regression backtest with sentiment regime adjustments.
    
    Parameters:
    -----------
    symbol_to_df : Dict[str, pd.DataFrame]
        Dictionary of fund DataFrames
    symbols : List[str]
        List of symbols to backtest
    pred_horizon : int
        Prediction horizon in days
    initial_capital : float
        Starting capital
    rebalance_days : int
        Rebalancing frequency
    lookback_days : int
        Look-back window for training features
    buffer : int
        Buffer period before backtesting starts
    ridge_alpha : float
        Ridge regression alpha
    sentiment_regime : str
        Sentiment regime ('fear', 'greed', 'extreme_greed', 'neutral')
    
    Returns:
    --------
    pd.DataFrame
        Portfolio value time series
    """
    from sklearn.linear_model import Ridge
    
    # Build price panel
    PX = build_price_panel(symbol_to_df, symbols)
    
    # Compute features for all symbols
    feat = {}
    for sym in PX.columns:
        feat[sym] = compute_fund_features(PX[sym])
    
    # Stack features
    F = pd.concat({sym: feat[sym] for sym in PX.columns}, axis=1)
    
    # Regime-based hyperparameters
    if sentiment_regime == "fear":
        topk, exposure = 1, 0.4
    elif sentiment_regime == "greed":
        topk, exposure = 5, 1.0
    elif sentiment_regime == "extreme_greed":
        topk, exposure = 6, 1.0
    else:  # neutral
        topk, exposure = 3, 0.7
    
    dates = PX.index.to_list()
    if len(PX) < (lookback_days + buffer + pred_horizon + 2):
        raise ValueError(
            f"Not enough history: {len(PX)} rows vs required "
            f"{lookback_days + buffer + pred_horizon + 2}"
        )
    
    portfolio_val = initial_capital
    curve = []
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    
    start_idx = lookback_days + buffer
    end_idx = len(dates) - pred_horizon - 1
    rebal_indices = list(range(start_idx, end_idx, rebalance_days))
    
    for t_idx in rebal_indices:
        if t_idx + pred_horizon >= len(dates):
            continue
        
        t0 = dates[t_idx]
        t1 = dates[t_idx + pred_horizon]
        
        # ====== Training Panel ======
        rows_X, rows_y = [], []
        
        for d_idx in range(t_idx - lookback_days, t_idx):
            d = dates[d_idx]
            d1_idx = d_idx + pred_horizon
            if d1_idx >= len(dates):
                continue
            d1 = dates[d1_idx]
            
            # Cross-sectional features at date d
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
            
            Xd = pd.DataFrame(xs_rows, index=syms_ok, columns=["r1", "r5", "r10", "r20", "vol20", "mom20"])
            Xd = cross_sectional_zscore(Xd)
            
            # Returns between d and d+pred_horizon
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
            curve.append((t1, portfolio_val))
            continue
        
        # Fit model
        X_train = np.vstack(rows_X)
        y_train = np.concatenate(rows_y)
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        if len(X_train) < 30:
            curve.append((t1, portfolio_val))
            continue
        
        ridge.fit(X_train, y_train)
        
        # ====== Score at t0 ======
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
            curve.append((t1, portfolio_val))
            continue
        
        X0 = pd.DataFrame(feat_rows).T
        X0.columns = ["r1", "r5", "r10", "r20", "vol20", "mom20"]
        X0 = cross_sectional_zscore(X0)
        
        # Get top-k predictions
        scores = ridge.predict(X0.values)
        rank = pd.Series(scores, index=X0.index).sort_values(ascending=False)
        k = min(topk, len(rank))
        
        if k == 0:
            curve.append((t1, portfolio_val))
            continue
        
        picks = list(rank.index[:k])
        
        # Compute equal-weight returns
        rets = []
        for sym in picks:
            p0 = PX.loc[t0, sym]
            p1 = PX.loc[t1, sym]
            r = p1 / p0 - 1.0
            if np.isfinite(r):
                rets.append(r)
        
        if not rets:
            curve.append((t1, portfolio_val))
            continue
        
        portfolio_val *= (1.0 + float(np.mean(rets)))
        curve.append((t1, portfolio_val))
    
    out = pd.DataFrame(curve, columns=["date", "portfolio_value"]).dropna()
    return out


def multi_fund_rotation(
    symbol_to_df: Dict[str, pd.DataFrame],
    symbols: List[str],
    model_name_list: List[str],
    features: List[str],
    train_size: int = 60,
    pred_horizon: int = 5,
    threshold: float = 0.0,
    initial_capital: float = 10_000.0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Multi-model rotation strategy for mutual funds.
    Picks the symbol with highest predicted return at each date.
    
    Parameters:
    -----------
    symbol_to_df : Dict[str, pd.DataFrame]
        Dictionary of fund DataFrames
    symbols : List[str]
        List of symbols
    model_name_list : List[str]
        List of model names to use
    features : List[str]
        List of feature names
    train_size : int
        Training window
    pred_horizon : int
        Prediction horizon
    threshold : float
        Prediction threshold
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, float]]
        (portfolio_curves, sharpe_ratios)
    """
    from .reader import add_lagged_return_target  # Fallback if not in reader
    
    model_symbol_preds = {model: {} for model in model_name_list}
    
    # Get predictions for each symbol and model
    for sym in symbols:
        df = symbol_to_df[sym]
        
        # Prepare data
        if "return_target" not in df.columns:
            df = add_lagged_return_target(df)
        
        # Get predictions for each model
        for model_name in model_name_list:
            try:
                preds = rolling_fund_backtest(
                    df=df,
                    feature_cols=features,
                    model_name=model_name,
                    train_size=train_size,
                    pred_horizon=pred_horizon,
                )
                
                pred_df = df.iloc[-len(preds):].copy()
                pred_df["pred"] = preds
                
                if "date" not in pred_df.columns:
                    pred_df["date"] = pred_df.index
                
                model_symbol_preds[model_name][sym] = pred_df[["date", "return", "pred"]]
            
            except Exception:
                continue
    
    portfolio_curves = {}
    sharpe_ratios = {}
    
    # For each model, build rotation portfolio
    for model_name in model_name_list:
        sym_data = model_symbol_preds[model_name]
        
        if not sym_data:
            continue
        
        try:
            all_preds_df = pd.concat(
                [df.assign(symbol=sym).set_index("date") for sym, df in sym_data.items()]
            )
        except Exception:
            continue
        
        # Find max pred per date
        all_preds_df.index = pd.to_datetime(all_preds_df.index)
        all_preds_df = all_preds_df.sort_index()
        
        max_preds = all_preds_df.groupby("date")["pred"].transform("max")
        best_pred_per_date = all_preds_df[all_preds_df["pred"] == max_preds].copy()
        
        best_pred_per_date.reset_index(inplace=True)
        best_pred_per_date = best_pred_per_date.drop_duplicates(subset=["date"], keep="first")
        best_pred_per_date.set_index("date", inplace=True)
        best_pred_per_date.index.name = "date"
        
        # Create full date range
        full_date_range = pd.date_range(
            start=best_pred_per_date.index.min(),
            end=best_pred_per_date.index.max(),
            freq="D",
        )
        best_pred_per_date = best_pred_per_date.reindex(full_date_range)
        
        best_pred_per_date["pred"] = best_pred_per_date["pred"].fillna(0)
        best_pred_per_date["return"] = best_pred_per_date["return"].fillna(0)
        
        # Apply threshold
        daily_returns = best_pred_per_date.apply(
            lambda row: row["return"] if abs(row["pred"]) >= threshold else 0,
            axis=1,
        )
        
        # Compute portfolio values
        portfolio_values = (1 + daily_returns).cumprod() * initial_capital
        
        portfolio_curves[model_name] = pd.DataFrame({
            "date": portfolio_values.index,
            "portfolio_value": portfolio_values.values,
        })
        
        # Sharpe
        sharpe = 0.0
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        sharpe_ratios[model_name] = sharpe
    
    return portfolio_curves, sharpe_ratios
