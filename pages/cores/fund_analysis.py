"""
Mutual Fund Analysis Module
Adapted from Momentum_Explorer.py for core reusability
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet


# ============================================================================
# Feature Engineering for Mutual Funds
# ============================================================================

def compute_fund_features(px: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    Compute key momentum and volatility features for mutual funds.
    
    Parameters:
    -----------
    px : pd.Series
        Price series indexed by date
    periods : List[int]
        List of period windows for returns and volatility. Default: [1, 5, 10, 20]
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with computed features (r1, r5, r10, r20, vol20, mom20)
    """
    if periods is None:
        periods = [1, 5, 10, 20]
    
    features = {}
    
    # Period returns
    for p in periods:
        features[f"r{p}"] = px.pct_change(p)
    
    # Volatility (rolling std of 1-period returns)
    features["vol20"] = px.pct_change(1).rolling(20).std()
    
    # Momentum (same as 20-period return)
    features["mom20"] = px.pct_change(20)
    
    return pd.DataFrame(features)


def cross_sectional_zscore(df_xs: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize cross-sectional features using z-score normalization.
    Applied across symbols (rows) at each date.
    
    Parameters:
    -----------
    df_xs : pd.DataFrame
        DataFrame with symbols as index and features as columns
    
    Returns:
    --------
    pd.DataFrame
        Z-score normalized dataframe
    """
    return (df_xs - df_xs.mean()) / (df_xs.std(ddof=0) + 1e-9)


def prepare_fund_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and targets for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and 'return_target' column
    feature_cols : List[str]
        List of feature column names
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        (Features X, Target y)
    """
    X = df[feature_cols].copy()
    y = df["return_target"] if "return_target" in df.columns else df["return"].shift(-1)
    
    # Remove NaN rows
    valid_idx = X.notna().all(axis=1) & y.notna()
    return X[valid_idx], y[valid_idx]


def add_lagged_return_target(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Add lagged return as prediction target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'return' column
    lag : int
        Number of periods to lag
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'return_target' column added
    """
    df = df.copy()
    if "return" not in df.columns:
        df["return"] = df["close"].pct_change()
    df["return_target"] = df["return"].shift(-lag)
    return df


# ============================================================================
# Model Training for Mutual Funds
# ============================================================================

def train_fund_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "ridge",
    ridge_alpha: float = 1.0,
) -> Tuple:
    """
    Train a single model on fund data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fund data with features and return_target
    feature_cols : List[str]
        List of feature column names
    model_name : str
        Model type: 'ridge', 'ols', 'elasticnet'
    ridge_alpha : float
        Alpha parameter for Ridge regression
    
    Returns:
    --------
    Tuple
        (trained_model, feature_cols)
    """
    X, y = prepare_fund_features(df, feature_cols)
    
    if X.shape[0] < 10:
        raise ValueError(f"Not enough training samples: {X.shape[0]}")
    
    if model_name == "ridge":
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    elif model_name == "ols":
        model = LinearRegression(fit_intercept=True)
    elif model_name == "elasticnet":
        model = ElasticNet(alpha=ridge_alpha * 0.1, l1_ratio=0.5, fit_intercept=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fit(X, y)
    return model, feature_cols


def predict_fund_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model,
) -> np.ndarray:
    """
    Generate predictions using a trained model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fund data with features
    feature_cols : List[str]
        List of feature column names
    model : sklearn model
        Trained model object
    
    Returns:
    --------
    np.ndarray
        Predictions
    """
    X, _ = prepare_fund_features(df, feature_cols)
    return model.predict(X)


def rolling_fund_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "ridge",
    train_size: int = 60,
    pred_horizon: int = 5,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """
    Rolling window backtest for a single fund.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fund data with features and return_target
    feature_cols : List[str]
        List of feature column names
    model_name : str
        Model type
    train_size : int
        Training window size
    pred_horizon : int
        Prediction horizon
    ridge_alpha : float
        Alpha for Ridge
    
    Returns:
    --------
    np.ndarray
        Array of predictions
    """
    preds = []
    n = len(df)
    
    idx = train_size
    while idx + pred_horizon <= n:
        # Train on [idx-train_size : idx]
        train_df = df.iloc[idx - train_size : idx].copy()
        
        try:
            model, _ = train_fund_model(
                train_df,
                feature_cols,
                model_name=model_name,
                ridge_alpha=ridge_alpha,
            )
        except Exception:
            idx += pred_horizon
            continue
        
        # Predict on [idx : idx + pred_horizon]
        pred_df = df.iloc[idx : idx + pred_horizon].copy()
        
        try:
            pred_vals = predict_fund_model(pred_df, feature_cols, model)
            preds.extend(pred_vals)
        except Exception:
            pass
        
        idx += pred_horizon
    
    return np.array(preds)


# ============================================================================
# Portfolio Metrics
# ============================================================================

def compute_portfolio_metrics(
    portfolio_df: pd.DataFrame,
    annual_days: float = 252.0,
) -> Dict[str, float]:
    """
    Compute Sharpe, CAGR, and Max Drawdown for a portfolio.
    
    Parameters:
    -----------
    portfolio_df : pd.DataFrame
        DataFrame with 'date' and 'portfolio_value' columns
    annual_days : float
        Number of trading days per year
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with 'sharpe', 'cagr', 'max_drawdown' keys
    """
    metrics = {
        "sharpe": np.nan,
        "cagr": np.nan,
        "max_drawdown": np.nan,
    }
    
    if portfolio_df is None or portfolio_df.empty:
        return metrics
    
    df = portfolio_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    
    if df.shape[0] < 2:
        return metrics
    
    pv = df.set_index("date")["portfolio_value"]
    
    # Sharpe Ratio
    returns = pv.pct_change().dropna()
    if len(returns) > 5 and returns.std() > 0:
        ann_factor = np.sqrt(annual_days / (len(returns) / len(df)))
        metrics["sharpe"] = (returns.mean() / returns.std()) * ann_factor
    
    # CAGR
    total_days = (pv.index[-1] - pv.index[0]).days
    if total_days >= 90 and pv.iloc[0] > 0:
        metrics["cagr"] = (pv.iloc[-1] / pv.iloc[0]) ** (365.25 / total_days) - 1.0
    
    # Max Drawdown
    if len(pv) >= 3:
        running_max = pv.cummax()
        drawdown = pv / running_max - 1.0
        metrics["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else np.nan
    
    return metrics


def compute_fund_sharpe(
    symbol_to_df: Dict[str, pd.DataFrame],
    annual_factor: float = 252.0,
    risk_free: float = 0.0,
) -> Dict[str, float]:
    """
    Compute Sharpe ratio for all funds.
    
    Parameters:
    -----------
    symbol_to_df : Dict[str, pd.DataFrame]
        Dictionary mapping fund symbols to DataFrames
    annual_factor : float
        Days per year
    risk_free : float
        Risk-free rate
    
    Returns:
    --------
    Dict[str, float]
        Dictionary mapping symbols to Sharpe ratios
    """
    sharpe_dict = {}
    
    for sym, df in symbol_to_df.items():
        try:
            if df is None or df.empty:
                sharpe_dict[sym] = np.nan
                continue
            
            # Pick price column
            price_col = None
            for col in ["close", "price", "Close", "Price", "adj_close", "Adj Close"]:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                price_col = numeric_cols[-1] if numeric_cols else None
            
            if price_col is None:
                sharpe_dict[sym] = np.nan
                continue
            
            price = df[price_col].astype(float)
            if not price.notna().any():
                sharpe_dict[sym] = np.nan
                continue
            
            returns = price.pct_change().dropna()
            if returns.empty or returns.std() == 0:
                sharpe_dict[sym] = np.nan
                continue
            
            sharpe = (returns.mean() / (returns.std() + 1e-12)) * (annual_factor ** 0.5)
            sharpe_dict[sym] = float(sharpe)
        
        except Exception:
            sharpe_dict[sym] = np.nan
    
    return sharpe_dict


# ============================================================================
# Portfolio Optimization Weights
# ============================================================================

def safe_pinv(mat: np.ndarray) -> np.ndarray:
    """Safe pseudo-inverse with regularization."""
    return np.linalg.pinv(mat + 1e-10 * np.eye(mat.shape[0]))


def compute_minvar_weights(Sigma: np.ndarray, enforce_nonneg: bool = True) -> np.ndarray:
    """
    Compute minimum variance portfolio weights.
    
    Parameters:
    -----------
    Sigma : np.ndarray
        Covariance matrix
    enforce_nonneg : bool
        Whether to enforce non-negative weights
    
    Returns:
    --------
    np.ndarray
        Portfolio weights
    """
    n = Sigma.shape[0]
    invS = safe_pinv(Sigma)
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


def compute_meanvar_weights(
    Sigma: np.ndarray,
    mu: np.ndarray,
    enforce_nonneg: bool = True,
    cap: float = 0.5,
) -> np.ndarray:
    """
    Compute mean-variance (max Sharpe) portfolio weights.
    
    Parameters:
    -----------
    Sigma : np.ndarray
        Covariance matrix
    mu : np.ndarray
        Expected returns
    enforce_nonneg : bool
        Whether to enforce non-negative weights
    cap : float
        Maximum weight cap if enforcing non-negative
    
    Returns:
    --------
    np.ndarray
        Portfolio weights
    """
    invS = safe_pinv(Sigma)
    w = invS @ mu
    
    if enforce_nonneg:
        w = np.maximum(w, 0)
        w = np.minimum(w, cap)
    
    s = w.sum()
    w = (w / s) if s > 0 else np.ones_like(w) / len(w)
    
    return w


def compute_riskparity_weights(vol: np.ndarray) -> np.ndarray:
    """
    Compute risk parity portfolio weights.
    
    Parameters:
    -----------
    vol : np.ndarray
        Volatility of each asset
    
    Returns:
    --------
    np.ndarray
        Portfolio weights
    """
    inv_vol = 1.0 / np.maximum(vol, 1e-12)
    w = inv_vol / inv_vol.sum()
    return w
