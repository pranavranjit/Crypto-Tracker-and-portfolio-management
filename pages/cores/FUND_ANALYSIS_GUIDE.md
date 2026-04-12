"""
Mutual Fund Analysis Integration Guide
=======================================

This module provides a complete toolkit for analyzing and backtesting
mutual fund portfolios using machine learning models and portfolio optimization.

Quick Start Examples
====================

1. BASIC FEATURE ENGINEERING
---------------------------

from pages.cores.fund_analysis import compute_fund_features
from pages.cores.reader import getSymbolToDf

# Load fund data
symbol_to_df = getSymbolToDf("week5_funds/stage_1_fund_data.csv")

# Compute features for a single fund
fund_data = symbol_to_df["FUND_NAME"]
price_series = fund_data["close"]
features_df = compute_fund_features(price_series)
print(features_df)  # r1, r5, r10, r20, vol20, mom20


2. SENTIMENT REGIME-BASED BACKTEST
----------------------------------

from pages.cores.fund_runner import xsltr_regime_backtest

# Build a momentum explorer backtest with sentiment adjustments
portfolio = xsltr_regime_backtest(
    symbol_to_df=symbol_to_df,
    symbols=["FUND1", "FUND2", "FUND3"],
    pred_horizon=5,
    initial_capital=10_000,
    rebalance_days=5,
    lookback_days=120,
    sentiment_regime="neutral",  # or "fear", "greed", "extreme_greed"
)

# Portfolio metrics
print(portfolio)  # date, portfolio_value


3. MULTI-MODEL ROTATION STRATEGY
--------------------------------

from pages.cores.fund_runner import multi_fund_rotation

portfolio_curves, sharpe_ratios = multi_fund_rotation(
    symbol_to_df=symbol_to_df,
    symbols=["FUND1", "FUND2", "FUND3"],
    model_name_list=["ridge", "ols", "elasticnet"],
    features=["r1", "r5", "r10", "r20", "vol20", "mom20"],
    train_size=60,
    pred_horizon=5,
    threshold=0.0,
    initial_capital=10_000,
)

for model, curve in portfolio_curves.items():
    sharpe = sharpe_ratios[model]
    print(f"{model}: Sharpe={sharpe:.3f}, Final Value={curve['portfolio_value'].iloc[-1]:.2f}")


4. PORTFOLIO OPTIMIZATION (MINIMUM VARIANCE)
-------------------------------------------

from pages.cores.fund_runner import build_price_panel
from pages.cores.fund_analysis import compute_minvar_weights
import numpy as np

# Build price panel
PX = build_price_panel(symbol_to_df, ["FUND1", "FUND2", "FUND3"])

# Compute returns and covariance
rets = PX.pct_change().dropna()
Sigma = np.cov(rets.values.T)

# Get minimum variance weights
weights = compute_minvar_weights(Sigma)
print(dict(zip(PX.columns, weights)))


5. PERFORMANCE METRICS
---------------------

from pages.cores.fund_analysis import compute_portfolio_metrics, compute_fund_sharpe

# Single portfolio metrics
metrics = compute_portfolio_metrics(portfolio_df, annual_days=252)
print(f"Sharpe: {metrics['sharpe']:.3f}")
print(f"CAGR: {metrics['cagr']*100:.2f}%")
print(f"Max DD: {metrics['max_drawdown']*100:.2f}%")

# Sharpe ratios for all funds
sharpe_dict = compute_fund_sharpe(symbol_to_df)
for fund, sharpe in sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{fund}: {sharpe:.3f}")


6. CROSS-SECTIONAL ZSCORE NORMALIZATION
---------------------------------------

from pages.cores.fund_analysis import cross_sectional_zscore, compute_fund_features
import pandas as pd

# At a given date, normalize features across all funds
features_at_date = {}
for sym, price_series in price_dict.items():
    features_at_date[sym] = compute_fund_features(price_series).iloc[-1]

features_xs = pd.DataFrame(features_at_date).T
features_normalized = cross_sectional_zscore(features_xs)
print(features_normalized)  # Z-scored features


Core Modules Reference
======================

pages.cores.fund_analysis
-------------------------
compute_fund_features(px, periods=[1,5,10,20])
    - Compute momentum and volatility features from price series
    
cross_sectional_zscore(df_xs)
    - Z-score normalize features across symbols
    
train_fund_model(df, feature_cols, model_name, ridge_alpha)
    - Train a single regression model on fund data
    
rolling_fund_backtest(df, feature_cols, model_name, train_size, pred_horizon, ridge_alpha)
    - Perform rolling window backtest for single fund
    
compute_portfolio_metrics(portfolio_df, annual_days)
    - Calculate Sharpe, CAGR, Max Drawdown
    
compute_fund_sharpe(symbol_to_df, annual_factor)
    - Get Sharpe ratios for all funds
    
compute_minvar_weights(Sigma, enforce_nonneg)
    - Compute minimum variance portfolio weights
    
compute_meanvar_weights(Sigma, mu, enforce_nonneg, cap)
    - Compute mean-variance (max Sharpe) weights
    
compute_riskparity_weights(vol)
    - Compute risk parity weights


pages.cores.fund_runner
------------------------
build_price_panel(symbol_to_df, symbols, price_col)
    - Create wide price panel from multiple funds
    
xsltr_regime_backtest(symbol_to_df, symbols, pred_horizon, ...)
    - Cross-sectional regression with sentiment regime adjustments
    
multi_fund_rotation(symbol_to_df, symbols, model_name_list, ...)
    - Multi-model rotation strategy


pages.cores.reader
-------------------
getSymbolToDf(path, threshold)
    - Load and preprocess fund data from CSV
    
add_lagged_return_target(df, lag)
    - Add return_target for supervised learning
    
get_price_column(df)
    - Detect price column name


Integration with Momentum_Explorer.py
======================================

Instead of defining custom functions, use the cores modules:

# OLD (in Momentum_Explorer.py):
def _get_features(px):
    r1 = px.pct_change(1)
    r5 = px.pct_change(5)
    ...
    return pd.DataFrame({...})

# NEW (using cores):
from pages.cores.fund_analysis import compute_fund_features

px = symbol_to_df[symbol]["close"]
features = compute_fund_features(px)


Feature Engineering Pipeline
=============================

1. Load data:
   symbol_to_df = getSymbolToDf(path)

2. For each fund, compute features:
   features = compute_fund_features(fund_prices)

3. Normalize cross-sectionally (at each date):
   features_xs = cross_sectional_zscore(features_xs)

4. Prepare for ML:
   X, y = prepare_fund_features(df, ["r1", "r5", "r10", "r20", "vol20", "mom20"])

5. Train and backtest:
   preds = rolling_fund_backtest(df, features, model_name, train_size, pred_horizon)

6. Evaluate:
   metrics = compute_portfolio_metrics(portfolio_df)


Portfolio Strategies
====================

Ridge/OLS/ElasticNet Rotation
    - Pick symbol with highest predicted return at each rebalance
    - Suitable for tactical allocation
    
Sentiment-Adjusted Ridge (XSLTR)
    - Adjust topk selection and exposure based on market regime
    - Fear: 1 symbol, 40% exposure
    - Greed: 5 symbols, 100% exposure
    - Extreme Greed: 6 symbols, 100% exposure
    - Neutral: 3 symbols, 70% exposure
    
Minimum Variance
    - Lowest portfolio volatility
    - Good for risk-averse investors
    
Mean-Variance
    - Maximize Sharpe ratio
    - Requires stable expected returns
    
Risk Parity
    - Inverse volatility weighting
    - Simple and robust


Hyperparameter Tuning
=====================

Model hyperparameters to experiment with:
- ridge_alpha: 0.1 to 10.0 (higher = more regularization)
- train_size: 30 to 250 (longer lookback)
- pred_horizon: 1 to 20 (target prediction period)
- rebalance_days: 1 to 20 (portfolio update frequency)
- lookback_days: 60 to 250 (feature training window)


Common Issues & Solutions
=========================

Issue: "Not enough training samples"
Solution: Reduce train_size or increase data quantity

Issue: "No usable price data found"
Solution: Check CSV has 'close', 'price', or numeric columns

Issue: "No rebalancing points available"
Solution: Increase rebalance_freq or provide more history

Issue: NaN values in predictions
Solution: Use compute_fund_sharpe to identify problematic funds
         or increase threshold parameter to filter low-confidence predictions


Testing Strategy
================

1. Test with single fund:
   - Verify features are computed correctly
   - Check model trains without errors
   
2. Test with 3-5 funds:
   - Verify rotation strategy picks symbols
   - Check portfolio values are reasonable
   
3. Backtest with full universe:
   - Performance across all available funds
   - Check metrics (Sharpe, CAGR, MaxDD)
   
4. Compare models:
   - Ridge vs OLS vs ElasticNet
   - Identify which works best for your data
   
5. Optimize hyperparameters:
   - Grid search over train_size, pred_horizon
   - Use walk-forward analysis to avoid overfitting
"""

# Quick reference: Sample workflow
if __name__ == "__main__":
    import pandas as pd
    from pages.cores.reader import getSymbolToDf
    from pages.cores.fund_analysis import compute_fund_sharpe, compute_portfolio_metrics
    from pages.cores.fund_runner import xsltr_regime_backtest
    
    # Load data
    data_path = "week5_funds/stage_1_fund_data.csv"
    symbol_to_df = getSymbolToDf(data_path, threshold=100)
    
    # Quick analysis
    sharpe_dict = compute_fund_sharpe(symbol_to_df)
    top_funds = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    symbols = [sym for sym, _ in top_funds]
    
    print(f"Backtesting top 5 funds: {symbols}")
    
    # Run sentiment-neutral backtest
    portfolio = xsltr_regime_backtest(
        symbol_to_df=symbol_to_df,
        symbols=symbols,
        pred_horizon=5,
        initial_capital=10_000,
        sentiment_regime="neutral",
    )
    
    # Evaluate
    metrics = compute_portfolio_metrics(portfolio)
    print(f"Sharpe: {metrics['sharpe']:.3f}")
    print(f"CAGR: {metrics['cagr']*100:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Final Value: {portfolio['portfolio_value'].iloc[-1]:.2f}")
