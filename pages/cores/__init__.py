"""
Mutual Fund Analysis & Backtesting Cores Module
===============================================

This package provides comprehensive tools for analyzing, modeling, and backtesting
mutual fund portfolios with machine learning and portfolio optimization.

Submodules:
-----------
- fund_analysis: Core feature engineering, model training, and portfolio metrics
- fund_runner: Portfolio rotation strategies and sentiment-adjusted backtests
- reader: Data loading, preprocessing, and price column detection
- commons: Feature and model lists
- ml: Machine learning model training and prediction
- runners: Multi-symbol rotation strategies

Quick Import Examples:
----------------------

from pages.cores.fund_analysis import (
    compute_fund_features,
    cross_sectional_zscore,
    compute_portfolio_metrics,
    compute_fund_sharpe,
    compute_minvar_weights,
    compute_meanvar_weights,
    compute_riskparity_weights,
)

from pages.cores.fund_runner import (
    xsltr_regime_backtest,
    multi_fund_rotation,
    build_price_panel,
)

from pages.cores.reader import (
    getSymbolToDf,
    add_lagged_return_target,
    get_price_column,
)

from pages.cores.commons import FEATURES, MODELS

See FUND_ANALYSIS_GUIDE.md for detailed usage examples and integration patterns.
"""

__version__ = "1.0.0"
__all__ = [
    "fund_analysis",
    "fund_runner",
    "reader",
    "commons",
    "ml",
    "runners",
]

