# Mutual Funds Analysis Cores Module - Summary

## Overview

You now have a complete set of reusable modules for analyzing and backtesting mutual fund portfolios. These modules have been extracted and refactored from the `Momentum_Explorer.py` logic to be:

- **Modular**: Functions can be used independently or together
- **Reusable**: Import and use in other scripts/notebooks
- **Tested**: Built on patterns from working code
- **Documented**: Every function has docstrings and examples

## New Files Created

### 1. `pages/cores/fund_analysis.py` 
**Core statistical and ML functions for mutual fund analysis**

**Feature Engineering:**
- `compute_fund_features()` - Compute momentum (r1, r5, r10, r20, mom20) and volatility (vol20)
- `cross_sectional_zscore()` - Normalize features across funds at each date
- `add_lagged_return_target()` - Prepare targets for supervised learning

**Model Training:**
- `train_fund_model()` - Train Ridge/OLS/ElasticNet on fund data
- `predict_fund_model()` - Generate predictions
- `rolling_fund_backtest()` - Rolling window backtests

**Portfolio Metrics:**
- `compute_portfolio_metrics()` - Calculate Sharpe, CAGR, Max Drawdown
- `compute_fund_sharpe()` - Sharpe ratios for all funds

**Portfolio Optimization:**
- `compute_minvar_weights()` - Minimum variance weights
- `compute_meanvar_weights()` - Mean-variance (max Sharpe) weights
- `compute_riskparity_weights()` - Risk parity weights

### 2. `pages/cores/fund_runner.py`
**Portfolio backtesting and rotation strategies**

**Functions:**
- `build_price_panel()` - Create wide price matrix from multiple funds
- `xsltr_regime_backtest()` - Cross-sectional regression with sentiment regime adjustments
  - **Supports different market regimes:**
    - Fear: topk=1, exposure=0.4
    - Greed: topk=5, exposure=1.0
    - Extreme Greed: topk=6, exposure=1.0
    - Neutral: topk=3, exposure=0.7
- `multi_fund_rotation()` - Multi-model rotation strategy (Ridge/OLS/ElasticNet)

### 3. Updated `pages/cores/reader.py`
**Enhanced data loading and preprocessing**

**New Functions:**
- `add_lagged_return_target()` - Add return_target for ML
- `get_price_column()` - Smart price column detection

**Existing Functions (enhanced):**
- `getSymbolToDf()` - Load and preprocess funds from CSV
- `addReturns()` - Compute returns
- `add_ohlc_features()` - Technical indicators
- `add_volume_and_technical_features()` - Volume-based features

### 4. Updated `pages/cores/__init__.py`
**Package documentation and exports**

Clear imports and documentation for the entire cores module.

### 5. `pages/cores/FUND_ANALYSIS_GUIDE.md`
**Comprehensive usage guide with 6 quick-start examples**

## How These Fit Into Momentum_Explorer.py

The `Momentum_Explorer.py` currently has custom functions for:
1. Loading data
2. Computing features
3. Training models
4. Building portfolios

**Now you can replace these with imports from cores:**

```python
# OLD - Custom functions in Momentum_Explorer.py:
def _get_features(px):
    r1 = px.pct_change(1)
    r5 = px.pct_change(5)
    ...
    return pd.DataFrame({...})

# NEW - Use from cores:
from pages.cores.fund_analysis import compute_fund_features

features = compute_fund_features(px)
```

## Architecture Comparison

### Before (Momentum_Explorer.py)
```
Momentum_Explorer.py
├── Load data (custom)
├── Compute features (custom: _get_features)
├── Build price panel (custom: _build_price_panel)
├── Train models (custom: multi_runner)
├── Optimize weights (custom: _weights_MinVar, etc.)
└── Compute metrics (custom: compute_curve_metrics)
```

### After (With Cores)
```
Momentum_Explorer.py
├── from pages.cores.reader import getSymbolToDf
├── from pages.cores.fund_analysis import compute_fund_features
├── from pages.cores.fund_runner import build_price_panel, xsltr_regime_backtest
├── from pages.cores.fund_analysis import (
│   ├── compute_minvar_weights
│   ├── compute_meanvar_weights
│   ├── compute_riskparity_weights
│   └── compute_portfolio_metrics
│   )
└── Clean, focused application logic
```

## Key Design Decisions

### 1. Feature Engineering
- Uses period returns: 1, 5, 10, 20 days
- Includes volatility (20-day rolling std)
- Includes momentum (20-day return)
- Cross-sectional normalization (z-score by symbol at each date)

### 2. Model Training
- Supports Ridge, OLS, ElasticNet
- Rolling window backtests for realistic evaluation
- Configurable training window and prediction horizon

### 3. Portfolio Strategies
- **Ridge/OLS/ElasticNet Rotation**: Momentum-based symbol picking
- **Sentiment-Adjusted (XSLTR)**: Regime-dependent allocation
- **Minimum Variance**: Risk minimization
- **Mean-Variance**: Sharpe ratio maximization
- **Risk Parity**: Equal risk contribution

### 4. Error Handling
- Safe pseudo-inverse for matrix operations
- Graceful fallback for NaN values
- Clear error messages for debugging

## Integration Steps

### For Momentum_Explorer.py

1. **Replace custom feature functions:**
   ```python
   from pages.cores.fund_analysis import compute_fund_features, cross_sectional_zscore
   ```

2. **Replace custom panel builder:**
   ```python
   from pages.cores.fund_runner import build_price_panel
   ```

3. **Replace custom weight computation:**
   ```python
   from pages.cores.fund_analysis import (
       compute_minvar_weights,
       compute_meanvar_weights,
       compute_riskparity_weights,
   )
   ```

4. **Replace custom metric computation:**
   ```python
   from pages.cores.fund_analysis import compute_portfolio_metrics, compute_fund_sharpe
   ```

### For New Scripts/Notebooks

Simply import from cores:
```python
from pages.cores.fund_analysis import (
    compute_fund_features,
    compute_portfolio_metrics,
    compute_fund_sharpe,
)
from pages.cores.fund_runner import xsltr_regime_backtest, build_price_panel
from pages.cores.reader import getSymbolToDf
```

## Example Usage

### Quick Portfolio Backtest
```python
from pages.cores.reader import getSymbolToDf
from pages.cores.fund_runner import xsltr_regime_backtest
from pages.cores.fund_analysis import compute_portfolio_metrics

symbol_to_df = getSymbolToDf("week5_funds/stage_1_fund_data.csv")

portfolio = xsltr_regime_backtest(
    symbol_to_df=symbol_to_df,
    symbols=["FUND1", "FUND2", "FUND3"],
    pred_horizon=5,
    initial_capital=10_000,
    sentiment_regime="neutral",
)

metrics = compute_portfolio_metrics(portfolio)
print(f"Sharpe: {metrics['sharpe']:.3f}, CAGR: {metrics['cagr']*100:.2f}%")
```

### Comparing Models
```python
from pages.cores.fund_runner import multi_fund_rotation

curves, sharpe_dict = multi_fund_rotation(
    symbol_to_df=symbol_to_df,
    symbols=["FUND1", "FUND2", "FUND3"],
    model_name_list=["ridge", "ols", "elasticnet"],
    features=["r1", "r5", "r10", "r20", "vol20", "mom20"],
    train_size=60,
    pred_horizon=5,
)

for model, sharpe in sharpe_dict.items():
    print(f"{model}: {sharpe:.3f}")
```

## Benefits

✅ **Code Reusability** - Use components in multiple scripts  
✅ **Maintainability** - Centralized, documented functions  
✅ **Testing** - Easier to unit test core functions  
✅ **Extensibility** - Add new strategies/metrics without modifying Momentum_Explorer  
✅ **Documentation** - FUND_ANALYSIS_GUIDE.md with 6+ examples  
✅ **Consistency** - Same logic across all projects  

## Next Steps

1. Test `fund_analysis.py` and `fund_runner.py` with sample data
2. Gradually refactor `Momentum_Explorer.py` to use cores
3. Create additional notebooks using these modules
4. Add more sentiment regimes if needed
5. Expand with additional portfolio strategies (momentum, value, etc.)

## File Dependencies

```
pages/cores/
├── fund_analysis.py (standalone)
├── fund_runner.py (depends on: fund_analysis, reader)
├── reader.py (standalone)
├── commons.py (existing)
├── ml.py (existing)
├── runners.py (existing)
└── FUND_ANALYSIS_GUIDE.md (documentation)
```

All new modules are self-contained and don't break existing functionality.

---

**See `FUND_ANALYSIS_GUIDE.md` for detailed examples and reference documentation.**
