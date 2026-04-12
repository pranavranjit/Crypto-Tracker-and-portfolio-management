# Mutual Funds Cores Module - Quick Reference Cheat Sheet

## Import Statements

```python
# Data loading and preprocessing
from pages.cores.reader import getSymbolToDf, add_lagged_return_target, get_price_column

# Feature engineering and model training
from pages.cores.fund_analysis import (
    compute_fund_features,
    cross_sectional_zscore,
    train_fund_model,
    rolling_fund_backtest,
    compute_portfolio_metrics,
    compute_fund_sharpe,
    compute_minvar_weights,
    compute_meanvar_weights,
    compute_riskparity_weights,
)

# Portfolio strategies
from pages.cores.fund_runner import (
    build_price_panel,
    xsltr_regime_backtest,
    multi_fund_rotation,
)
```

## Common Workflows

### 1. Load and Analyze Data

```python
# Load data
symbol_to_df = getSymbolToDf("week5_funds/stage_1_fund_data.csv", threshold=100)

# Get Sharpe ratios for all funds
sharpe_dict = compute_fund_sharpe(symbol_to_df)
top_funds = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)[:5]
symbols = [sym for sym, _ in top_funds]
```

### 2. Single Fund Analysis

```python
# Get a fund's data
fund_data = symbol_to_df["FUND_NAME"]
price = fund_data["close"]

# Compute features
features = compute_fund_features(price)  # r1, r5, r10, r20, vol20, mom20

# Add target for ML
fund_data = add_lagged_return_target(fund_data, lag=1)

# Train Ridge model
model, features_cols = train_fund_model(
    fund_data,
    feature_cols=["r1", "r5", "r10", "r20", "vol20", "mom20"],
    model_name="ridge",
    ridge_alpha=1.0,
)
```

### 3. Build Price Panel for Multiple Funds

```python
# Create wide price matrix (date x symbol)
PX = build_price_panel(
    symbol_to_df,
    symbols=["FUND1", "FUND2", "FUND3"],
    price_col="close"
)
# Result: DataFrame with dates as index, symbols as columns
# Automatically drops rows with any NaN values
```

### 4. Momentum Backtest (Sentiment-Neutral)

```python
portfolio = xsltr_regime_backtest(
    symbol_to_df=symbol_to_df,
    symbols=symbols,
    pred_horizon=5,           # 5-day prediction
    initial_capital=10_000,
    rebalance_days=5,        # Rebalance every 5 days
    lookback_days=120,       # 120-day training window
    sentiment_regime="neutral",  # or "fear", "greed", "extreme_greed"
)

# Evaluate
metrics = compute_portfolio_metrics(portfolio)
print(f"Sharpe: {metrics['sharpe']:.3f}")
print(f"CAGR: {metrics['cagr']*100:.2f}%")
print(f"Max DD: {metrics['max_drawdown']*100:.2f}%")
```

### 5. Compare Multiple Models

```python
curves, sharpe_dict = multi_fund_rotation(
    symbol_to_df=symbol_to_df,
    symbols=symbols,
    model_name_list=["ridge", "ols", "elasticnet"],
    features=["r1", "r5", "r10", "r20", "vol20", "mom20"],
    train_size=60,
    pred_horizon=5,
    threshold=0.0,
    initial_capital=10_000,
)

for model, curve in curves.items():
    metrics = compute_portfolio_metrics(curve)
    print(f"{model}: Sharpe={metrics['sharpe']:.3f}")
```

### 6. Portfolio Optimization - Minimum Variance

```python
PX = build_price_panel(symbol_to_df, symbols)
returns = PX.pct_change().dropna()

# Compute covariance
Sigma = np.cov(returns.values.T)

# Get weights
weights = compute_minvar_weights(Sigma, enforce_nonneg=True)

# Apply to portfolio
portfolio_return = np.dot(weights, returns.mean())
portfolio_std = np.sqrt(weights @ Sigma @ weights.T)
sharpe = portfolio_return / portfolio_std * np.sqrt(252)
```

### 7. Portfolio Optimization - Mean-Variance

```python
Sigma = np.cov(returns.values.T)
mu = returns.mean().values

weights = compute_meanvar_weights(Sigma, mu, enforce_nonneg=True, cap=0.5)

# Portfolio metrics
portfolio_return = np.dot(weights, mu)
portfolio_std = np.sqrt(weights @ Sigma @ weights.T)
```

### 8. Portfolio Optimization - Risk Parity

```python
vol = returns.std().values
weights = compute_riskparity_weights(vol)

# Equal contribution to volatility
for i, (sym, w, v) in enumerate(zip(PX.columns, weights, vol)):
    contribution = w * v / (weights @ vol)
    print(f"{sym}: weight={w:.2%}, vol_contribution={contribution:.2%}")
```

## Feature Engineering Pipeline

```python
# Step 1: Compute base features
features = compute_fund_features(price)
# Output: DataFrame with columns [r1, r5, r10, r20, vol20, mom20]

# Step 2: Cross-sectional normalization (at each date)
# Normalize across symbols to account for market regime
features_xs = pd.DataFrame({sym: features.loc[date] for sym in symbols})
features_normalized = cross_sectional_zscore(features_xs)

# Step 3: Prepare for ML
X, y = prepare_fund_features(df, feature_cols)
# X: features, y: targets (next period returns)
```

## Model Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `ridge_alpha` | 0.01 - 10 | 1.0 | Higher = more shrinkage |
| `train_size` | 30 - 250 | 60 | Training window (days) |
| `pred_horizon` | 1 - 20 | 5 | Prediction horizon (days) |
| `rebalance_days` | 1 - 20 | 5 | Rebalancing frequency |
| `lookback_days` | 60 - 250 | 120 | Feature training window |
| `threshold` | 0 - 1 | 0.0 | Min prediction confidence |

## Sentiment Regimes

| Regime | topk | exposure | Use Case |
|--------|------|----------|----------|
| fear | 1 | 40% | Highly concentrated, reduced risk |
| neutral | 3 | 70% | Balanced allocation |
| greed | 5 | 100% | Diversified, full exposure |
| extreme_greed | 6 | 100% | Maximum diversification |

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Not enough training samples" | `train_size` too large | Reduce `train_size` or add more data |
| "No usable price data found" | Missing price column | Check CSV has 'close' or 'price' column |
| "No rebalancing points" | Insufficient history | Reduce `lookback_days` or get more data |
| "All selected models failed" | Bad feature combinations | Test with `["r1", "r5", "r10", "r20", "vol20", "mom20"]` |
| NaN in metrics | Invalid portfolio values | Check initial_capital > 0 and data quality |

## Optimization Tips

### 1. Feature Selection
```python
# Test different feature combinations
features_to_test = [
    ["r1", "r5", "r10", "r20"],        # Returns only
    ["vol20", "mom20"],                 # Volatility & momentum
    ["r1", "r5", "r10", "r20", "vol20", "mom20"],  # All (default)
]
```

### 2. Cross-Validation
```python
# Use rolling forecast (already in rolling_fund_backtest)
# Out-of-sample performance is realistic
```

### 3. Ensemble Approach
```python
# Combine multiple models
curves, sharpe = multi_fund_rotation(...)
# Take average or weighted combination of predictions
```

### 4. Regime Switching
```python
# Test different regimes based on market conditions
for regime in ["fear", "neutral", "greed", "extreme_greed"]:
    portfolio = xsltr_regime_backtest(..., sentiment_regime=regime)
    metrics = compute_portfolio_metrics(portfolio)
    print(f"{regime}: Sharpe={metrics['sharpe']:.3f}")
```

## Performance Benchmarks

### Expected Sharpe Ratios (Annual)
- Random: ~0.0
- Buy & Hold: 0.3 - 0.8
- Simple 60/40: 0.5 - 1.0
- **Good ML Strategy: 1.0 - 2.0+**
- Best practices: < 2.0 in real trading

### Common Portfolio Returns
- Market (benchmark): 8-10% annualized
- Min Variance: 5-8% (lower risk)
- Mean-Variance: 10-15% (if stable expected returns)
- Risk Parity: 8-12%

## Useful Utilities

### Get Portfolio Statistics
```python
metrics = compute_portfolio_metrics(portfolio_df)
# Returns: {'sharpe': float, 'cagr': float, 'max_drawdown': float}
```

### Detect Price Column
```python
price_col = get_price_column(fund_df)
# Automatically finds 'close', 'price', 'adj_close', etc.
```

### Prepare Target Column
```python
df = add_lagged_return_target(df, lag=1)
# Adds 'return_target' column for supervised learning
```

## Full Example (10 Lines)

```python
from pages.cores.reader import getSymbolToDf
from pages.cores.fund_runner import xsltr_regime_backtest
from pages.cores.fund_analysis import compute_portfolio_metrics

symbol_to_df = getSymbolToDf("week5_funds/stage_1_fund_data.csv")
symbols = list(symbol_to_df.keys())[:5]  # Top 5 funds

portfolio = xsltr_regime_backtest(symbol_to_df, symbols, initial_capital=10_000)
metrics = compute_portfolio_metrics(portfolio)

print(f"Sharpe: {metrics['sharpe']:.3f}")
print(f"Final Value: ${portfolio['portfolio_value'].iloc[-1]:.2f}")
```

---

**See `FUND_ANALYSIS_GUIDE.md` for detailed examples and `fund_analysis_example.py` for a complete script.**
