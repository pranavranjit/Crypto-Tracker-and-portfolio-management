# Mutual Fund Analysis Cores Module - README

## 📌 What Is This?

A complete, production-ready toolkit for analyzing, modeling, and backtesting mutual fund portfolios using **machine learning** and **portfolio optimization**.

Extracted and refactored from `Momentum_Explorer.py` into reusable, well-documented modules.

---

## ⚡ Quick Start (2 Minutes)

### Installation
```python
# Copy fund_analysis.py and fund_runner.py to your project
# No additional dependencies (uses pandas, numpy, sklearn)
```

### Minimal Example
```python
from pages.cores.reader import getSymbolToDf
from pages.cores.fund_runner import xsltr_regime_backtest
from pages.cores.fund_analysis import compute_portfolio_metrics

# Load data
symbol_to_df = getSymbolToDf("week5_funds/stage_1_fund_data.csv")

# Run backtest
portfolio = xsltr_regime_backtest(
    symbol_to_df=symbol_to_df,
    symbols=list(symbol_to_df.keys())[:5],
    initial_capital=10_000
)

# Evaluate
metrics = compute_portfolio_metrics(portfolio)
print(f"Sharpe: {metrics['sharpe']:.3f}, CAGR: {metrics['cagr']*100:.2f}%")
```

---

## 📚 Documentation Map

```
START HERE ↓

1. README.md (this file)
   ↓ Want quick examples? →  QUICK_REFERENCE.md
   ↓ Want full details? →    FUND_ANALYSIS_GUIDE.md
   ↓ Want architecture? →    INSTALLATION_AND_SUMMARY.md
   ↓ Want file overview? →   FILE_STRUCTURE.md
   ↓ Want working code? →    fund_analysis_example.py

KEY FILES ↓

fund_analysis.py ............... Core functions (feature eng, ML, metrics)
fund_runner.py ................ Portfolio strategies (momentum, regime-based)
reader.py ..................... Data loading & preprocessing [UPDATED]
fund_analysis_example.py ....... Complete working example (run it!)
```

---

## 🎯 What Problems Does This Solve?

### Problem 1: Code Duplication
**Before**: Custom functions scattered in `Momentum_Explorer.py`  
**After**: Import from cores modules, one source of truth

### Problem 2: Hard to Extend
**Before**: To add new strategy, modify Momentum_Explorer  
**After**: Use cores components to build new analyses quickly

### Problem 3: Not Reusable
**Before**: Features only available in Momentum_Explorer  
**After**: Use in notebooks, scripts, or new projects

### Problem 4: Hard to Maintain
**Before**: Bug fixes affect multiple implementations  
**After**: Fix once in cores, benefits entire codebase

---

## 🚀 What Can You Do?

### 1. Model Individual Funds
```python
from pages.cores.fund_analysis import compute_fund_features, train_fund_model

features = compute_fund_features(price_series)
model, _ = train_fund_model(df, ["r1", "r5", "r10", "r20", "vol20", "mom20"])
```

### 2. Compare ML Models
```python
from pages.cores.fund_runner import multi_fund_rotation

curves, sharpe = multi_fund_rotation(
    symbol_to_df, symbols,
    model_name_list=["ridge", "ols", "elasticnet"],
    features=["r1", "r5", "r10", "r20", "vol20", "mom20"]
)
```

### 3. Sentiment-Adjusted Backtests
```python
from pages.cores.fund_runner import xsltr_regime_backtest

portfolio = xsltr_regime_backtest(
    symbol_to_df, symbols,
    sentiment_regime="greed"  # or "fear", "extreme_greed", "neutral"
)
```

### 4. Optimize Portfolios
```python
from pages.cores.fund_analysis import (
    compute_minvar_weights,
    compute_meanvar_weights,
    compute_riskparity_weights
)

weights_minvar = compute_minvar_weights(covariance_matrix)
weights_maxsharpe = compute_meanvar_weights(cov, expected_returns)
weights_riskparity = compute_riskparity_weights(volatility)
```

### 5. Analyze Individual Funds
```python
from pages.cores.fund_analysis import compute_fund_sharpe

sharpe_dict = compute_fund_sharpe(symbol_to_df)
for fund, sharpe in sorted(sharpe_dict.items()):
    print(f"{fund}: {sharpe:.3f}")
```

---

## 📖 Documentation Files

### `QUICK_REFERENCE.md` ⭐ (Start here for coding)
- Import statements
- 8 common workflows with code
- Parameter tuning guide
- Troubleshooting table
- **Best for**: Developers actively coding

### `FUND_ANALYSIS_GUIDE.md` (Deep dive)
- 6 complete step-by-step examples
- Feature engineering pipeline explanation
- Portfolio strategies overview
- Common issues & solutions
- Testing strategy
- **Best for**: Understanding the system

### `INSTALLATION_AND_SUMMARY.md` (Architecture)
- Before/after comparison
- How it fits into Momentum_Explorer
- Design decisions
- Integration steps
- **Best for**: Project managers, architects

### `FILE_STRUCTURE.md` (Overview)
- Detailed file inventory
- Dependencies and relationships
- Lines of code summary
- Next steps
- **Best for**: New team members

### `fund_analysis_example.py` (Working code)
- Complete runnable example
- 5 different analyses:
  1. Individual fund analysis
  2. Momentum backtest
  3. ML model comparison
  4. Portfolio optimization
  5. Performance report
- **Best for**: Copy-paste starting point

---

## 🔧 Core Modules

### `fund_analysis.py` (Core algorithms)

**Feature Engineering**
- `compute_fund_features()` → r1, r5, r10, r20, vol20, mom20
- `cross_sectional_zscore()` → Normalize across funds

**Model Training**
- `train_fund_model()` → Ridge/OLS/ElasticNet
- `rolling_fund_backtest()` → Rolling window backtests

**Metrics**
- `compute_portfolio_metrics()` → Sharpe, CAGR, Max DD
- `compute_fund_sharpe()` → Sharpe for all funds

**Portfolio Optimization**
- `compute_minvar_weights()` → Minimum variance
- `compute_meanvar_weights()` → Max Sharpe
- `compute_riskparity_weights()` → Risk parity

### `fund_runner.py` (Portfolio strategies)

- `build_price_panel()` → Create wide price matrix
- `xsltr_regime_backtest()` → Sentiment-adjusted momentum
- `multi_fund_rotation()` → Multi-model rotation

### `reader.py` (Data loading)

**New**
- `add_lagged_return_target()` → Prepare training targets
- `get_price_column()` → Smart price detection

**Existing**
- `getSymbolToDf()` → Load from CSV
- `addReturns()` → Compute returns
- `add_ohlc_features()` → Technical indicators

---

## 🎓 Learning Path

### Level 1: Beginner (30 min)
1. Read this README
2. Look at `QUICK_REFERENCE.md` examples
3. Run `fund_analysis_example.py`

### Level 2: Intermediate (2 hours)
1. Read `FUND_ANALYSIS_GUIDE.md`
2. Try modifying `fund_analysis_example.py`
3. Create your own Jupyter notebook using cores

### Level 3: Advanced (1 day)
1. Read `INSTALLATION_AND_SUMMARY.md` for architecture
2. Read individual function docstrings in code
3. Extend cores with your own functions
4. Refactor Momentum_Explorer to use cores

---

## 📊 Common Use Cases

### Use Case 1: Compare 3 Models
```python
curves, sharpe = multi_fund_rotation(
    symbol_to_df, symbols,
    model_name_list=["ridge", "ols", "elasticnet"],
    features=["r1", "r5", "r10", "r20", "vol20", "mom20"]
)
# ridge: Sharpe=0.857
# ols: Sharpe=0.623
# elasticnet: Sharpe=0.741
```

### Use Case 2: Test Different Market Regimes
```python
for regime in ["fear", "neutral", "greed", "extreme_greed"]:
    p = xsltr_regime_backtest(..., sentiment_regime=regime)
    m = compute_portfolio_metrics(p)
    print(f"{regime}: Sharpe={m['sharpe']:.3f}")
```

### Use Case 3: Build Better Portfolio
```python
PX = build_price_panel(symbol_to_df, symbols)
returns = PX.pct_change().dropna()
Sigma = np.cov(returns.values.T)

# Min variance
w_minvar = compute_minvar_weights(Sigma)

# Max Sharpe
mu = returns.mean().values
w_maxsharpe = compute_meanvar_weights(Sigma, mu)

# Risk parity
w_riskparity = compute_riskparity_weights(returns.std().values)
```

### Use Case 4: Analyze Fund Universe
```python
sharpe_dict = compute_fund_sharpe(symbol_to_df)
top_5 = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 funds by Sharpe:")
for name, sharpe in top_5:
    print(f"  {name}: {sharpe:.3f}")
```

---

## ✅ Features

✅ **Ridge/OLS/ElasticNet Regression**  
✅ **Rolling Window Backtests**  
✅ **Sentiment-Adjusted Strategies**  
✅ **Portfolio Optimization** (MinVar, MeanVar, RiskParity)  
✅ **Comprehensive Metrics** (Sharpe, CAGR, Max Drawdown)  
✅ **Cross-Sectional Feature Normalization**  
✅ **Automatic Price Column Detection**  
✅ **Error Handling & Validation**  
✅ **Extensive Documentation**  
✅ **Working Examples**  

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pages'"
**Solution**: Run scripts from the project root directory

### "Not enough training samples"
**Solution**: Reduce `train_size` or ensure you have enough data

### "No usable price data found"
**Solution**: Check CSV has 'close', 'price', or numeric columns

### "All models failed"
**Solution**: Try simpler features or different `train_size`

See `QUICK_REFERENCE.md` for more troubleshooting tips.

---

## 📈 Performance Tips

1. **Feature Selection**: Test different feature combinations
2. **Hyperparameter Tuning**: Grid search over `train_size`, `pred_horizon`
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Out-of-Sample Testing**: Use rolling window naturally does this
5. **Regime Switching**: Test all 4 sentiment regimes

---

## 🔗 Integration with Momentum_Explorer.py

Can replace custom functions with imports:

```python
# OLD (Momentum_Explorer.py)
def _get_features(px):
    r1 = px.pct_change(1)
    ...

# NEW
from pages.cores.fund_analysis import compute_fund_features
features = compute_fund_features(px)
```

See `INSTALLATION_AND_SUMMARY.md` for full integration guide.

---

## 📞 Need Help?

1. **For quick syntax**: Check `QUICK_REFERENCE.md`
2. **For examples**: See `FUND_ANALYSIS_GUIDE.md`
3. **For understanding**: Read `INSTALLATION_AND_SUMMARY.md`
4. **For working code**: Run `fund_analysis_example.py`
5. **For specifics**: Read docstrings in source files

---

## 🎯 Next Steps

1. **Try it**: Run `fund_analysis_example.py`
2. **Explore**: Modify the example with your own data
3. **Learn**: Read the guides based on your interest
4. **Build**: Create your own analyses using cores
5. **Extend**: Add new features or strategies

---

## 📝 Version Info

- **Status**: Production Ready
- **Version**: 1.0.0
- **Created**: February 2026
- **Tested**: With mutual fund data

---

## 📚 File Quick Links

| Document | Purpose | Read Time |
|----------|---------|-----------|
| `README.md` | You are here | 5 min |
| `QUICK_REFERENCE.md` | Cheat sheet for coding | 10 min |
| `FUND_ANALYSIS_GUIDE.md` | Complete guide | 30 min |
| `INSTALLATION_AND_SUMMARY.md` | Architecture details | 20 min |
| `FILE_STRUCTURE.md` | File overview | 15 min |
| `fund_analysis_example.py` | Working example | 10 min to run |

---

**All set? Start with `QUICK_REFERENCE.md` →**

Or if you prefer to learn by doing: Run `fund_analysis_example.py` first!
