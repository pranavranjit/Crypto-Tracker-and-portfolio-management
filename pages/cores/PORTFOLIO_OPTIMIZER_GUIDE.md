# Portfolio Optimizer - Complete Guide

## Overview

The **Portfolio Optimizer** is an advanced tool designed to help you optimize your portfolio allocation based on historical returns and modern portfolio theory. It uses **Sharpe ratio maximization** to identify the optimal allocation of your investments while considering your risk tolerance and investment goals.

## Features

### 📊 1. Portfolio Input & Analysis
- **Manual Asset Entry**: Add individual assets with their current weights
- **CSV Import**: Bulk import portfolio data from spreadsheet
- **Automatic Data Fetching**: Retrieves historical price data from Yahoo Finance
- **Portfolio Analysis**: Calculates key performance metrics for your current portfolio

### 🎯 2. Optimization Results
- **Sharpe Ratio Optimization**: Maximizes risk-adjusted returns
- **Side-by-Side Comparison**: View current vs. optimal allocation
- **Performance Metrics**: Compare return, volatility, and Sharpe ratio
- **Visual Comparison**: Bar charts showing allocation changes

### 📈 3. Efficient Frontier
- **Interactive Visualization**: Plot all possible portfolio combinations
- **Optimal Portfolio Location**: See where your optimized portfolio sits
- **Minimum Variance Portfolio**: Find the lowest-risk allocation
- **Color-Coded Returns**: Visualize risk-return tradeoff

### 📋 4. Detailed Suggestions
- **Action Plan**: Clear buy/sell recommendations with amounts
- **Rebalancing Timeline**: Step-by-step implementation guide
- **Export Options**: Download suggestions as CSV or text report
- **Tax Consideration Notes**: Reminders about potential tax implications

## How to Use

### Step 1: Configure Parameters (Sidebar)
```
Investment Amount: Enter the total capital to invest
Risk-Free Rate: Annual risk-free rate (typically 2-5%)
Lookback Period: Historical data window (252 days = 1 year)
```

### Step 2: Add Assets
Choose between:
- **Manual Input**: Click "Add Asset" button and enter symbol and weight
- **CSV Upload**: Prepare a CSV with columns: `Symbol`, `Weight`

Supported symbols:
- Stocks: `AAPL`, `MSFT`, `SPY`, etc.
- ETFs: `VTI`, `BND`, `QQQ`, etc.

### Step 3: Analyze Current Portfolio
1. Click "Fetch Data & Analyze Portfolio"
2. System fetches historical data and calculates metrics
3. Review current performance:
   - Annual Return
   - Annual Volatility
   - Sharpe Ratio

### Step 4: Optimize Portfolio
1. Go to "Optimization Results" tab
2. Click "Optimize Portfolio"
3. System runs Sharpe ratio optimization
4. Compare metrics and allocations

### Step 5: Review Recommendations
1. Navigate to "Detailed Suggestions" tab
2. Review buy/increase recommendations
3. Review sell/reduce recommendations
4. Download the action plan

## Key Metrics Explained

### Annual Return
- Expected annual percentage return based on historical data
- Formula: `(1 + Daily Return)^252 - 1`
- Represents potential upside

### Annual Volatility
- Measure of risk or price fluctuation
- Formula: `Standard Deviation × √252`
- Higher volatility = higher risk

### Sharpe Ratio
- Risk-adjusted return metric
- Formula: `(Portfolio Return - Risk-Free Rate) / Portfolio Volatility`
- Higher Sharpe ratio = better risk-adjusted performance
- Used as the optimization objective

### Efficient Frontier
- Shows the best possible combinations of risk and return
- Every point represents a valid portfolio allocation
- Optimal portfolio sits where Sharpe ratio is maximized

## Mathematical Foundation

### Sharpe Ratio Optimization

The optimizer solves the following problem:

```
Maximize: (R_p - R_f) / σ_p

Subject to:
- Σ w_i = 1 (weights sum to 100%)
- 0 ≤ w_i ≤ 1 (no shorting, each weight between 0-100%)

Where:
R_p = Portfolio Return = Σ (w_i × R_i)
σ_p = Portfolio Volatility = √(w^T × Σ × w)
R_f = Risk-Free Rate
w = Weight vector
Σ = Covariance matrix of returns
```

### Covariance Matrix
- Captures relationships between asset returns
- Used to calculate portfolio volatility considering diversification benefits
- Calculated from historical daily returns

### Risk-Free Rate
- Rate of risk-free investment (typically government bonds)
- Used as benchmark to measure excess returns
- Default: 5% annually

## Best Practices

### 1. Data Considerations
- Use at least 1 year of historical data for reliable statistics
- Adjust lookback period based on market regime changes
- Rebalance quarterly or semi-annually

### 2. Implementation
- Don't rebalance too frequently (high trading costs)
- Consider tax implications before selling winners
- Implement changes gradually over days/weeks to reduce market impact
- Monitor holdings for major changes in correlation

### 3. Risk Management
- Don't over-concentrate in high-return assets
- Maintain minimum allocation to defensive assets
- Consider your actual investment timeline vs. historical period
- Account for transaction costs (usually 0.1-0.5% per trade)

### 4. Portfolio Constraints
- Current optimizer allows long-only positions (0-100% weights)
- Consider adding position limits (e.g., max 30% in any single asset)
- Ensure minimum allocations for regulatory/policy requirements

## Advanced Features

### Efficient Frontier Generation
- Creates 5,000 random portfolios
- Identifies frontier curve
- Shows Sharpe ratio distribution
- Helps understand risk-return space

### Minimum Variance Portfolio
- Identifies lowest-risk allocation
- May have lower returns than optimal portfolio
- Useful for conservative investors

### Correlation Analysis
- Shows relationships between assets
- High correlation = less diversification benefit
- Useful for understanding portfolio construction

## Troubleshooting

### Common Issues

**"Data not found for symbols"**
- Check symbol format (e.g., BTC-USD for bitcoin, not BTC)
- Ensure symbols are valid on Yahoo Finance
- Try different spelling variations

**"Weights sum to X%"**
- Normalize weights to 100% before optimizing
- Use the percentage inputs, not decimal

**"Optimization failed to converge"**
- Try adjusting lookback period
- Ensure at least 2 assets in portfolio
- Check for extreme correlation between assets

**"No data available"**
- Make sure to run "Fetch Data & Analyze Portfolio" first
- Wait for analysis to complete before optimizing

## Example Workflow

### Scenario: Example $100,000 Portfolio (Stocks/ETFs)

**Input:**
- SPY: 50%
- QQQ: 30%
- VTI: 20%
- Total Investment: $100,000
- Risk-Free Rate: 5%
- Lookback Period: 1 year (252 days)

**Current Portfolio:**
- Return: 8.2%
- Volatility: 12.3%
- Sharpe Ratio: 0.48

**Optimized Portfolio:**
- Return: 9.1%
- Volatility: 11.0%
- Sharpe Ratio: 0.62

**Recommendations:**
- SPY: Increase by $5,000 (from 50% to 55%)
- QQQ: Decrease by $3,000 (from 30% to 27%)
- VTI: Decrease by $2,000 (from 20% to 18%)

## API Reference

### PortfolioOptimizer Class

```python
from cores.portfolio_optimizer import PortfolioOptimizer

# Initialize
optimizer = PortfolioOptimizer(risk_free_rate=0.05, lookback_periods=252)

# Calculate metrics
metrics = optimizer.calculate_metrics(returns_df)

# Optimize to maximize Sharpe ratio
result = optimizer.optimize_portfolio(returns_df)

# Get current portfolio performance
perf = optimizer.current_portfolio_performance(weights, returns_df)

# Generate suggestions for rebalancing
suggestions = optimizer.generate_suggestions(current_weights, optimal_weights, 100000)

# Generate efficient frontier
returns, stds, sharpes = optimizer.efficient_frontier(returns_df, num_portfolios=5000)
```

## Parameters Reference

### Initialization Parameters
- `risk_free_rate` (float): Annual risk-free rate (default 0.05 = 5%)
- `lookback_periods` (int): Trading days for calculations (default 252 = 1 year)

### Configuration
- `Investment Amount`: Total capital to allocate ($1,000+)
- `Risk-Free Rate`: Annual yield on risk-free asset (0-10%)
- `Lookback Period`: Historical data to use (30-750 days)

## Limitations & Disclaimers

### Important Notes
1. **Past Performance**: Historical returns don't guarantee future results
2. **Volatility Changes**: Market conditions change, volatility fluctuates
3. **Correlation Breakdown**: Asset correlations change during market stress
4. **No Transaction Costs**: Recommendations don't include trading fees
5. **Tax Implications**: Selling may trigger capital gains taxes
6. **Market Impact**: Large trades may move prices
7. **Rebalancing Risk**: Frequent rebalancing increases costs
8. **Model Risk**: Optimization assumes normal distributions and historical relationships continue

### Required Disclaimers
This tool provides analysis for **informational purposes only**. It is **NOT financial advice**. Always consult with a qualified financial advisor before making investment decisions. Past returns do not guarantee future results.

## Version History

### v1.0 - Current
- Sharpe ratio optimization
- Efficient frontier visualization
- Actionable rebalancing suggestions
- CSV import/export
- Support for stocks, ETFs, and cryptocurrencies

## Technical Details

### Technology Stack
- **Framework**: Streamlit (web interface)
- **Optimization**: SciPy (SLSQP algorithm)
- **Data**: yfinance (historical data)
- **Visualization**: Plotly (interactive charts)
- **Language**: Python 3.8+

### File Structure
```
pages/
├── Portfolio_Optimizer.py          # Main Streamlit app
└── cores/
    ├── portfolio_optimizer.py      # Optimization algorithms
    └── reader.py                   # Data utilities
```

### Dependencies
- streamlit >= 1.49.1
- pandas >= 2.3.2
- numpy >= 2.3.3
- scipy >= 1.16.2
- yfinance >= 0.2.44
- plotly >= 6.3.0

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run app**: `streamlit run Momentum_Explorer.py`
3. **Navigate**: Click "Portfolio_Optimizer" in the sidebar
4. **Start optimizing**: Input your portfolio and explore

## Support & Feedback

For issues or feature requests, please contribute to the project repository.

---

**Last Updated**: February 2026
**Version**: 1.0
**Status**: Production Ready
