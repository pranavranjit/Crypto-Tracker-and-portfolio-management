#!/usr/bin/env python3
"""
Complete Mutual Fund Analysis Example
======================================

This script demonstrates a complete workflow using the cores modules:
1. Load fund data
2. Analyze individual fund performance
3. Run multi-model rotation backtest
4. Compare with portfolio optimization strategies
5. Generate performance report

Usage:
------
python fund_analysis_example.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Import from cores modules
from pages.cores.reader import getSymbolToDf, get_price_column
from pages.cores.fund_analysis import (
    compute_fund_features,
    compute_fund_sharpe,
    compute_portfolio_metrics,
    compute_minvar_weights,
    compute_meanvar_weights,
    compute_riskparity_weights,
)
from pages.cores.fund_runner import (
    build_price_panel,
    xsltr_regime_backtest,
    multi_fund_rotation,
)


def load_fund_data(csv_path: str, threshold: int = 100) -> Dict[str, pd.DataFrame]:
    """Load and preprocess mutual fund data."""
    print(f"Loading fund data from {csv_path}...")
    symbol_to_df = getSymbolToDf(csv_path, threshold=threshold)
    print(f"  ✓ Loaded {len(symbol_to_df)} funds\n")
    return symbol_to_df


def analyze_individual_funds(symbol_to_df: Dict[str, pd.DataFrame], top_n: int = 10):
    """Analyze individual fund performance and select top performers."""
    print(f"Analyzing individual fund performance...")
    
    sharpe_dict = compute_fund_sharpe(symbol_to_df)
    sharpe_df = (
        pd.Series(sharpe_dict)
        .sort_values(ascending=False)
        .rename("Sharpe Ratio")
        .to_frame()
    )
    
    top_funds = sharpe_df.head(top_n).index.tolist()
    
    print(f"  Top {top_n} funds by Sharpe ratio:")
    for i, (fund, sharpe) in enumerate(sharpe_df.head(top_n).iterrows(), 1):
        print(f"    {i:2d}. {fund:30s} → {sharpe['Sharpe Ratio']:7.3f}")
    print()
    
    return top_funds


def run_momentum_backtest(
    symbol_to_df: Dict[str, pd.DataFrame],
    symbols: list,
    pred_horizon: int = 5,
    initial_capital: float = 10_000,
) -> Dict:
    """Run sentiment-neutral momentum backtest."""
    print(f"Running momentum backtest ({pred_horizon}-day horizon)...")
    
    portfolio = xsltr_regime_backtest(
        symbol_to_df=symbol_to_df,
        symbols=symbols,
        pred_horizon=pred_horizon,
        initial_capital=initial_capital,
        rebalance_days=5,
        lookback_days=120,
        sentiment_regime="neutral",
    )
    
    metrics = compute_portfolio_metrics(portfolio)
    
    print(f"  Momentum Strategy Results:")
    print(f"    Sharpe Ratio:    {metrics['sharpe']:7.3f}")
    print(f"    CAGR:            {metrics['cagr']*100:7.2f}%")
    print(f"    Max Drawdown:    {metrics['max_drawdown']*100:7.2f}%")
    final_value = portfolio["portfolio_value"].iloc[-1]
    print(f"    Final Value:     ${final_value:10.2f}\n")
    
    return {"portfolio": portfolio, "metrics": metrics}


def run_model_comparison(
    symbol_to_df: Dict[str, pd.DataFrame],
    symbols: list,
    initial_capital: float = 10_000,
) -> Dict:
    """Compare Ridge, OLS, and ElasticNet models."""
    print(f"Comparing ML models (Ridge, OLS, ElasticNet)...")
    
    portfolio_curves, sharpe_ratios = multi_fund_rotation(
        symbol_to_df=symbol_to_df,
        symbols=symbols,
        model_name_list=["ridge", "ols", "elasticnet"],
        features=["r1", "r5", "r10", "r20", "vol20", "mom20"],
        train_size=60,
        pred_horizon=5,
        threshold=0.0,
        initial_capital=initial_capital,
    )
    
    results = {}
    for model_name in ["ridge", "ols", "elasticnet"]:
        if model_name in portfolio_curves:
            curve = portfolio_curves[model_name]
            metrics = compute_portfolio_metrics(curve)
            final_value = curve["portfolio_value"].iloc[-1]
            
            print(f"  {model_name.upper():12s} → Sharpe: {metrics['sharpe']:7.3f}, "
                  f"Final: ${final_value:10.2f}")
            
            results[model_name] = {
                "portfolio": curve,
                "metrics": metrics,
                "sharpe": sharpe_ratios.get(model_name, np.nan),
            }
    print()
    
    return results


def run_portfolio_optimization(
    symbol_to_df: Dict[str, pd.DataFrame],
    symbols: list,
    initial_capital: float = 10_000,
    rebalance_freq: str = "W",
) -> Dict:
    """Compare portfolio optimization strategies."""
    print(f"Running portfolio optimization strategies...")
    
    # Build price panel
    PX = build_price_panel(symbol_to_df, symbols)
    returns = PX.pct_change().dropna()
    
    results = {}
    strategies = {
        "MinVar": "Minimum Variance",
        "MeanVar": "Mean-Variance",
        "RiskParity": "Risk Parity",
    }
    
    for key, weight_func, label in [
        ("MinVar", compute_minvar_weights, "Minimum Variance"),
        ("MeanVar", lambda S, mu: compute_meanvar_weights(S, mu), "Mean-Variance"),
        ("RiskParity", lambda S, mu: compute_riskparity_weights(returns.std().values), "Risk Parity"),
    ]:
        try:
            # Simple rebalance at month ends
            dates = returns.index
            portfolio_val = initial_capital
            curve = []
            
            # Monthly rebalance
            monthly_dates = returns.resample("M").last().index
            
            for i, rebal_date in enumerate(monthly_dates[:-1]):
                next_rebal = monthly_dates[i + 1]
                
                # Window for covariance calculation
                window_start = max(0, (dates.get_loc(rebal_date) - 60))
                window_end = dates.get_loc(rebal_date)
                
                window_returns = returns.iloc[window_start:window_end]
                
                # Compute weights
                Sigma = np.cov(window_returns.values.T)
                mu = window_returns.mean().values if key == "MeanVar" else None
                
                if key == "MinVar":
                    weights = compute_minvar_weights(Sigma)
                elif key == "MeanVar":
                    weights = compute_meanvar_weights(Sigma, mu)
                else:  # RiskParity
                    weights = compute_riskparity_weights(returns.std().values)
                
                # Return during holding period
                holding_returns = returns.loc[rebal_date:next_rebal]
                period_ret = np.dot(weights, holding_returns.mean()) * len(holding_returns)
                
                portfolio_val *= (1.0 + period_ret)
                curve.append((next_rebal, portfolio_val))
            
            # Create DataFrame
            portfolio_df = pd.DataFrame(curve, columns=["date", "portfolio_value"])
            metrics = compute_portfolio_metrics(portfolio_df)
            final_value = portfolio_df["portfolio_value"].iloc[-1]
            
            print(f"  {label:20s} → Sharpe: {metrics['sharpe']:7.3f}, "
                  f"Final: ${final_value:10.2f}")
            
            results[key] = {
                "portfolio": portfolio_df,
                "metrics": metrics,
                "weights": weights,
            }
        
        except Exception as e:
            print(f"  {label:20s} → Failed: {str(e)[:50]}")
    
    print()
    return results


def generate_report(
    momentum_results: Dict,
    model_results: Dict,
    port_opt_results: Dict,
):
    """Generate comparison report."""
    print("=" * 80)
    print("MUTUAL FUND PORTFOLIO ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Momentum strategy
    print("MOMENTUM STRATEGY (Sentiment-Neutral)")
    print("-" * 80)
    metrics = momentum_results["metrics"]
    print(f"Sharpe Ratio:      {metrics['sharpe']:7.3f}")
    print(f"CAGR:              {metrics['cagr']*100:7.2f}%")
    print(f"Max Drawdown:      {metrics['max_drawdown']*100:7.2f}%")
    print()
    
    # ML models
    print("MACHINE LEARNING MODELS")
    print("-" * 80)
    for model_name, results in model_results.items():
        metrics = results["metrics"]
        print(f"{model_name.upper():12s} | "
              f"Sharpe: {metrics['sharpe']:7.3f} | "
              f"CAGR: {metrics['cagr']*100:7.2f}% | "
              f"Max DD: {metrics['max_drawdown']*100:7.2f}%")
    print()
    
    # Portfolio optimization
    print("PORTFOLIO OPTIMIZATION STRATEGIES")
    print("-" * 80)
    for strat_name, results in port_opt_results.items():
        metrics = results["metrics"]
        print(f"{strat_name:20s} | "
              f"Sharpe: {metrics['sharpe']:7.3f} | "
              f"CAGR: {metrics['cagr']*100:7.2f}% | "
              f"Max DD: {metrics['max_drawdown']*100:7.2f}%")
    print()
    
    # Best performer
    print("BEST PERFORMERS")
    print("-" * 80)
    all_results = {
        "Momentum": momentum_results["metrics"],
        **{f"ML-{k}": v["metrics"] for k, v in model_results.items()},
        **{f"PO-{k}": v["metrics"] for k, v in port_opt_results.items()},
    }
    
    sharpe_sorted = sorted(
        all_results.items(),
        key=lambda x: x[1]["sharpe"],
        reverse=True,
    )
    
    print("By Sharpe Ratio:")
    for i, (name, metrics) in enumerate(sharpe_sorted[:5], 1):
        if np.isfinite(metrics["sharpe"]):
            print(f"  {i}. {name:25s} → {metrics['sharpe']:7.3f}")
    print()
    
    print("=" * 80)


def main():
    """Main execution."""
    # Configuration
    csv_path = "week5_funds/stage_1_fund_data.csv"
    initial_capital = 10_000
    top_n_funds = 5
    
    # Check if data exists
    if not Path(csv_path).exists():
        print(f"Error: Data file not found at {csv_path}")
        print("Please ensure the CSV file exists in the correct location.")
        return
    
    # Step 1: Load data
    symbol_to_df = load_fund_data(csv_path)
    
    # Step 2: Analyze individual funds
    top_funds = analyze_individual_funds(symbol_to_df, top_n=top_n_funds)
    
    # Step 3: Momentum backtest
    momentum_results = run_momentum_backtest(
        symbol_to_df,
        top_funds,
        pred_horizon=5,
        initial_capital=initial_capital,
    )
    
    # Step 4: Model comparison
    model_results = run_model_comparison(
        symbol_to_df,
        top_funds,
        initial_capital=initial_capital,
    )
    
    # Step 5: Portfolio optimization
    port_opt_results = run_portfolio_optimization(
        symbol_to_df,
        top_funds,
        initial_capital=initial_capital,
    )
    
    # Step 6: Generate report
    generate_report(momentum_results, model_results, port_opt_results)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
