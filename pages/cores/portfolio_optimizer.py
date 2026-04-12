"""
Portfolio Optimization Module
Implements Sharpe ratio optimization and portfolio analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization using Sharpe ratio maximization
    Supports both cryptocurrency and mutual fund portfolios
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.05,
                 lookback_periods: int = 252):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
            lookback_periods: Trading days to lookback (default 252 = 1 year)
        """
        self.risk_free_rate = risk_free_rate
        self.lookback_periods = lookback_periods
    
    def calculate_metrics(self, 
                         returns_df: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics from returns
        
        Args:
            returns_df: DataFrame with daily returns (symbols as columns)
            
        Returns:
            Dictionary with mean returns, covariance, and other metrics
        """
        # Calculate annualized metrics (252 trading days)
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        daily_returns = returns_df.mean()
        daily_std = returns_df.std()
        std_returns = returns_df.std() * np.sqrt(252)
        
        return {
            'mean_returns': mean_returns,
            'std_returns': std_returns,
            'cov_matrix': cov_matrix,
            'corr_matrix': returns_df.corr(),
            'daily_returns': daily_returns,
            'daily_std': daily_std,
        }
    
    def portfolio_performance(self, 
                             weights: np.ndarray, 
                             mean_returns: np.ndarray, 
                             cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Args:
            weights: Portfolio weights
            mean_returns: Mean returns for assets
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def negative_sharpe(self, 
                       weights: np.ndarray, 
                       mean_returns: np.ndarray, 
                       cov_matrix: np.ndarray) -> float:
        """
        Negative Sharpe ratio for minimization
        """
        return -self.portfolio_performance(weights, mean_returns, cov_matrix)[2]
    
    def optimize_portfolio(self, 
                          returns_df: pd.DataFrame,
                          initial_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Optimize portfolio weights to maximize Sharpe ratio
        
        Args:
            returns_df: DataFrame with daily returns (symbols as columns)
            initial_weights: Starting weights (default: equal weights)
            
        Returns:
            Dictionary with optimized weights and performance metrics
        """
        metrics = self.calculate_metrics(returns_df)
        mean_returns = metrics['mean_returns'].values
        cov_matrix = metrics['cov_matrix'].values
        
        num_assets = len(returns_df.columns)
        
        # Initial guess (equal weights)
        if initial_weights is None:
            x0 = np.array([1/num_assets] * num_assets)
        else:
            x0 = initial_weights
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Optimization
        result = minimize(
            self.negative_sharpe,
            x0,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        opt_weights = result.x
        opt_return, opt_std, opt_sharpe = self.portfolio_performance(
            opt_weights, mean_returns, cov_matrix
        )
        
        return {
            'weights': opt_weights,
            'return': opt_return,
            'volatility': opt_std,
            'sharpe_ratio': opt_sharpe,
            'symbols': list(returns_df.columns),
            'success': result.success,
            'message': result.message
        }
    
    def current_portfolio_performance(self, 
                                     current_weights: Dict[str, float],
                                     returns_df: pd.DataFrame) -> Dict:
        """
        Analyze current portfolio performance
        
        Args:
            current_weights: Dictionary of {symbol: weight}
            returns_df: DataFrame with returns
            
        Returns:
            Portfolio performance metrics
        """
        # Align weights with returns columns
        symbols = returns_df.columns.tolist()
        weights = np.array([current_weights.get(s, 0) for s in symbols])
        
        # Ensure weights sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        metrics = self.calculate_metrics(returns_df)
        mean_returns = metrics['mean_returns'].values
        cov_matrix = metrics['cov_matrix'].values
        
        ret, std, sharpe = self.portfolio_performance(weights, mean_returns, cov_matrix)
        
        return {
            'weights': dict(zip(symbols, weights)),
            'return': ret,
            'volatility': std,
            'sharpe_ratio': sharpe,
            'mean_returns': dict(zip(symbols, mean_returns))
        }
    
    def generate_suggestions(self,
                           current_weights: Dict[str, float],
                           optimal_weights: Dict[str, float],
                           total_investment: float) -> pd.DataFrame:
        """
        Generate actionable suggestions for portfolio rebalancing
        
        Args:
            current_weights: Current portfolio allocation
            optimal_weights: Optimal allocation from optimization
            total_investment: Total amount to invest
            
        Returns:
            DataFrame with change recommendations
        """
        suggestions = []
        
        for symbol in optimal_weights.keys():
            current_weight = current_weights.get(symbol, 0)
            optimal_weight = optimal_weights.get(symbol, 0)
            weight_change = optimal_weight - current_weight
            
            current_amount = current_weight * total_investment
            optimal_amount = optimal_weight * total_investment
            amount_change = optimal_amount - current_amount
            
            if abs(weight_change) > 0.001:  # Only show meaningful changes
                suggestions.append({
                    'Symbol': symbol,
                    'Current Weight %': current_weight * 100,
                    'Optimal Weight %': optimal_weight * 100,
                    'Weight Change %': weight_change * 100,
                    'Current Amount': current_amount,
                    'Optimal Amount': optimal_amount,
                    'Amount Change': amount_change,
                    'Action': 'Buy' if amount_change > 0 else 'Sell/Reduce'
                })
        
        df = pd.DataFrame(suggestions)
        if not df.empty:
            df = df.sort_values('Amount Change', key=abs, ascending=False)
        
        return df
    
    def efficient_frontier(self,
                          returns_df: pd.DataFrame,
                          num_portfolios: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate efficient frontier by creating random portfolios
        
        Args:
            returns_df: DataFrame with daily returns
            num_portfolios: Number of random portfolios to generate
            
        Returns:
            Tuple of (returns, volatilities, sharpe_ratios)
        """
        metrics = self.calculate_metrics(returns_df)
        mean_returns = metrics['mean_returns'].values
        cov_matrix = metrics['cov_matrix'].values
        
        num_assets = len(returns_df.columns)
        results = np.zeros((3, num_portfolios))
        np.random.seed(42)
        
        for i in range(num_portfolios):
            # Random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate metrics
            port_return, port_std, sharpe = self.portfolio_performance(
                weights, mean_returns, cov_matrix
            )
            
            results[0,i] = port_return
            results[1,i] = port_std
            results[2,i] = sharpe
        
        return results[0], results[1], results[2]
    
    def min_variance_portfolio(self, 
                              returns_df: pd.DataFrame) -> Dict:
        """
        Find minimum variance portfolio
        
        Args:
            returns_df: DataFrame with daily returns
            
        Returns:
            Minimum variance portfolio weights and metrics
        """
        metrics = self.calculate_metrics(returns_df)
        mean_returns = metrics['mean_returns'].values
        cov_matrix = metrics['cov_matrix'].values
        
        num_assets = len(returns_df.columns)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        x0 = np.array([1/num_assets] * num_assets)
        
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        mv_weights = result.x
        mv_return, mv_std, mv_sharpe = self.portfolio_performance(
            mv_weights, mean_returns, cov_matrix
        )
        
        return {
            'weights': mv_weights,
            'return': mv_return,
            'volatility': mv_std,
            'sharpe_ratio': mv_sharpe,
            'symbols': list(returns_df.columns)
        }
