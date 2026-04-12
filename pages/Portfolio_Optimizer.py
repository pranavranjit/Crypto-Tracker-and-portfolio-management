"""
Portfolio Optimizer - Streamlit App
Interactive portfolio optimization with Sharpe ratio analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple

from pages.cores.portfolio_optimizer import PortfolioOptimizer

# Configuration
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("🎯 Portfolio Optimizer - Sharpe Ratio Optimization")
st.markdown("""
Optimize your portfolio allocation based on historical returns and risk metrics.
Get personalized suggestions to maximize your Sharpe ratio for your investment amount.
""")

# Sidebar configuration
st.sidebar.header("Portfolio Configuration")
investment_amount = st.sidebar.number_input(
    "Total Investment Amount ($)",
    min_value=1000,
    value=100000,
    step=1000,
    help="Standard amount to invest across all assets"
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%) - Annual",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    help="Used to calculate Sharpe ratio"
) / 100

lookback_days = st.sidebar.slider(
    "Lookback Period (Days)",
    min_value=30,
    max_value=750,
    value=252,
    step=30,
    help="Historical data for calculations"
)

# Initialize optimizer
optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate, lookback_periods=lookback_days)

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Input & Analysis",
    "🎯 Optimization Results",
    "📈 Efficient Frontier",
    "📋 Detailed Suggestions"
])

# ==================== TAB 1: Portfolio Input & Analysis ====================
with tab1:
    st.header("Step 1: Define Your Current Portfolio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Assets to Your Portfolio")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["Manual Input", "Load from CSV"],
            horizontal=True,
            help="Choose how to input your portfolio"
        )
        
        if data_source == "Manual Input":
            # Manual portfolio entry
            if 'portfolio_assets' not in st.session_state:
                st.session_state.portfolio_assets = {}
            
            col_symbol, col_weight, col_add = st.columns([2, 2, 1])
            
            with col_symbol:
                new_symbol = st.text_input(
                    "Asset Symbol (e.g., BTC, ETH, SPY)",
                    key="new_symbol_input"
                )
            
            with col_weight:
                new_weight = st.number_input(
                    "Weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    key="new_weight_input"
                )
            
            with col_add:
                st.write("")  # Spacer
                if st.button("➕ Add Asset", use_container_width=True):
                    if new_symbol:
                        st.session_state.portfolio_assets[new_symbol.upper()] = new_weight / 100
                        st.success(f"Added {new_symbol.upper()}")
                        st.rerun()
            
            # Display current portfolio
            if st.session_state.portfolio_assets:
                st.subheader("Current Portfolio Assets")
                portfolio_df = pd.DataFrame([
                    {'Symbol': k, 'Weight %': v*100} 
                    for k, v in st.session_state.portfolio_assets.items()
                ])
                
                # Check if weights sum to ~100%
                total_weight = portfolio_df['Weight %'].sum()
                col_display, col_clear = st.columns([3, 1])
                
                with col_display:
                    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
                    
                    # Weight validation
                    if abs(total_weight - 100) > 1:
                        st.warning(f"⚠️ Weights sum to {total_weight:.1f}% - Normalize them for optimization")
                    else:
                        st.success(f"✅ Weights sum to {total_weight:.1f}%")
                
                with col_clear:
                    st.write("")  # Spacer
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.portfolio_assets = {}
                        st.rerun()
        
        else:  # CSV upload
            uploaded_file = st.file_uploader(
                "Upload CSV with columns: Symbol, Weight",
                type=['csv']
            )
            
            if uploaded_file:
                try:
                    csv_df = pd.read_csv(uploaded_file)
                    if 'Symbol' in csv_df.columns and 'Weight' in csv_df.columns:
                        st.session_state.portfolio_assets = dict(
                            zip(csv_df['Symbol'].values, csv_df['Weight'].values / 100)
                        )
                        st.success("Portfolio loaded from CSV")
                        st.dataframe(csv_df, use_container_width=True, hide_index=True)
                    else:
                        st.error("CSV must have 'Symbol' and 'Weight' columns")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
    
    with col2:
        st.subheader("Investment Summary")
        if st.session_state.portfolio_assets:
            total_assets = len(st.session_state.portfolio_assets)
            total_weight = sum(st.session_state.portfolio_assets.values())
            
            st.metric("Number of Assets", total_assets)
            st.metric("Total Weight", f"{total_weight*100:.1f}%")
            st.metric("Investment Amount", f"${investment_amount:,.0f}")
        else:
            st.info("Add assets to see summary")
    
    # Analyze current portfolio
    if st.session_state.portfolio_assets:
        st.divider()
        st.subheader("Step 2: Analyze Current Portfolio")
        
        if st.button("📊 Fetch Data & Analyze Portfolio", key="analyze_btn"):
            with st.spinner("Fetching historical data and calculating metrics..."):
                try:
                    # Fetch data
                    symbols = list(st.session_state.portfolio_assets.keys())
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_days + 30)
                    
                    # Download data
                    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
                    if isinstance(data, pd.Series):
                        data = pd.DataFrame(data)
                    
                    # Calculate daily returns
                    close_prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                    returns_df = close_prices.pct_change().dropna()
                    
                    # Ensure symbols exist in returns
                    available_symbols = [s for s in symbols if s in returns_df.columns]
                    if len(available_symbols) < len(symbols):
                        missing = [s for s in symbols if s not in returns_df.columns]
                        st.warning(f"⚠️ Data not found for: {', '.join(missing)}")
                    
                    returns_df = returns_df[available_symbols]
                    
                    # Update session state with fetched data
                    st.session_state.returns_df = returns_df
                    st.session_state.symbols = available_symbols
                    
                    # Calculate current portfolio performance
                    current_weights = {
                        s: st.session_state.portfolio_assets.get(s, 0) 
                        for s in available_symbols
                    }
                    
                    perf = optimizer.current_portfolio_performance(current_weights, returns_df)
                    st.session_state.current_performance = perf
                    
                    st.success("✅ Data loaded and analyzed!")
                    
                    # Display current portfolio metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Annual Return",
                            f"{perf['return']*100:.2f}%",
                            help="Annualized portfolio return"
                        )
                    
                    with col2:
                        st.metric(
                            "Annual Volatility",
                            f"{perf['volatility']*100:.2f}%",
                            help="Annualized standard deviation"
                        )
                    
                    with col3:
                        st.metric(
                            "Sharpe Ratio",
                            f"{perf['sharpe_ratio']:.3f}",
                            help="Risk-adjusted return"
                        )
                    
                    with col4:
                        st.metric(
                            "Data Points",
                            len(returns_df),
                            help="Days of historical data used"
                        )
                    
                    # Asset-wise metrics
                    st.subheader("Individual Asset Metrics")
                    metrics_df = pd.DataFrame({
                        'Symbol': list(perf['mean_returns'].keys()),
                        'Annual Return %': [v*100 for v in perf['mean_returns'].values()],
                        'Current Weight %': [current_weights[s]*100 for s in perf['mean_returns'].keys()]
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error analyzing portfolio: {str(e)}")
                    st.info("Make sure all symbols are valid (e.g., BTC-USD, ETH-USD for crypto, or SPY, QQQ for stocks)")
    else:
        st.info("👈 Add assets to your portfolio using the input section")

# ==================== TAB 2: Optimization Results ====================
with tab2:
    st.header("Optimization Results")
    
    if 'returns_df' in st.session_state and 'current_performance' in st.session_state:
        if st.button("🚀 Optimize Portfolio", key="optimize_btn"):
            with st.spinner("Running Sharpe ratio optimization..."):
                try:
                    returns_df = st.session_state.returns_df
                    
                    # Run optimization
                    opt_result = optimizer.optimize_portfolio(returns_df)
                    st.session_state.optimal_result = opt_result
                    
                    st.success("✅ Optimization complete!")
                    
                    # Create comparison
                    col1, col2, col3 = st.columns(3)
                    
                    current = st.session_state.current_performance
                    optimal = opt_result
                    
                    with col1:
                        st.subheader("📊 Current Portfolio")
                        st.metric("Return", f"{current['return']*100:.2f}%")
                        st.metric("Volatility", f"{current['volatility']*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{current['sharpe_ratio']:.3f}")
                    
                    with col2:
                        st.subheader("🎯 Optimal Portfolio")
                        st.metric("Return", f"{optimal['return']*100:.2f}%", 
                                 delta=f"{(optimal['return']-current['return'])*100:.2f}%")
                        st.metric("Volatility", f"{optimal['volatility']*100:.2f}%",
                                 delta=f"{(optimal['volatility']-current['volatility'])*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{optimal['sharpe_ratio']:.3f}",
                                 delta=f"{(optimal['sharpe_ratio']-current['sharpe_ratio']):.3f}")
                    
                    with col3:
                        st.subheader("📈 Improvement")
                        improvement_return = optimal['return'] - current['return']
                        improvement_sharpe = optimal['sharpe_ratio'] - current['sharpe_ratio']
                        
                        st.info(f"""
                        **Sharpe Ratio Improvement**\n
                        {improvement_sharpe/current['sharpe_ratio']*100:.1f}%
                        
                        **Risk Reduction**\n
                        {(optimal['volatility']-current['volatility'])/current['volatility']*100:.1f}%
                        """)
                    
                    # Allocation comparison
                    st.divider()
                    st.subheader("Allocation Comparison")
                    
                    current_weights = {
                        s: st.session_state.portfolio_assets.get(s, 0)
                        for s in st.session_state.symbols
                    }
                    
                    comparison_df = pd.DataFrame({
                        'Symbol': opt_result['symbols'],
                        'Current Weight %': [current_weights.get(s, 0)*100 for s in opt_result['symbols']],
                        'Optimal Weight %': [w*100 for w in opt_result['weights']],
                    })
                    comparison_df['Change %'] = comparison_df['Optimal Weight %'] - comparison_df['Current Weight %']
                    comparison_df = comparison_df.sort_values('Change %', key=abs, ascending=False)
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Current',
                        x=comparison_df['Symbol'],
                        y=comparison_df['Current Weight %'],
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Optimal',
                        x=comparison_df['Symbol'],
                        y=comparison_df['Optimal Weight %'],
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Allocation: Current vs Optimal",
                        barmode='group',
                        xaxis_title="Symbol",
                        yaxis_title="Weight (%)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Optimization error: {str(e)}")
    
    elif 'returns_df' not in st.session_state:
        st.info("👈 First, analyze your portfolio in the 'Portfolio Input & Analysis' tab")
    else:
        st.info("No data available. Please analyze your portfolio first.")

# ==================== TAB 3: Efficient Frontier ====================
with tab3:
    st.header("Efficient Frontier Analysis")
    st.markdown("""
    The efficient frontier shows the relationship between risk and return 
    for all possible portfolio combinations.
    """)
    
    if 'returns_df' in st.session_state:
        if st.button("📈 Generate Efficient Frontier", key="frontier_btn"):
            with st.spinner("Generating efficient frontier..."):
                try:
                    returns_df = st.session_state.returns_df
                    
                    # Generate frontier
                    returns, stds, sharpes = optimizer.efficient_frontier(returns_df, num_portfolios=5000)
                    
                    # Min variance portfolio
                    min_var = optimizer.min_variance_portfolio(returns_df)
                    
                    # Get current and optimal
                    current = st.session_state.current_performance
                    optimal = st.session_state.optimal_result if 'optimal_result' in st.session_state else None
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Scatter plot of random portfolios
                    fig.add_trace(go.Scatter(
                        x=stds*100, y=returns*100,
                        mode='markers',
                        name='Random Portfolios',
                        marker=dict(
                            color=sharpes,
                            colorscale='Viridis',
                            size=5,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        text=[''] * len(stds),
                        hovertemplate='<b>Risk: %{x:.2f}%</b><br>Return: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # Current portfolio
                    fig.add_trace(go.Scatter(
                        x=[current['volatility']*100],
                        y=[current['return']*100],
                        mode='markers+text',
                        name='Current Portfolio',
                        marker=dict(size=15, color='red', symbol='star'),
                        text=['Current'],
                        textposition='top center',
                        hovertemplate='<b>Current Portfolio</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # Optimal portfolio
                    if optimal:
                        fig.add_trace(go.Scatter(
                            x=[optimal['volatility']*100],
                            y=[optimal['return']*100],
                            mode='markers+text',
                            name='Optimal Portfolio',
                            marker=dict(size=15, color='green', symbol='star'),
                            text=['Optimal'],
                            textposition='top center',
                            hovertemplate='<b>Optimal Portfolio</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                    
                    # Min variance
                    fig.add_trace(go.Scatter(
                        x=[min_var['volatility']*100],
                        y=[min_var['return']*100],
                        mode='markers+text',
                        name='Min Variance',
                        marker=dict(size=12, color='orange', symbol='diamond'),
                        text=['Min Var'],
                        textposition='top center',
                        hovertemplate='<b>Minimum Variance</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Efficient Frontier with Portfolio Positions",
                        xaxis_title="Risk (Annual Volatility %)",
                        yaxis_title="Expected Return (%)",
                        height=600,
                        hovermode='closest',
                        width=1000
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Minimum Variance Portfolio")
                        st.metric("Return", f"{min_var['return']*100:.2f}%")
                        st.metric("Volatility", f"{min_var['volatility']*100:.2f}%")
                        st.metric("Sharpe Ratio", f"{min_var['sharpe_ratio']:.3f}")
                    
                    with col2:
                        st.subheader("Frontier Statistics")
                        st.metric("Max Return", f"{returns.max()*100:.2f}%")
                        st.metric("Min Volatility", f"{stds.min()*100:.2f}%")
                        st.metric("Max Sharpe", f"{sharpes.max():.3f}")
                    
                    with col3:
                        st.subheader("Risk-Return Range")
                        st.metric("Return Range", f"{(returns.max()-returns.min())*100:.2f}%")
                        st.metric("Volatility Range", f"{(stds.max()-stds.min())*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error generating frontier: {str(e)}")
    else:
        st.info("👈 First, analyze your portfolio in the 'Portfolio Input & Analysis' tab")

# ==================== TAB 4: Detailed Suggestions ====================
with tab4:
    st.header("Detailed Rebalancing Suggestions")
    st.markdown("""
    Actionable recommendations for moving from your current portfolio to the optimal allocation.
    """)
    
    if 'optimal_result' in st.session_state and 'current_performance' in st.session_state:
        try:
            current_weights = {
                s: st.session_state.portfolio_assets.get(s, 0)
                for s in st.session_state.symbols
            }
            
            optimal_weights = dict(zip(
                st.session_state.optimal_result['symbols'],
                st.session_state.optimal_result['weights']
            ))
            
            suggestions_df = optimizer.generate_suggestions(
                current_weights,
                optimal_weights,
                investment_amount
            )
            
            if not suggestions_df.empty:
                # Summary
                col1, col2, col3 = st.columns(3)
                
                buy_suggestions = suggestions_df[suggestions_df['Amount Change'] > 0]
                sell_suggestions = suggestions_df[suggestions_df['Amount Change'] < 0]
                
                with col1:
                    st.metric("Assets to Buy/Increase", len(buy_suggestions))
                
                with col2:
                    st.metric("Assets to Sell/Reduce", len(sell_suggestions))
                
                with col3:
                    total_rebalance = suggestions_df['Amount Change'].abs().sum()
                    st.metric("Total Rebalancing Needed", f"${total_rebalance:,.0f}")
                
                st.divider()
                
                # Detailed suggestions
                st.subheader("📋 Rebalancing Recommendations")
                
                if not buy_suggestions.empty:
                    st.subheader("✅ BUY / INCREASE")
                    buy_display = buy_suggestions[[
                        'Symbol', 'Current Weight %', 'Optimal Weight %', 
                        'Weight Change %', 'Current Amount', 'Optimal Amount', 'Amount Change'
                    ]].copy()
                    buy_display['Current Amount'] = buy_display['Current Amount'].apply(lambda x: f"${x:,.0f}")
                    buy_display['Optimal Amount'] = buy_display['Optimal Amount'].apply(lambda x: f"${x:,.0f}")
                    buy_display['Amount Change'] = buy_display['Amount Change'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(buy_display, use_container_width=True, hide_index=True)
                
                if not sell_suggestions.empty:
                    st.subheader("❌ SELL / REDUCE")
                    sell_display = sell_suggestions[[
                        'Symbol', 'Current Weight %', 'Optimal Weight %', 
                        'Weight Change %', 'Current Amount', 'Optimal Amount', 'Amount Change'
                    ]].copy()
                    sell_display['Current Amount'] = sell_display['Current Amount'].apply(lambda x: f"${x:,.0f}")
                    sell_display['Optimal Amount'] = sell_display['Optimal Amount'].apply(lambda x: f"${x:,.0f}")
                    sell_display['Amount Change'] = sell_display['Amount Change'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(sell_display, use_container_width=True, hide_index=True)
                
                # Action plan
                st.divider()
                st.subheader("📝 Action Plan")
                
                st.markdown("""
                ### Implementation Steps:
                
                1. **Review Recommendations**: Examine all buy/sell suggestions above
                2. **Consider Taxes**: Selling may trigger capital gains taxes
                3. **Transaction Costs**: Factor in trading commissions
                4. **Timing**: Consider market conditions before executing large trades
                5. **Gradual Implementation**: Consider rebalancing over time to reduce market impact
                6. **Monitor**: Follow up monthly or quarterly to maintain target allocation
                """)
                
                # Export suggestions
                col1, col2 = st.columns(2)
                with col1:
                    csv = suggestions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Suggestions (CSV)",
                        data=csv,
                        file_name="portfolio_suggestions.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create Excel-like summary
                    summary_text = f"""
PORTFOLIO OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

INVESTMENT AMOUNT: ${investment_amount:,.0f}
RISK-FREE RATE: {risk_free_rate*100:.1f}%
LOOKBACK PERIOD: {lookback_days} days

PERFORMANCE COMPARISON:
Current Portfolio:
  - Return: {st.session_state.current_performance['return']*100:.2f}%
  - Volatility: {st.session_state.current_performance['volatility']*100:.2f}%
  - Sharpe Ratio: {st.session_state.current_performance['sharpe_ratio']:.3f}

Optimal Portfolio:
  - Return: {st.session_state.optimal_result['return']*100:.2f}%
  - Volatility: {st.session_state.optimal_result['volatility']*100:.2f}%
  - Sharpe Ratio: {st.session_state.optimal_result['sharpe_ratio']:.3f}

REBALANCING NEEDED:
Total Requirement: ${suggestions_df['Amount Change'].abs().sum():,.0f}
{len(buy_suggestions)} assets to buy/increase
{len(sell_suggestions)} assets to sell/reduce
                    """
                    st.download_button(
                        label="📄 Download Report (TXT)",
                        data=summary_text,
                        file_name="portfolio_report.txt",
                        mime="text/plain"
                    )
            else:
                st.success("✅ Your portfolio is already well-optimized! Only minor adjustments suggested.")
        
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
    else:
        st.info("👈 First optimize your portfolio in the 'Optimization Results' tab")

# Footer
st.divider()


st.markdown("<div style='text-align: center; color: gray;'>Portfolio Optimizer v1.0 | Analytics Dashboard</div>", unsafe_allow_html=True)
