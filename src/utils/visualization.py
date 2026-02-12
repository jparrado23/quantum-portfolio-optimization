"""
Visualization utilities for portfolio optimization results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
sns.set_style("darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')


def plot_price_history(
    prices: pd.DataFrame,
    figsize: tuple = (14, 6),
    title: str = "Historical Price Data"
):
    """
    Plot historical price data for all assets.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize to 100 at start for comparison
    normalized = (prices / prices.iloc[0] * 100)
    
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Price (Base=100)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_returns_distribution(
    returns: pd.DataFrame,
    figsize: tuple = (14, 10)
):
    """
    Plot return distributions for all assets.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    figsize : tuple
        Figure size
    """
    n_assets = len(returns.columns)
    n_cols = 3
    n_rows = (n_assets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(returns.columns):
        ax = axes[i]
        returns[col].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'{col} Returns', fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Frequency')
        ax.axvline(returns[col].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns[col].mean():.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_assets, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    figsize: tuple = (12, 10),
    title: str = "Asset Correlation Matrix"
):
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_efficient_frontier(
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
    sharpe_ratios: np.ndarray,
    optimal_portfolio: Optional[dict] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot efficient frontier with color-coded Sharpe ratios.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Array of portfolio returns
    portfolio_volatilities : np.ndarray
        Array of portfolio volatilities
    sharpe_ratios : np.ndarray
        Array of Sharpe ratios
    optimal_portfolio : dict, optional
        Dict with 'return', 'volatility', 'sharpe' for optimal portfolio
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with Sharpe ratio color coding
    scatter = ax.scatter(
        portfolio_volatilities,
        portfolio_returns,
        c=sharpe_ratios,
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=12)
    
    # Mark optimal portfolio if provided
    if optimal_portfolio:
        ax.scatter(
            optimal_portfolio['volatility'],
            optimal_portfolio['return'],
            color='red',
            marker='*',
            s=500,
            edgecolors='black',
            linewidth=2,
            label=f"Optimal (Sharpe={optimal_portfolio['sharpe']:.3f})",
            zorder=5
        )
        ax.legend(fontsize=12)
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_portfolio_weights(
    weights: pd.Series,
    title: str = "Portfolio Allocation",
    figsize: tuple = (10, 6)
):
    """
    Plot portfolio weights as a bar chart.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (ticker symbols as index)
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    # Filter out zero or negligible weights
    weights_filtered = weights[weights > 0.001].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette('husl', len(weights_filtered))
    bars = ax.bar(range(len(weights_filtered)), weights_filtered.values, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(weights_filtered)))
    ax.set_xticklabels(weights_filtered.index, rotation=45, ha='right')
    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(weights_filtered.values) * 1.1)
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_portfolio_pie(
    weights: pd.Series,
    title: str = "Portfolio Composition",
    figsize: tuple = (10, 8)
):
    """
    Plot portfolio weights as a pie chart.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    # Filter out zero weights
    weights_filtered = weights[weights > 0.001].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette('husl', len(weights_filtered))
    wedges, texts, autotexts = ax.pie(
        weights_filtered.values,
        labels=weights_filtered.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_comparison_table(
    results: Dict[str, dict],
    figsize: tuple = (12, 6)
):
    """
    Create comparison table of different optimization methods.
    
    Parameters
    ----------
    results : Dict[str, dict]
        Dictionary with method names as keys and performance metrics as values
    figsize : tuple
        Figure size
    """
    # Create DataFrame from results
    df = pd.DataFrame(results).T
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style row headers
    for i in range(1, len(df.index) + 1):
        table[(i, -1)].set_facecolor('#40466e')
        table[(i, -1)].set_text_props(weight='bold', color='white')
    
    plt.title('Portfolio Optimization Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_cumulative_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_weights: Optional[pd.Series] = None,
    figsize: tuple = (14, 7)
):
    """
    Plot cumulative returns of portfolio vs benchmark.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    weights : pd.Series
        Portfolio weights
    benchmark_weights : pd.Series, optional
        Benchmark weights (e.g., equal weight)
    figsize : tuple
        Figure size
    """
    # Calculate portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumulative_portfolio.index, cumulative_portfolio.values, 
            label='Optimized Portfolio', linewidth=2.5, color='blue')
    
    if benchmark_weights is not None:
        benchmark_returns = (returns * benchmark_weights).sum(axis=1)
        cumulative_benchmark = (1 + benchmark_returns).cumprod()
        ax.plot(cumulative_benchmark.index, cumulative_benchmark.values,
                label='Benchmark', linewidth=2.5, color='red', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_interactive_frontier(
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
    sharpe_ratios: np.ndarray,
    optimal_portfolio: Optional[dict] = None
):
    """
    Create interactive Plotly efficient frontier plot.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Array of portfolio returns
    portfolio_volatilities : np.ndarray
        Array of portfolio volatilities
    sharpe_ratios : np.ndarray
        Array of Sharpe ratios
    optimal_portfolio : dict, optional
        Dict with optimal portfolio metrics
    """
    fig = go.Figure()
    
    # Add frontier points
    fig.add_trace(go.Scatter(
        x=portfolio_volatilities,
        y=portfolio_returns,
        mode='markers',
        marker=dict(
            size=8,
            color=sharpe_ratios,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            line=dict(width=1, color='black')
        ),
        text=[f'Sharpe: {sr:.3f}' for sr in sharpe_ratios],
        hovertemplate='Volatility: %{x:.3f}<br>Return: %{y:.3f}<br>%{text}<extra></extra>',
        name='Portfolios'
    ))
    
    # Add optimal portfolio
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['volatility']],
            y=[optimal_portfolio['return']],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name=f"Optimal (Sharpe={optimal_portfolio['sharpe']:.3f})",
            hovertemplate=f"Optimal Portfolio<br>Volatility: {optimal_portfolio['volatility']:.3f}<br>" +
                         f"Return: {optimal_portfolio['return']:.3f}<br>Sharpe: {optimal_portfolio['sharpe']:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title='Interactive Efficient Frontier',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=600
    )
    
    return fig
