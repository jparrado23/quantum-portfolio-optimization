"""
Statistical analysis and risk metrics for portfolio optimization.

Implements comprehensive risk and performance metrics used in finance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Calculate portfolio risk and performance metrics.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data
    risk_free_rate : float
        Annual risk-free rate
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.04
    ):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252  # Annualization factor
        
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio return."""
        return np.dot(self.returns.mean() * self.trading_days, weights)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (std dev)."""
        cov = self.returns.cov() * self.trading_days
        return np.sqrt(np.dot(weights, np.dot(cov, weights)))
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe = (Return - RiskFreeRate) / Volatility
        """
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0
    
    def sortino_ratio(self, weights: np.ndarray, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside risk measure).
        
        Uses only downside volatility (negative returns).
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        target_return : float
            Target return threshold
            
        Returns
        -------
        float
            Sortino ratio
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Downside returns (below target)
        downside_returns = portfolio_returns[portfolio_returns < target_return]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std() * np.sqrt(self.trading_days)
        expected_return = portfolio_returns.mean() * self.trading_days
        
        return (expected_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
    
    def value_at_risk(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR is the maximum loss not exceeded with a given confidence level.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        method : str
            'historical', 'parametric', or 'cornish_fisher'
            
        Returns
        -------
        float
            VaR (positive number representing potential loss)
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        if method == 'historical':
            # Historical simulation
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Assumes normal distribution
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
        
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion (accounts for skewness and kurtosis)
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            skew = portfolio_returns.skew()
            kurt = portfolio_returns.kurtosis()
            
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = (z +
                    (z**2 - 1) * skew / 6 +
                    (z**3 - 3*z) * kurt / 24 -
                    (2*z**3 - 5*z) * skew**2 / 36)
            
            var = mean + z_cf * std
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Return as positive number (loss)
        return -var
    
    def conditional_var(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        CVaR is the expected loss given that the loss exceeds VaR.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        confidence_level : float
            Confidence level
            
        Returns
        -------
        float
            CVaR (positive number)
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Calculate expected value of returns below VaR
        tail_losses = portfolio_returns[portfolio_returns <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var
        
        return -cvar
    
    def maximum_drawdown(self, weights: np.ndarray) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.
        
        Maximum peak-to-trough decline in cumulative returns.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
            
        Returns
        -------
        max_dd : float
            Maximum drawdown (positive number)
        peak_date : pd.Timestamp
            Date of peak
        trough_date : pd.Timestamp
            Date of trough
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        
        # Find corresponding peak
        peak_date = cumulative[:trough_date].idxmax()
        
        return -max_dd, peak_date, trough_date
    
    def information_ratio(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray
    ) -> float:
        """
        Calculate information ratio.
        
        Measures excess return per unit of tracking error.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        benchmark_weights : np.ndarray
            Benchmark portfolio weights
            
        Returns
        -------
        float
            Information ratio
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        benchmark_returns = (self.returns * benchmark_weights).sum(axis=1)
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        
        if tracking_error == 0:
            return 0
        
        # Annualized excess return
        excess_return = active_returns.mean() * self.trading_days
        
        return excess_return / tracking_error
    
    def calmar_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Calmar ratio.
        
        Annual return divided by maximum drawdown.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
            
        Returns
        -------
        float
            Calmar ratio
        """
        annual_return = self.portfolio_return(weights)
        max_dd, _, _ = self.maximum_drawdown(weights)
        
        return annual_return / max_dd if max_dd > 0 else 0
    
    def omega_ratio(
        self,
        weights: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.
        
        Probability-weighted ratio of gains to losses.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        threshold : float
            Return threshold (default: 0)
            
        Returns
        -------
        float
            Omega ratio
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        gains = portfolio_returns[portfolio_returns > threshold] - threshold
        losses = threshold - portfolio_returns[portfolio_returns < threshold]
        
        if losses.sum() == 0:
            return np.inf
        
        return gains.sum() / losses.sum()
    
    def comprehensive_report(self, weights: np.ndarray) -> Dict:
        """
        Generate comprehensive risk/return report.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
            
        Returns
        -------
        dict
            Dictionary of all metrics
        """
        max_dd, peak_date, trough_date = self.maximum_drawdown(weights)
        
        report = {
            'Expected Return (Annual)': self.portfolio_return(weights),
            'Volatility (Annual)': self.portfolio_volatility(weights),
            'Sharpe Ratio': self.sharpe_ratio(weights),
            'Sortino Ratio': self.sortino_ratio(weights),
            'VaR (95%, Historical)': self.value_at_risk(weights, 0.95, 'historical'),
            'CVaR (95%)': self.conditional_var(weights, 0.95),
            'Maximum Drawdown': max_dd,
            'Calmar Ratio': self.calmar_ratio(weights),
            'Omega Ratio': self.omega_ratio(weights),
        }
        
        logger.info("Comprehensive risk report generated")
        
        return report
    
    def print_report(self, weights: np.ndarray, name: str = "Portfolio"):
        """
        Print formatted risk report.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        name : str
            Portfolio name for display
        """
        report = self.comprehensive_report(weights)
        
        print(f"\n{'='*60}")
        print(f"{name} - Risk & Performance Metrics")
        print(f"{'='*60}")
        
        for metric, value in report.items():
            if 'Return' in metric or 'Drawdown' in metric or 'VaR' in metric or 'CVaR' in metric:
                print(f"{metric:30s}: {value:>8.2%}")
            else:
                print(f"{metric:30s}: {value:>8.3f}")
        
        print(f"{'='*60}\n")


def calculate_correlation_metrics(returns: pd.DataFrame) -> Dict:
    """
    Calculate correlation-based risk metrics.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
        
    Returns
    -------
    dict
        Correlation metrics
    """
    corr = returns.corr()
    
    metrics = {
        'avg_correlation': corr.values[np.triu_indices_from(corr.values, k=1)].mean(),
        'max_correlation': corr.values[np.triu_indices_from(corr.values, k=1)].max(),
        'min_correlation': corr.values[np.triu_indices_from(corr.values, k=1)].min(),
        'correlation_matrix': corr
    }
    
    return metrics


def calculate_diversification_ratio(
    weights: np.ndarray,
    returns: pd.DataFrame
) -> float:
    """
    Calculate diversification ratio.
    
    Ratio of weighted average volatility to portfolio volatility.
    Higher is better (more diversified).
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    returns : pd.DataFrame
        Returns data
        
    Returns
    -------
    float
        Diversification ratio
    """
    # Individual volatilities
    individual_vols = returns.std() * np.sqrt(252)
    
    # Weighted average volatility
    weighted_avg_vol = np.dot(weights, individual_vols)
    
    # Portfolio volatility
    cov = returns.cov() * 252
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    
    return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0


if __name__ == "__main__":
    # Test risk metrics
    np.random.seed(42)
    
    # Simulate returns
    n_days = 252 * 3  # 3 years
    n_assets = 5
    
    returns_data = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (n_days, n_assets)),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Equal weight portfolio
    weights = np.array([0.2] * 5)
    
    # Calculate metrics
    metrics = RiskMetrics(returns_data, risk_free_rate=0.04)
    metrics.print_report(weights, "Equal Weight Portfolio")
    
    # Diversification ratio
    div_ratio = calculate_diversification_ratio(weights, returns_data)
    print(f"Diversification Ratio: {div_ratio:.3f}")
