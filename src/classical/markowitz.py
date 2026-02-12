"""
Classical portfolio optimization using Mean-Variance (Markowitz) framework.

Implements the foundational portfolio theory from Harry Markowitz (1952).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkowitzOptimizer:
    """
    Mean-Variance Portfolio Optimizer (Markowitz Model).
    
    Solves the optimization problem:
        minimize: w^T Σ w  (portfolio variance)
        subject to:
            - w^T μ >= target_return (minimum return constraint)
            - sum(w) = 1 (budget constraint)
            - w >= 0 (long-only constraint)
            - optional: max_weight, min_weight constraints
    
    Parameters
    ----------
    mean_returns : pd.Series or np.ndarray
        Expected returns for each asset (annualized)
    cov_matrix : pd.DataFrame or np.ndarray
        Covariance matrix (annualized)
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio calculation
    """
    
    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.04
    ):
        self.mean_returns = np.array(mean_returns)
        self.cov_matrix = np.array(cov_matrix)
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(self.mean_returns)
        self.asset_names = mean_returns.index if isinstance(mean_returns, pd.Series) else None
        
        logger.info(f"Initialized Markowitz optimizer for {self.n_assets} assets")
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return."""
        return np.dot(weights, self.mean_returns)
    
    def portfolio_variance(self, weights: np.ndarray) -> float:
        """Calculate portfolio variance."""
        return np.dot(weights, np.dot(self.cov_matrix, weights))
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(self.portfolio_variance(weights))
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        return -self.sharpe_ratio(weights)
    
    def optimize_max_sharpe(
        self,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Parameters
        ----------
        max_weight : float
            Maximum weight per asset (default: 1.0 = 100%)
        min_weight : float
            Minimum weight per asset (default: 0.0)
            
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights
        info : dict
            Performance metrics
        """
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
        ]
        
        # Bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        w0 = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")
        
        weights = result.x
        
        # Calculate performance metrics
        info = {
            'return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe': self.sharpe_ratio(weights),
            'success': result.success,
            'iterations': result.nit
        }
        
        logger.info(f"Max Sharpe optimization complete: Sharpe={info['sharpe']:.3f}")
        
        return weights, info
    
    def optimize_min_variance(
        self,
        target_return: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize portfolio for minimum variance.
        
        Parameters
        ----------
        target_return : float, optional
            Minimum required return. If None, finds global minimum variance.
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
            
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights
        info : dict
            Performance metrics
        """
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.portfolio_return(w) - target_return
            })
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess
        w0 = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self.portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Optimization warning: {result.message}")
        
        weights = result.x
        
        info = {
            'return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe': self.sharpe_ratio(weights),
            'success': result.success,
            'target_return': target_return
        }
        
        logger.info(f"Min variance optimization complete: Vol={info['volatility']:.3f}")
        
        return weights, info
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate efficient frontier.
        
        Parameters
        ----------
        n_points : int
            Number of points on the frontier
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
            
        Returns
        -------
        pd.DataFrame
            DataFrame with returns, volatilities, and Sharpe ratios
        """
        # Find range of feasible returns
        min_var_weights, _ = self.optimize_min_variance(
            target_return=None,
            max_weight=max_weight,
            min_weight=min_weight
        )
        min_return = self.portfolio_return(min_var_weights)
        
        # Maximum return is putting all weight in best asset (within constraints)
        max_return = np.max(self.mean_returns) * max_weight
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return * 0.95, n_points)
        
        frontier_data = []
        
        for target_ret in target_returns:
            try:
                weights, info = self.optimize_min_variance(
                    target_return=target_ret,
                    max_weight=max_weight,
                    min_weight=min_weight
                )
                
                frontier_data.append({
                    'return': info['return'],
                    'volatility': info['volatility'],
                    'sharpe': info['sharpe']
                })
            except:
                continue
        
        df = pd.DataFrame(frontier_data)
        logger.info(f"Generated efficient frontier with {len(df)} points")
        
        return df
    
    def get_weights_series(self, weights: np.ndarray) -> pd.Series:
        """Convert weights array to pandas Series with asset names."""
        if self.asset_names is not None:
            return pd.Series(weights, index=self.asset_names)
        else:
            return pd.Series(weights)


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """
    Create equal-weight portfolio (1/N strategy).
    
    This is a common benchmark that's surprisingly hard to beat.
    
    Parameters
    ----------
    n_assets : int
        Number of assets
        
    Returns
    -------
    np.ndarray
        Equal weights
    """
    return np.ones(n_assets) / n_assets


def market_cap_weighted(market_caps: pd.Series) -> pd.Series:
    """
    Create market-cap weighted portfolio.
    
    Parameters
    ----------
    market_caps : pd.Series
        Market capitalizations for each asset
        
    Returns
    -------
    pd.Series
        Market-cap weights
    """
    return market_caps / market_caps.sum()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate data for 5 assets
    n_assets = 5
    mean_returns = pd.Series(
        np.random.uniform(0.05, 0.20, n_assets),
        index=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Generate random covariance matrix
    corr = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    
    vols = np.random.uniform(0.15, 0.35, n_assets)
    cov_matrix = pd.DataFrame(
        np.outer(vols, vols) * corr,
        index=mean_returns.index,
        columns=mean_returns.index
    )
    
    # Create optimizer
    optimizer = MarkowitzOptimizer(mean_returns, cov_matrix, risk_free_rate=0.04)
    
    # Optimize for max Sharpe
    weights, info = optimizer.optimize_max_sharpe(max_weight=0.30)
    
    print("Optimal Portfolio (Max Sharpe):")
    print(optimizer.get_weights_series(weights))
    print(f"\nExpected Return: {info['return']:.2%}")
    print(f"Volatility: {info['volatility']:.2%}")
    print(f"Sharpe Ratio: {info['sharpe']:.3f}")
