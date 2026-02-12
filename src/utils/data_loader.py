"""
Data loading and processing utilities.

Downloads financial data from Yahoo Finance API and processes it for
portfolio optimization.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_portfolio_data(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start : str
        Start date in format 'YYYY-MM-DD'
    end : str, optional
        End date in format 'YYYY-MM-DD'. If None, uses today.
    interval : str
        Data frequency: '1d' (daily), '1wk' (weekly), '1mo' (monthly)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted close prices, columns are ticker symbols
        
    Examples
    --------
    >>> data = download_portfolio_data(['AAPL', 'MSFT'], '2020-01-01', '2025-01-01')
    >>> print(data.head())
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Downloading data for {len(tickers)} tickers from {start} to {end}")
    
    # Download data
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        progress=True,
        auto_adjust=True  # Use adjusted close prices
    )
    
    # Handle single ticker case
    if len(tickers) == 1:
        prices = data['Close'].to_frame()
        prices.columns = tickers
    else:
        # Multi-ticker: extract Close prices
        prices = data['Close']
    
    # Remove any rows with NaN values
    prices = prices.dropna()
    
    logger.info(f"Downloaded {len(prices)} data points for {len(tickers)} tickers")
    logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    return prices


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "log"
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index
    method : str
        'log' for log returns, 'simple' for simple returns
        
    Returns
    -------
    pd.DataFrame
        Returns DataFrame
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    
    return returns.dropna()


def calculate_statistics(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04
) -> dict:
    """
    Calculate portfolio statistics from returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    risk_free_rate : float
        Annual risk-free rate (e.g., 0.04 for 4%)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean_returns: Average annual returns per asset
        - cov_matrix: Covariance matrix
        - corr_matrix: Correlation matrix
        - volatilities: Annual volatility per asset
    """
    # Annualize returns (assuming daily data: 252 trading days)
    trading_days = 252
    
    mean_returns = returns.mean() * trading_days
    cov_matrix = returns.cov() * trading_days
    corr_matrix = returns.corr()
    volatilities = returns.std() * np.sqrt(trading_days)
    
    stats = {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'corr_matrix': corr_matrix,
        'volatilities': volatilities,
        'trading_days': trading_days,
        'risk_free_rate': risk_free_rate
    }
    
    logger.info(f"Calculated statistics for {len(returns.columns)} assets")
    logger.info(f"Average annual return: {mean_returns.mean():.2%}")
    logger.info(f"Average annual volatility: {volatilities.mean():.2%}")
    
    return stats


def get_ticker_info(tickers: List[str]) -> pd.DataFrame:
    """
    Get detailed information about tickers.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ticker information (name, sector, industry, etc.)
    """
    info_list = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            info_list.append({
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'country': info.get('country', 'N/A')
            })
        except Exception as e:
            logger.warning(f"Could not fetch info for {ticker}: {e}")
            info_list.append({
                'ticker': ticker,
                'name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'country': 'N/A'
            })
    
    return pd.DataFrame(info_list)


def save_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    stats: dict,
    output_dir: str = "data/processed"
):
    """
    Save processed data to files.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Returns data
    stats : dict
        Statistics dictionary
    output_dir : str
        Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save price and returns data
    prices.to_csv(f"{output_dir}/prices.csv")
    returns.to_csv(f"{output_dir}/returns.csv")
    
    # Save statistics
    stats['mean_returns'].to_csv(f"{output_dir}/mean_returns.csv")
    stats['cov_matrix'].to_csv(f"{output_dir}/cov_matrix.csv")
    stats['corr_matrix'].to_csv(f"{output_dir}/corr_matrix.csv")
    stats['volatilities'].to_csv(f"{output_dir}/volatilities.csv")
    
    logger.info(f"Data saved to {output_dir}")


def load_data(input_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load previously saved data.
    
    Parameters
    ----------
    input_dir : str
        Input directory path
        
    Returns
    -------
    tuple
        (prices, returns, stats)
    """
    prices = pd.read_csv(f"{input_dir}/prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(f"{input_dir}/returns.csv", index_col=0, parse_dates=True)
    
    stats = {
        'mean_returns': pd.read_csv(f"{input_dir}/mean_returns.csv", index_col=0, squeeze=True),
        'cov_matrix': pd.read_csv(f"{input_dir}/cov_matrix.csv", index_col=0),
        'volatilities': pd.read_csv(f"{input_dir}/volatilities.csv", index_col=0, squeeze=True)
    }
    
    logger.info(f"Data loaded from {input_dir}")
    
    return prices, returns, stats


if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM']
    
    # Download data
    prices = download_portfolio_data(tickers, start='2020-01-01', end='2025-01-28')
    
    # Calculate returns and statistics
    returns = calculate_returns(prices, method='log')
    stats = calculate_statistics(returns, risk_free_rate=0.04)
    
    # Get ticker info
    info = get_ticker_info(tickers)
    print("\nTicker Information:")
    print(info)
    
    # Save data
    save_data(prices, returns, stats)
    
    print(f"\nData shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\nMean annual returns:\n{stats['mean_returns']}")
    print(f"\nAnnual volatilities:\n{stats['volatilities']}")
