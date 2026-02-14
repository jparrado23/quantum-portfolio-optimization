# Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd quantum-portfolio-optimization

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Demo Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/01-complete-demo.ipynb
# Run all cells to see the complete demo
```

## Quick Code Examples

### 1. Download Data

```python
from src.utils.data_loader import download_portfolio_data, calculate_returns, calculate_statistics

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
prices = download_portfolio_data(tickers, start='2020-01-01', end='2025-01-01')

# Calculate returns and statistics
returns = calculate_returns(prices)
stats = calculate_statistics(returns, risk_free_rate=0.04)

print(f"Mean returns:\n{stats['mean_returns']}")
print(f"\nCovariance matrix:\n{stats['cov_matrix']}")
```

### 2. Classical Optimization (Markowitz)

```python
from src.classical.markowitz import MarkowitzOptimizer

# Create optimizer
optimizer = MarkowitzOptimizer(
    mean_returns=stats['mean_returns'],
    cov_matrix=stats['cov_matrix'],
    risk_free_rate=0.04
)

# Optimize for maximum Sharpe ratio
weights, info = optimizer.optimize_max_sharpe(max_weight=0.30)

print(f"Optimal weights:\n{optimizer.get_weights_series(weights)}")
print(f"\nSharpe ratio: {info['sharpe']:.3f}")
print(f"Expected return: {info['return']:.2%}")
print(f"Volatility: {info['volatility']:.2%}")
```

### 3. Genetic Algorithm (with Cardinality Constraint)

```python
from src.classical.heuristics import GeneticAlgorithmOptimizer

# Select exactly 5 assets from 10
ga = GeneticAlgorithmOptimizer(
    mean_returns=stats['mean_returns'].values,
    cov_matrix=stats['cov_matrix'].values,
    risk_free_rate=0.04,
    cardinality=5
)

weights, info = ga.optimize(
    population_size=100,
    generations=200,
    mutation_rate=0.1
)

print(f"Selected {info['n_selected_assets']} assets")
print(f"Sharpe ratio: {info['sharpe']:.3f}")
```

### 4. Quantum Optimization (QAOA)

```python
from src.quantum.qaoa_portfolio import QAOAPortfolioOptimizer

# Create QAOA optimizer
qaoa = QAOAPortfolioOptimizer(
    mean_returns=stats['mean_returns'],
    cov_matrix=stats['cov_matrix'],
    n_assets_to_select=5,
    risk_aversion=0.5,
    p=3  # QAOA depth
)

# Run optimization
weights, info = qaoa.optimize(
    shots=1024,
    optimizer_method='COBYLA',
    max_iter=200,
    backend='statevector',
    n_runs=3
)

print(f"QAOA solution:")
print(f"Selected assets: {info['selected_names']}")
print(f"Sharpe ratio: {info['sharpe']:.3f}")
print(f"Valid solution: {info['is_valid']}")
```

### 5. Risk Analysis

```python
from src.statistics.risk_metrics import RiskMetrics

# Create risk analyzer
risk = RiskMetrics(returns, risk_free_rate=0.04)

# Calculate comprehensive metrics
report = risk.comprehensive_report(weights)

# Print formatted report
risk.print_report(weights, name="My Portfolio")

# Individual metrics
var_95 = risk.value_at_risk(weights, confidence_level=0.95)
cvar_95 = risk.conditional_var(weights, confidence_level=0.95)
max_dd, peak_date, trough_date = risk.maximum_drawdown(weights)

print(f"Value at Risk (95%): {var_95:.2%}")
print(f"CVaR (95%): {cvar_95:.2%}")
print(f"Maximum Drawdown: {max_dd:.2%}")
```

### 6. Visualization

```python
from src.utils.visualization import (
    plot_price_history,
    plot_correlation_matrix,
    plot_portfolio_weights,
    plot_efficient_frontier
)

# Plot price history
fig = plot_price_history(prices)

# Plot correlation matrix
fig = plot_correlation_matrix(stats['corr_matrix'])

# Plot portfolio weights
weights_series = pd.Series(weights, index=tickers)
fig = plot_portfolio_weights(weights_series, title="My Portfolio")

# All plots use matplotlib, call plt.show() to display
import matplotlib.pyplot as plt
plt.show()
```

## Running Scripts Programmatically

### From Python

```python
# Import and run
from src.classical.markowitz import MarkowitzOptimizer
from src.utils.data_loader import download_portfolio_data, calculate_returns, calculate_statistics

# Your code here...
```

### From Command Line

You can run individual modules:

```bash
# Test data loader
python -m src.utils.data_loader

# Test Markowitz optimizer
python -m src.classical.markowitz

# Test QUBO encoding
python -m src.quantum.qubo_encoding
```

## Configuration

Edit `config/settings.yaml` to customize:
- Default tickers
- Date ranges
- Optimization parameters
- Risk parameters
- Visualization settings

## Troubleshooting

### Issue: yfinance not downloading data
**Solution**: Check internet connection, verify ticker symbols are valid

### Issue: Qiskit import errors
**Solution**: Ensure Qiskit is properly installed:
```bash
pip install --upgrade qiskit qiskit-algorithms
```

### Issue: Memory errors with large portfolios
**Solution**: Reduce the number of assets or QAOA depth parameter

### Issue: QAOA not converging
**Solution**: 
- Increase `max_iter`
- Try different `n_runs`
- Adjust `risk_aversion` parameter
- Increase QAOA depth `p`

## Next Steps

1. **Run the demo notebook**: `notebooks/01-complete-demo.ipynb`
2. **Customize portfolio**: Edit tickers in the notebook
3. **Adjust constraints**: Modify cardinality, max_weight, etc.
4. **Compare methods**: Run all three optimizers and compare
5. **Deploy**: Use the trained model for real portfolio management

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check documentation in `docs/`
- Review example notebooks in `notebooks/`

## Resources

- **Qiskit Documentation**: https://qiskit.org/documentation/
- **Portfolio Theory**: Markowitz (1952) "Portfolio Selection"
- **QAOA**: Farhi & Goldstone (2014) "A Quantum Approximate Optimization Algorithm"
- **Yahoo Finance API**: https://github.com/ranaroussi/yfinance
