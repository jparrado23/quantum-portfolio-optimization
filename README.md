# Quantum Portfolio Optimization

A research project comparing **classical** and **quantum** approaches to portfolio optimization using real financial data. This implementation demonstrates how quantum computing algorithms like QAOA can be applied to combinatorial financial optimization problems, while benchmarking against traditional methods.

## Overview

This project solves the portfolio optimization problem: selecting a subset of assets and determining optimal allocation weights to maximize risk-adjusted returns (Sharpe ratio). It compares the performance of quantum algorithms against classical optimization methods on real stock market data downloaded from Yahoo Finance.

**Key Features:**
- 📈 Real financial data integration (Yahoo Finance API)
- ⚛️ Quantum optimization using QAOA (Qiskit)
- 📊 Classical baselines (Markowitz, Genetic Algorithm, Simulated Annealing)
- 📉 Comprehensive risk metrics (VaR, CVaR, Sharpe ratio, etc.)
- 🤖 LLM-powered portfolio explanations
- 📓 Interactive Jupyter notebook demo

## Problem Formulation

Given a universe of assets, the optimizer selects an optimal portfolio subject to:
- **Objective**: Maximize Sharpe ratio (risk-adjusted returns)
- **Constraints**: 
  - Weights sum to 1 (fully invested)
  - Long-only positions (no short selling)
  - Cardinality constraint (select exactly N assets)
  - Maximum position size limits

This is an NP-hard combinatorial optimization problem when cardinality constraints are included.

## Implemented Approaches

### Classical Optimization
1. **Markowitz Mean-Variance** - Quadratic programming for continuous weights
2. **Genetic Algorithm** - Evolutionary computation for discrete asset selection
3. **Simulated Annealing** - Metropolis-Hastings probabilistic search

### Quantum Optimization
1. **QAOA** (Quantum Approximate Optimization Algorithm)
   - Encodes portfolio problem as QUBO (Quadratic Unconstrained Binary Optimization)
   - Uses parameterized quantum circuits
   - Solves via hybrid quantum-classical optimization
   - Implemented using Qiskit's StatevectorSampler

## Repository Structure

```
quantum-portfolio-optimization/
├── src/
│   ├── classical/             # Classical optimization
│   │   ├── markowitz.py      # Mean-variance optimization
│   │   └── heuristics.py     # Genetic Algorithm, Simulated Annealing
│   ├── quantum/               # Quantum algorithms
│   │   ├── qaoa_portfolio.py # QAOA implementation
│   │   └── qubo_encoding.py  # Portfolio QUBO formulation
│   ├── statistics/            # Risk analysis
│   │   └── risk_metrics.py   # VaR, CVaR, Sharpe, etc.
│   ├── llm/                   # LLM integration
│   │   └── explainer.py      # Portfolio explanation generation
│   └── utils/
│       ├── data_loader.py    # Yahoo Finance data fetching
│       └── visualization.py  # Portfolio visualization
├── notebooks/
│   └── 01-complete-demo.ipynb  # Full demonstration
├── config/
│   └── settings.yaml          # Configuration
└── requirements.txt
```

## Technologies Used

- **Quantum Computing**: Qiskit 1.0+
- **Optimization**: CVXPY, SciPy
- **Data**: yfinance, pandas
- **Analysis**: NumPy, statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly

## Installation

```bash
# Clone repository
git clone https://github.com/jparrado23/quantum-portfolio-optimization.git
cd quantum-portfolio-optimization

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The easiest way to explore the project is through the Jupyter notebook:

```bash
jupyter notebook notebooks/01-complete-demo.ipynb
```

This notebook demonstrates:
1. Downloading real stock data
2. Calculating returns and risk metrics
3. Running classical optimization methods
4. Running QAOA quantum optimization
5. Comparing results with visualizations
6. Generating LLM explanations

## Usage Examples

### 1. Download Financial Data

```python
from src.utils.data_loader import download_portfolio_data, calculate_returns, calculate_statistics

# Download stock prices
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
prices = download_portfolio_data(tickers, start='2020-01-01', end='2025-01-01')

# Calculate returns and statistics
returns = calculate_returns(prices)
stats = calculate_statistics(returns, risk_free_rate=0.04)
```

### 2. Classical Optimization

```python
from src.classical.markowitz import MarkowitzOptimizer

# Markowitz mean-variance optimization
optimizer = MarkowitzOptimizer(
    mean_returns=stats['mean_returns'],
    cov_matrix=stats['cov_matrix'],
    risk_free_rate=0.04
)

weights, info = optimizer.optimize_max_sharpe(max_weight=0.30)
print(f"Sharpe ratio: {info['sharpe']:.3f}")
print(f"Expected return: {info['return']:.2%}")
```

### 3. Genetic Algorithm (with Cardinality)

```python
from src.classical.heuristics import GeneticAlgorithmOptimizer

# Select exactly 5 assets from the universe
ga = GeneticAlgorithmOptimizer(
    mean_returns=stats['mean_returns'].values,
    cov_matrix=stats['cov_matrix'].values,
    risk_free_rate=0.04,
)

weights, info = ga.optimize(population_size=100, generations=200)
```

### 4. Quantum QAOA Optimization

```python
from src.quantum.qaoa_portfolio import QAOAPortfolioOptimizer

# QAOA with 3 layers
qaoa = QAOAPortfolioOptimizer(
    mean_returns=stats['mean_returns'],
    cov_matrix=stats['cov_matrix'],
    n_assets_to_select=5,
    risk_aversion=0.5,
    p=3  # QAOA circuit depth
)

result = qaoa.optimize(maxiter=100)
weights = result['weights']
print(f"QAOA Sharpe ratio: {result['sharpe']:.3f}")
```

### 5. Risk Analysis

```python
from src.statistics.risk_metrics import RiskMetrics

# Calculate comprehensive risk metrics
risk = RiskMetrics(returns, weights)

metrics = risk.calculate_all_metrics(confidence_level=0.95)
print(f"Value at Risk (VaR): {metrics['var']:.2%}")
print(f"Conditional VaR (CVaR): {metrics['cvar']:.2%}")
print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
```

### 6. Portfolio Explanation with LLM

```python
from src.llm.explainer import explain_portfolio

# Generate natural language explanation
explanation = explain_portfolio(
    weights=weights,
    tickers=tickers,
    stats=stats,
    method='QAOA'
)
print(explanation)
```

## Risk Metrics Implemented

- **Sharpe Ratio** - Risk-adjusted return measure
- **Sortino Ratio** - Downside deviation-adjusted return
- **Value at Risk (VaR)** - Potential loss at confidence level
- **Conditional VaR (CVaR)** - Expected loss beyond VaR
- **Maximum Drawdown** - Peak-to-trough decline
- **Calmar Ratio** - Return/max drawdown ratio
- **Volatility** - Standard deviation of returns
- **Beta** - Systematic risk relative to market

## How QAOA Works for Portfolio Optimization

1. **QUBO Encoding**: Convert portfolio problem to binary optimization
   - Each asset selection is a binary variable (0 = not selected, 1 = selected)
   - Objective combines return maximization and risk minimization
   
2. **Circuit Construction**: Build parameterized quantum circuit
   - Problem Hamiltonian: encodes optimization objective
   - Mixer Hamiltonian: enables exploration of solution space
   - p layers for depth-quality tradeoff

3. **Hybrid Optimization**: Classical optimizer tunes quantum circuit parameters
   - Measure quantum circuit to sample solutions
   - Evaluate objective function classically
   - Update parameters to minimize objective

4. **Solution Extraction**: Decode binary result to portfolio weights

### QAOA Parameters

- **Circuit Depth (p)**: 1-5 layers (tradeoff between solution quality and execution time)
- **Qubits**: One per asset in the universe
- **Classical Optimizer**: COBYLA (Constrained Optimization BY Linear Approximations)
- **Shots**: 1024 measurements per circuit evaluation
- **Backend**: Qiskit StatevectorSampler (local simulation)

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
data:
  tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
  start_date: '2020-01-01'
  end_date: '2025-01-01'

optimization:
  risk_free_rate: 0.04
  cardinality: 5
  max_weight: 0.30

qaoa:
  p_layers: 3
  penalty_lambda: 10.0
  risk_aversion: 0.5
  maxiter: 100
```

## Project Highlights

- ✅ Real financial data from Yahoo Finance API
- ✅ Complete QAOA implementation with QUBO encoding
- ✅ Multiple classical baselines for fair comparison
- ✅ Comprehensive risk metrics (VaR, CVaR, Sharpe, etc.)
- ✅ LLM-powered portfolio explanations
- ✅ Interactive visualizations (efficient frontier, correlation heatmaps)
- ✅ Modular, well-documented code structure
- ✅ Jupyter notebook with full walkthrough

## Limitations & Future Work

**Current Limitations:**
- QAOA runs on simulator (no real quantum hardware access)
- Limited to small portfolios (≤20 assets due to qubit constraints)
- Historical returns may not predict future performance
- No transaction costs or liquidity constraints

**Potential Extensions:**
- VQE (Variational Quantum Eigensolver) implementation
- Integration with IBM Quantum or AWS Braket for real hardware
- Multi-period dynamic rebalancing
- Factor models (Fama-French)
- Transaction cost optimization
- Alternative risk measures (CVaR-optimal portfolios)

## References

**Quantum Algorithms:**
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm"
- Peruzzo, A., et al. (2014). "A variational eigenvalue solver on a photonic quantum processor"

**Portfolio Optimization:**
- Markowitz, H. (1952). "Portfolio Selection" - *Journal of Finance*
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*

**Quantum Finance:**
- Orus, R., Mugel, S., & Lizaso, E. (2019). "Quantum computing for finance: Overview and prospects"
- Egger, D. J., et al. (2020). "Quantum computing for Finance: State of the art and future prospects"

## License

MIT License - See LICENSE file for details

## Contact

**Juan Parrado**  
GitHub: [@jparrado23](https://github.com/jparrado23)

---

*Note: This is a research and educational project. Not intended for actual investment decisions.* 
