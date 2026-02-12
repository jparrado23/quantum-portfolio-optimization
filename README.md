# Quantum Portfolio Optimization

A comprehensive implementation comparing **classical** and **quantum** approaches to portfolio optimization, demonstrating expertise in:
- ✨ **Quantum Computing** (QAOA, VQE)
- 📊 **Statistical Analysis** (risk metrics, backtesting)
- 🎯 **Operations Research** (optimization algorithms)
- 🤖 **Large Language Models** (portfolio explanation, chatbot)

## Problem Statement

Optimize asset allocation to **maximize risk-adjusted returns** (Sharpe ratio) subject to:
- Full investment constraint (weights sum to 1)
- Long-only positions (no short selling)
- Cardinality constraints (limit number of assets)
- Maximum position size limits

## Approaches Compared

### Classical Methods
1. **Mean-Variance Optimization** (Markowitz) - Convex optimization
2. **Genetic Algorithm** - Metaheuristic for cardinality constraints
3. **Simulated Annealing** - Probabilistic optimization
4. **Risk Parity** - Equal risk contribution baseline

### Quantum Methods
1. **QAOA** - Quantum Approximate Optimization Algorithm
2. **VQE** - Variational Quantum Eigensolver with custom ansatz

## Repository Structure

```
quantum-portfolio-optimization/
├── data/                       # Financial data (gitignored)
│   ├── raw/                   # Downloaded price data
│   └── processed/             # Returns, covariance matrices
├── src/
│   ├── classical/             # Classical optimization methods
│   │   ├── markowitz.py      # Mean-variance optimization
│   │   ├── heuristics.py     # GA, SA, PSO
│   │   └── risk_models.py    # Risk parity, min variance
│   ├── quantum/               # Quantum algorithms
│   │   ├── qaoa_portfolio.py # QAOA implementation
│   │   ├── vqe_portfolio.py  # VQE implementation
│   │   └── qubo_encoding.py  # Problem encoding
│   ├── statistics/            # Statistical analysis
│   │   ├── returns_analysis.py
│   │   ├── risk_metrics.py
│   │   └── backtesting.py
│   ├── llm/                   # LLM integration
│   │   ├── chatbot.py
│   │   └── explainer.py
│   └── utils/
│       ├── data_loader.py    # Yahoo Finance API
│       ├── visualization.py
│       └── comparison.py
├── notebooks/                  # Jupyter analysis
│   ├── 01-data-exploration.ipynb
│   ├── 02-classical-baseline.ipynb
│   ├── 03-quantum-qaoa.ipynb
│   ├── 04-comparison-analysis.ipynb
│   └── 05-llm-integration.ipynb
├── config/
│   └── settings.yaml
├── tests/
└── requirements.txt
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd quantum-portfolio-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.utils.data_loader import download_portfolio_data
from src.classical.markowitz import MarkowitzOptimizer
from src.quantum.qaoa_portfolio import QAOAPortfolioOptimizer

# Download data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
data = download_portfolio_data(tickers, start='2020-01-01', end='2025-01-01')

# Classical optimization
classical = MarkowitzOptimizer(data)
classical_weights = classical.optimize(target_return=0.15)

# Quantum optimization
quantum = QAOAPortfolioOptimizer(data, p=3)
quantum_weights = quantum.optimize()

# Compare results
print(f"Classical Sharpe: {classical.sharpe_ratio(classical_weights):.3f}")
print(f"Quantum Sharpe: {quantum.sharpe_ratio(quantum_weights):.3f}")
```

## Example Problem

**Portfolio**: 10 Technology Stocks
- AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC, CRM

**Historical Period**: 2020-2025 (5 years)

**Objective**: Maximize Sharpe Ratio

**Constraints**:
- Budget: Weights sum to 1.0
- Long-only: w_i ≥ 0
- Max position: 30% per asset
- Cardinality: Select exactly 5 assets

## Key Results

| Method | Sharpe Ratio | Time (s) | Solution Quality |
|--------|--------------|----------|------------------|
| Markowitz (no cardinality) | 1.45 | 0.1 | Optimal* |
| Genetic Algorithm | 1.38 | 12.3 | Near-optimal |
| QAOA (p=5) | 1.36 | 45.2 | Approximate |
| Equal Weight | 1.12 | 0.01 | Baseline |

*Optimal for continuous relaxation (cardinality constraint relaxed)

## Statistical Analysis

- **Return Distributions**: Mean, std, skewness, kurtosis
- **Covariance Estimation**: Sample, Ledoit-Wolf shrinkage
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown, Sortino Ratio
- **Backtesting**: Rolling window, transaction costs
- **Efficient Frontier**: Risk-return tradeoff visualization

## Quantum Implementation Details

### QUBO Formulation

The portfolio optimization problem is encoded as:

$$H = -\sum_i r_i x_i + \lambda \sum_{i,j} \sigma_{ij} x_i x_j + \text{penalty terms}$$

Where:
- $r_i$: Expected return of asset i
- $\sigma_{ij}$: Covariance between assets i and j
- $x_i \in \{0,1\}$: Binary selection variable
- $\lambda$: Risk aversion parameter

### QAOA Circuit

- **Depth**: p=1 to 5 layers
- **Qubits**: One per asset (10 qubits for 10 stocks)
- **Optimizer**: COBYLA (derivative-free)
- **Shots**: 1024 per evaluation

## LLM Integration

### Portfolio Explainer
```python
from src.llm.explainer import PortfolioExplainer

explainer = PortfolioExplainer(api_key='...')
explanation = explainer.explain_allocation(weights, returns_data)
print(explanation)
```

**Output**: 
"This portfolio allocates 30% to NVDA and 25% to MSFT, emphasizing growth-oriented tech stocks. The allocation balances high returns from semiconductors with stability from enterprise software..."

### Interactive Chatbot
Ask questions about your portfolio:
- "Why was Tesla excluded?"
- "What's the risk of this allocation?"
- "How would adding bonds affect the Sharpe ratio?"

## References

### Academic Papers
1. Farhi, E., & Goldstone, J. (2014). "A Quantum Approximate Optimization Algorithm"
2. Slate, N., et al. (2020). "Quantum Walk-Based Portfolio Optimisation"
3. Markowitz, H. (1952). "Portfolio Selection" - Journal of Finance

### Quantum Finance Libraries
- **Qiskit Finance**: IBM's quantum finance module
- **PennyLane**: Differentiable quantum computing
- **AWS Braket**: Portfolio optimization tutorials

## Contributing

Contributions welcome! Areas of interest:
- Additional quantum algorithms (QUBO, Grover)
- More classical heuristics
- Real-time portfolio rebalancing
- Multi-period optimization

## License

MIT License

## Author

Juan Parrado  
Expertise: Quantum Computing | Operations Research | Machine Learning 
