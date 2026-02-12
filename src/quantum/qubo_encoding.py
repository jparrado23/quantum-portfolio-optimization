"""
QUBO (Quadratic Unconstrained Binary Optimization) encoding for portfolio optimization.

Converts portfolio optimization problem into a form suitable for quantum algorithms.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioQUBOEncoder:
    """
    Encode portfolio optimization as a QUBO problem.
    
    The general form is:
        H = -∑_i r_i x_i + λ ∑_{i,j} σ_{ij} x_i x_j + penalty_terms
    
    Where:
    - x_i ∈ {0,1}: binary variable for asset selection
    - r_i: expected return of asset i
    - σ_{ij}: covariance between assets i and j
    - λ: risk aversion parameter (balance return vs risk)
    
    This formulation assumes discrete asset selection (buy/don't buy).
    For fractional weights, we can use multiple bits per asset.
    
    Parameters
    ----------
    mean_returns : np.ndarray
        Expected returns (annualized)
    cov_matrix : np.ndarray
        Covariance matrix (annualized)
    risk_aversion : float
        Risk aversion parameter λ (0 = no risk consideration, 1 = balanced)
    budget_constraint : int
        Exactly K assets must be selected (cardinality constraint)
    penalty_lambda : float
        Penalty coefficient for constraint violations
    """
    
    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 0.5,
        budget_constraint: int = 5,
        penalty_lambda: float = 10.0
    ):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_aversion = risk_aversion
        self.budget_constraint = budget_constraint
        self.penalty_lambda = penalty_lambda
        self.n_assets = len(mean_returns)
        
        # Normalize returns to avoid numerical issues
        self.return_scale = np.max(np.abs(mean_returns))
        self.risk_scale = np.max(np.abs(cov_matrix))
        
        logger.info(f"Initialized QUBO encoder for {self.n_assets} assets, "
                   f"cardinality={budget_constraint}, risk_aversion={risk_aversion}")
    
    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build QUBO matrix Q where objective is x^T Q x.
        
        Returns
        -------
        np.ndarray
            QUBO matrix (n_assets x n_assets)
        """
        n = self.n_assets
        Q = np.zeros((n, n))
        
        # Normalize returns and risk
        norm_returns = self.mean_returns / self.return_scale
        norm_cov = self.cov_matrix / self.risk_scale
        
        # Return contribution (diagonal): maximize returns
        # We want to maximize ∑r_i x_i, so in minimization form: -∑r_i x_i
        for i in range(n):
            Q[i, i] -= norm_returns[i]  # Negative for maximization
        
        # Risk contribution (off-diagonal + diagonal): minimize risk
        # Add λ * covariance terms
        for i in range(n):
            for j in range(n):
                Q[i, j] += self.risk_aversion * norm_cov[i, j]
        
        # Budget constraint penalty: (∑x_i - K)^2
        # Expanding: ∑x_i^2 - 2K∑x_i + K^2
        # Since x_i^2 = x_i for binary variables:
        # ∑x_i - 2K∑x_i + ∑∑x_i*x_j = ∑∑x_i*x_j - 2K∑x_i + K^2
        
        K = self.budget_constraint
        penalty = self.penalty_lambda
        
        # Diagonal terms: x_i (from x_i^2 = x_i)
        for i in range(n):
            Q[i, i] += penalty * (1 - 2 * K)
        
        # Off-diagonal terms: x_i * x_j
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += penalty * 2
                Q[j, i] += penalty * 2  # Symmetric
        
        logger.info(f"Built QUBO matrix: shape={Q.shape}, "
                   f"min={Q.min():.3f}, max={Q.max():.3f}")
        
        return Q
    
    def build_ising_hamiltonian(self) -> Tuple[Dict, float]:
        """
        Convert QUBO to Ising Hamiltonian.
        
        Maps binary variables x ∈ {0,1} to spin variables s ∈ {-1,+1}
        via: x_i = (1 - s_i) / 2
        
        Returns
        -------
        ising_dict : dict
            Dictionary mapping qubit pairs (i,j) to coefficients
        offset : float
            Constant offset
        """
        Q = self.build_qubo_matrix()
        n = self.n_assets
        
        # Initialize Ising parameters
        h = np.zeros(n)  # Linear terms (local fields)
        J = {}  # Coupling terms (interactions)
        offset = 0.0
        
        # Convert QUBO to Ising
        # x_i = (1 - s_i) / 2
        # ∑Q_ij x_i x_j = ∑Q_ij (1-s_i)(1-s_j) / 4
        #                = (1/4) ∑Q_ij (1 - s_i - s_j + s_i*s_j)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal term
                    h[i] -= Q[i, i] / 2
                    offset += Q[i, i] / 4
                else:
                    # Off-diagonal term (only store upper triangle)
                    if i < j:
                        J[(i, j)] = Q[i, j] / 4
                        h[i] -= Q[i, j] / 4
                        h[j] -= Q[i, j] / 4
                        offset += Q[i, j] / 4
        
        # Package as dictionary
        ising_dict = {
            'h': h,  # Linear terms
            'J': J,   # Coupling terms
            'offset': offset
        }
        
        logger.info(f"Converted to Ising: {len(h)} linear terms, {len(J)} coupling terms")
        
        return ising_dict, offset
    
    def decode_solution(self, bitstring: str) -> Tuple[np.ndarray, Dict]:
        """
        Decode binary solution to portfolio weights.
        
        Parameters
        ----------
        bitstring : str
            Binary string (e.g., '10110')
            
        Returns
        -------
        weights : np.ndarray
            Portfolio weights (equal weight among selected assets)
        info : dict
            Solution information
        """
        # Convert bitstring to array
        x = np.array([int(b) for b in bitstring])
        
        # Check if solution is valid (correct number of assets)
        n_selected = x.sum()
        is_valid = (n_selected == self.budget_constraint)
        
        # Create weights: equal weight among selected assets
        weights = np.zeros(self.n_assets)
        if n_selected > 0:
            weights[x == 1] = 1.0 / n_selected
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, self.mean_returns) if n_selected > 0 else 0
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights)) if n_selected > 0 else 0
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        info = {
            'bitstring': bitstring,
            'selected_assets': np.where(x == 1)[0].tolist(),
            'n_selected': n_selected,
            'is_valid': is_valid,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': (portfolio_return - 0.04) / portfolio_volatility if portfolio_volatility > 0 else 0
        }
        
        return weights, info
    
    def evaluate_bitstring(self, bitstring: str) -> float:
        """
        Evaluate objective function for a given bitstring.
        
        Parameters
        ----------
        bitstring : str
            Binary solution
            
        Returns
        -------
        float
            Objective value (lower is better for minimization)
        """
        x = np.array([int(b) for b in bitstring])
        Q = self.build_qubo_matrix()
        
        # Calculate x^T Q x
        objective = np.dot(x, np.dot(Q, x))
        
        return objective
    
    def get_qiskit_operator(self):
        """
        Get Qiskit operator for quantum algorithms.
        
        Returns
        -------
        SparsePauliOp
            Qiskit operator
        """
        from qiskit.quantum_info import SparsePauliOp
        
        ising_dict, offset = self.build_ising_hamiltonian()
        h = ising_dict['h']
        J = ising_dict['J']
        
        # Build Pauli operator list
        pauli_list = []
        
        # Linear terms (Z operators)
        for i, h_i in enumerate(h):
            if abs(h_i) > 1e-10:
                pauli_str = ['I'] * self.n_assets
                pauli_str[i] = 'Z'
                pauli_list.append((''.join(pauli_str), h_i))
        
        # Coupling terms (ZZ operators)
        for (i, j), J_ij in J.items():
            if abs(J_ij) > 1e-10:
                pauli_str = ['I'] * self.n_assets
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_list.append((''.join(pauli_str), J_ij))
        
        # Constant offset (identity)
        if abs(offset) > 1e-10:
            pauli_list.append(('I' * self.n_assets, offset))
        
        operator = SparsePauliOp.from_list(pauli_list)
        
        logger.info(f"Created Qiskit operator with {len(pauli_list)} Pauli terms")
        
        return operator


def create_portfolio_qubo(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_assets_to_select: int = 5,
    risk_aversion: float = 0.5,
    penalty: float = 10.0
) -> PortfolioQUBOEncoder:
    """
    Convenience function to create QUBO encoder from pandas data.
    
    Parameters
    ----------
    mean_returns : pd.Series
        Expected returns with asset names as index
    cov_matrix : pd.DataFrame
        Covariance matrix with asset names as index/columns
    n_assets_to_select : int
        Number of assets to include in portfolio
    risk_aversion : float
        Risk aversion parameter (0-1)
    penalty : float
        Penalty for constraint violations
        
    Returns
    -------
    PortfolioQUBOEncoder
        Configured QUBO encoder
    """
    encoder = PortfolioQUBOEncoder(
        mean_returns=mean_returns.values,
        cov_matrix=cov_matrix.values,
        risk_aversion=risk_aversion,
        budget_constraint=n_assets_to_select,
        penalty_lambda=penalty
    )
    
    return encoder


if __name__ == "__main__":
    # Test QUBO encoding
    np.random.seed(42)
    
    n_assets = 5
    mean_returns = np.array([0.12, 0.15, 0.10, 0.18, 0.08])
    
    # Simple covariance matrix
    cov_matrix = np.array([
        [0.04, 0.01, 0.01, 0.02, 0.01],
        [0.01, 0.05, 0.01, 0.02, 0.01],
        [0.01, 0.01, 0.03, 0.01, 0.01],
        [0.02, 0.02, 0.01, 0.06, 0.02],
        [0.01, 0.01, 0.01, 0.02, 0.03]
    ])
    
    # Create encoder
    encoder = PortfolioQUBOEncoder(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_aversion=0.5,
        budget_constraint=3,
        penalty_lambda=10.0
    )
    
    # Build QUBO
    Q = encoder.build_qubo_matrix()
    print("QUBO Matrix:")
    print(Q)
    
    # Test a solution: select assets 0, 1, 3
    bitstring = "11010"
    weights, info = encoder.decode_solution(bitstring)
    
    print(f"\nSolution: {bitstring}")
    print(f"Selected assets: {info['selected_assets']}")
    print(f"Weights: {weights}")
    print(f"Return: {info['return']:.2%}")
    print(f"Volatility: {info['volatility']:.2%}")
    print(f"Sharpe: {info['sharpe']:.3f}")
    print(f"Objective value: {encoder.evaluate_bitstring(bitstring):.3f}")
