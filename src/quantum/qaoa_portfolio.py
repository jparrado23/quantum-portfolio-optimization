"""
QAOA (Quantum Approximate Optimization Algorithm) for portfolio optimization.

Implements QAOA to solve the portfolio selection problem encoded as QUBO.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from scipy.optimize import minimize
import logging

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False

from .qubo_encoding import PortfolioQUBOEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAOAPortfolioOptimizer:
    """
    QAOA optimizer for portfolio selection.
    
    Uses Quantum Approximate Optimization Algorithm to find optimal
    portfolio allocation. Particularly effective for discrete asset
    selection problems (combinatorial optimization).
    
    Parameters
    ----------
    mean_returns : np.ndarray or pd.Series
        Expected returns for each asset
    cov_matrix : np.ndarray or pd.DataFrame
        Covariance matrix
    n_assets_to_select : int
        Number of assets to include in portfolio (cardinality)
    risk_aversion : float
        Risk aversion parameter (0-1)
    p : int
        Number of QAOA layers (circuit depth)
    penalty_lambda : float
        Penalty for constraint violations
    """
    
    def __init__(
        self,
        mean_returns,
        cov_matrix,
        n_assets_to_select: int = 5,
        risk_aversion: float = 0.5,
        p: int = 3,
        penalty_lambda: float = 10.0
    ):
        # Convert to numpy if pandas
        if isinstance(mean_returns, pd.Series):
            self.asset_names = mean_returns.index.tolist()
            mean_returns = mean_returns.values
        else:
            self.asset_names = [f"Asset_{i}" for i in range(len(mean_returns))]
        
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.values
        
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.n_assets = len(mean_returns)
        self.n_assets_to_select = n_assets_to_select
        self.risk_aversion = risk_aversion
        self.p = p
        self.penalty_lambda = penalty_lambda
        
        # Create QUBO encoder
        self.encoder = PortfolioQUBOEncoder(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_aversion=risk_aversion,
            budget_constraint=n_assets_to_select,
            penalty_lambda=penalty_lambda
        )
        
        # Get Hamiltonian
        self.hamiltonian = self.encoder.get_qiskit_operator()
        
        # Create QAOA circuit
        self.circuit, self.gamma, self.beta = self._create_qaoa_circuit()
        
        logger.info(f"Initialized QAOA optimizer: {self.n_assets} qubits, p={p}, "
                   f"selecting {n_assets_to_select} assets")
    
    def _create_qaoa_circuit(self) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        """
        Create QAOA circuit.
        
        Returns
        -------
        circuit : QuantumCircuit
            QAOA circuit
        gamma : ParameterVector
            Cost layer parameters
        beta : ParameterVector
            Mixer layer parameters
        """
        from qiskit.circuit.library import QAOAAnsatz
        
        # Create QAOA ansatz
        qaoa = QAOAAnsatz(
            cost_operator=self.hamiltonian,
            reps=self.p,
            initial_state=None,  # Uses default |+⟩^n
            mixer_operator=None  # Uses default X mixer
        )
        
        # Extract parameters
        params = list(qaoa.parameters)
        gamma = ParameterVector("γ", self.p)
        beta = ParameterVector("β", self.p)
        
        # Create parameter mapping
        param_dict = {}
        for i in range(self.p):
            param_dict[params[2*i]] = gamma[i]
            param_dict[params[2*i + 1]] = beta[i]
        
        # Rebind parameters
        qaoa = qaoa.assign_parameters(param_dict)
        
        return qaoa, gamma, beta
    
    def _expectation_value(
        self,
        params: np.ndarray,
        sampler,
        shots: int = 1024
    ) -> float:
        """
        Calculate expectation value of Hamiltonian.
        
        Parameters
        ----------
        params : np.ndarray
            QAOA parameters [γ_0, β_0, γ_1, β_1, ...]
        sampler : Sampler
            Qiskit sampler
        shots : int
            Number of measurement shots
            
        Returns
        -------
        float
            Expectation value (objective function value)
        """
        # Bind parameters
        param_dict = {}
        for i in range(self.p):
            param_dict[self.gamma[i]] = params[2*i]
            param_dict[self.beta[i]] = params[2*i + 1]
        
        bound_circuit = self.circuit.assign_parameters(param_dict)
        
        # Add measurements
        meas_circuit = bound_circuit.copy()
        meas_circuit.measure_all()
        
        # Run circuit
        job = sampler.run([meas_circuit], shots=shots)
        result = job.result()
        
        # Get counts
        counts = result[0].data.meas.get_counts()
        
        # Calculate expectation value
        expectation = 0.0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Reverse bitstring (Qiskit convention)
            bitstring = bitstring[::-1]
            
            # Evaluate objective
            obj_value = self.encoder.evaluate_bitstring(bitstring)
            expectation += obj_value * (count / total_shots)
        
        return expectation
    
    def optimize(
        self,
        shots: int = 1024,
        optimizer_method: str = 'COBYLA',
        max_iter: int = 200,
        initial_params: Optional[np.ndarray] = None,
        backend: str = 'statevector',
        n_runs: int = 3
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run QAOA optimization.
        
        Parameters
        ----------
        shots : int
            Number of measurement shots per evaluation
        optimizer_method : str
            Classical optimizer ('COBYLA', 'SLSQP', 'Powell')
        max_iter : int
            Maximum optimizer iterations
        initial_params : np.ndarray, optional
            Initial parameter values
        backend : str
            'statevector' or 'hardware'
        n_runs : int
            Number of optimization runs (takes best result)
            
        Returns
        -------
        best_weights : np.ndarray
            Optimal portfolio weights
        info : dict
            Optimization results and metrics
        """
        # Create sampler
        if backend == 'statevector':
            sampler = StatevectorSampler()
        elif backend == 'hardware':
            if not HAS_IBM_RUNTIME:
                raise ImportError("qiskit-ibm-runtime not installed")
            service = QiskitRuntimeService()
            backend_obj = service.least_busy(operational=True, simulator=False)
            sampler = RuntimeSampler(backend=backend_obj)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        best_result = None
        best_energy = float('inf')
        all_results = []
        
        for run in range(n_runs):
            logger.info(f"QAOA run {run + 1}/{n_runs}")
            
            # Initialize parameters
            if initial_params is None:
                params = np.random.uniform(0, 2*np.pi, 2 * self.p)
            else:
                params = initial_params.copy()
            
            # Track progress
            iteration_count = [0]
            energy_history = []
            
            def objective(params):
                iteration_count[0] += 1
                energy = self._expectation_value(params, sampler, shots)
                energy_history.append(energy)
                
                if iteration_count[0] % 20 == 0:
                    logger.info(f"  Iteration {iteration_count[0]}: Energy = {energy:.4f}")
                
                return energy
            
            # Optimize
            result = minimize(
                objective,
                params,
                method=optimizer_method,
                options={'maxiter': max_iter}
            )
            
            all_results.append({
                'params': result.x,
                'energy': result.fun,
                'success': result.success,
                'history': energy_history
            })
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_result = result
        
        logger.info(f"Best energy across {n_runs} runs: {best_energy:.4f}")
        
        # Get final solution by sampling with optimal parameters
        optimal_params = best_result.x
        param_dict = {}
        for i in range(self.p):
            param_dict[self.gamma[i]] = optimal_params[2*i]
            param_dict[self.beta[i]] = optimal_params[2*i + 1]
        
        bound_circuit = self.circuit.assign_parameters(param_dict)
        meas_circuit = bound_circuit.copy()
        meas_circuit.measure_all()
        
        # Final sampling with more shots
        job = sampler.run([meas_circuit], shots=shots * 10)
        result = job.result()
        counts = result[0].data.meas.get_counts()
        
        # Find best bitstring
        best_bitstring = max(counts.items(), key=lambda x: x[1])[0][::-1]
        
        # Decode to weights
        weights, solution_info = self.encoder.decode_solution(best_bitstring)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        sharpe = (portfolio_return - 0.04) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        info = {
            'optimal_params': optimal_params,
            'optimal_energy': best_energy,
            'bitstring': best_bitstring,
            'selected_assets': solution_info['selected_assets'],
            'selected_names': [self.asset_names[i] for i in solution_info['selected_assets']],
            'n_selected': solution_info['n_selected'],
            'is_valid': solution_info['is_valid'],
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': sharpe,
            'counts': counts,
            'all_runs': all_results,
            'n_iterations': sum(len(r['history']) for r in all_results),
            'backend': backend,
            'shots': shots
        }
        
        logger.info(f"QAOA optimization complete:")
        logger.info(f"  Selected assets: {info['selected_names']}")
        logger.info(f"  Sharpe ratio: {sharpe:.3f}")
        logger.info(f"  Return: {portfolio_return:.2%}")
        logger.info(f"  Volatility: {portfolio_volatility:.2%}")
        
        return weights, info
    
    def get_weights_series(self, weights: np.ndarray) -> pd.Series:
        """Convert weights to pandas Series with asset names."""
        return pd.Series(weights, index=self.asset_names)


if __name__ == "__main__":
    # Test QAOA optimizer
    np.random.seed(42)
    
    # Small portfolio for testing
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    mean_returns = pd.Series([0.12, 0.15, 0.10, 0.18, 0.08], index=assets)
    
    cov_matrix = pd.DataFrame([
        [0.04, 0.01, 0.01, 0.02, 0.01],
        [0.01, 0.05, 0.01, 0.02, 0.01],
        [0.01, 0.01, 0.03, 0.01, 0.01],
        [0.02, 0.02, 0.01, 0.06, 0.02],
        [0.01, 0.01, 0.01, 0.02, 0.03]
    ], index=assets, columns=assets)
    
    # Create optimizer
    qaoa = QAOAPortfolioOptimizer(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        n_assets_to_select=3,
        risk_aversion=0.5,
        p=2
    )
    
    # Run optimization
    weights, info = qaoa.optimize(
        shots=1024,
        optimizer_method='COBYLA',
        max_iter=100,
        n_runs=2
    )
    
    print("\nQAOA Result:")
    print(qaoa.get_weights_series(weights))
    print(f"\nSelected: {info['selected_names']}")
    print(f"Sharpe: {info['sharpe']:.3f}")
