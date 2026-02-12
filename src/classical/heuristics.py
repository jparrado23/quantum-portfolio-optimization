"""
Heuristic optimization methods for portfolio selection.

Implements metaheuristic algorithms that can handle discrete cardinality
constraints which make the problem NP-hard.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for portfolio optimization with cardinality constraints.
    
    Particularly useful when we want to limit the number of assets (e.g., select
    exactly K assets from N candidates).
    
    Parameters
    ----------
    mean_returns : np.ndarray
        Expected returns for each asset
    cov_matrix : np.ndarray
        Covariance matrix
    risk_free_rate : float
        Risk-free rate for Sharpe calculation
    cardinality : int
        Number of assets to select
    """
    
    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.04,
        cardinality: Optional[int] = None,
        max_weight: float = 1.0
    ):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(mean_returns)
        self.cardinality = cardinality if cardinality else self.n_assets
        self.max_weight = max_weight
        
        logger.info(f"Initialized GA optimizer: {self.n_assets} assets, cardinality={self.cardinality}, max_weight={max_weight}")
    
    def create_individual(self) -> np.ndarray:
        """
        Create random portfolio with cardinality and max_weight constraints.
        
        Returns
        -------
        np.ndarray
            Random portfolio weights
        """
        weights = np.zeros(self.n_assets)
        
        # Select random assets
        selected_idx = np.random.choice(self.n_assets, self.cardinality, replace=False)
        
        # Assign random weights and normalize
        random_weights = np.random.random(self.cardinality)
        random_weights /= random_weights.sum()
        
        weights[selected_idx] = random_weights
        
        # Enforce max_weight constraint
        weights = self._enforce_max_weight(weights)
        
        return weights
    
    def _enforce_max_weight(self, weights: np.ndarray) -> np.ndarray:
        """
        Ensure no weight exceeds max_weight constraint.
        Redistributes excess weight to other selected positions.
        """
        if self.max_weight >= 1.0:
            # No constraint to enforce
            return weights
        
        # Iteratively clip and redistribute excess weight
        max_iterations = 20
        for iteration in range(max_iterations):
            # Find weights exceeding max
            exceeds = weights > self.max_weight
            
            if not np.any(exceeds):
                # All weights are within bounds
                break
            
            # Calculate total excess
            excess = np.sum(weights[exceeds] - self.max_weight)
            
            # Clip exceeding weights
            weights[exceeds] = self.max_weight
            
            # Find positions that can absorb excess (selected but not at max)
            selected = weights > 0
            can_absorb = selected & (weights < self.max_weight)
            
            if not np.any(can_absorb):
                # All selected assets are at max - can't redistribute
                # This means max_weight is too restrictive for cardinality
                logger.warning(f"Cannot enforce max_weight={self.max_weight} with cardinality={self.cardinality}. "
                              f"Need max_weight >= {1.0/self.cardinality:.3f}")
                break
            
            # Distribute excess proportionally to available capacity
            available_capacity = self.max_weight - weights[can_absorb]
            total_capacity = np.sum(available_capacity)
            
            if total_capacity > 0:
                # Distribute proportionally
                weights[can_absorb] += excess * (available_capacity / total_capacity)
            else:
                break
        
        # Final normalization to ensure sum = 1
        if weights.sum() > 0:
            weights /= weights.sum()
        
        return weights
    
    def fitness(self, weights: np.ndarray) -> float:
        """
        Calculate fitness (Sharpe ratio).
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
            
        Returns
        -------
        float
            Sharpe ratio (fitness score)
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            return 0
        
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return sharpe
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents with weight inheritance.
        
        Children inherit both asset selection AND weights from parents.
        For assets present in both parents, weights are averaged.
        For assets in only one parent, weight is inherited directly.
        """
        # Get selected assets from both parents
        selected1 = np.where(parent1 > 0)[0]
        selected2 = np.where(parent2 > 0)[0]
        
        # Combine and randomly select cardinality assets
        combined = np.unique(np.concatenate([selected1, selected2]))
        
        if len(combined) >= self.cardinality:
            child1_selected = np.random.choice(combined, self.cardinality, replace=False)
            child2_selected = np.random.choice(combined, self.cardinality, replace=False)
        else:
            # If not enough unique assets, add random ones
            remaining = np.setdiff1d(np.arange(self.n_assets), combined)
            needed = self.cardinality - len(combined)
            additional = np.random.choice(remaining, needed, replace=False)
            
            all_selected = np.concatenate([combined, additional])
            child1_selected = all_selected[:self.cardinality]
            child2_selected = np.random.permutation(all_selected)[:self.cardinality]
        
        # Create children with inherited weights
        child1 = np.zeros(self.n_assets)
        child2 = np.zeros(self.n_assets)
        
        # Child 1: Inherit weights from parents
        for asset in child1_selected:
            if asset in selected1 and asset in selected2:
                # Both parents have it: average their weights
                child1[asset] = (parent1[asset] + parent2[asset]) / 2
            elif asset in selected1:
                # Only parent1 has it
                child1[asset] = parent1[asset]
            elif asset in selected2:
                # Only parent2 has it
                child1[asset] = parent2[asset]
            else:
                # New asset not in either parent (from additional)
                child1[asset] = 1.0 / self.cardinality
        
        # Child 2: Inherit weights from parents
        for asset in child2_selected:
            if asset in selected1 and asset in selected2:
                # Both parents have it: average their weights
                child2[asset] = (parent1[asset] + parent2[asset]) / 2
            elif asset in selected1:
                # Only parent1 has it
                child2[asset] = parent1[asset]
            elif asset in selected2:
                # Only parent2 has it
                child2[asset] = parent2[asset]
            else:
                # New asset not in either parent (from additional)
                child2[asset] = 1.0 / self.cardinality
        
        # Normalize to ensure weights sum to 1
        if child1.sum() > 0:
            child1 /= child1.sum()
        if child2.sum() > 0:
            child2 /= child2.sum()
        
        # Enforce max_weight constraint
        child1 = self._enforce_max_weight(child1)
        child2 = self._enforce_max_weight(child2)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """
        Mutate an individual while STRICTLY maintaining cardinality constraint.
        
        With probability mutation_rate, either:
        - Swap one asset for another (maintains cardinality)
        - Adjust weights slightly (without eliminating assets)
        """
        if np.random.random() < mutation_rate:
            selected_idx = np.where(individual > 0)[0]
            
            # Ensure we have exactly the right number of assets
            if len(selected_idx) != self.cardinality:
                # Fix cardinality violation (shouldn't happen, but safety check)
                individual = self._enforce_cardinality(individual)
                selected_idx = np.where(individual > 0)[0]
            
            if np.random.random() < 0.5 and len(selected_idx) > 0:
                # Swap asset (maintains cardinality)
                unselected = np.where(individual == 0)[0]
                if len(unselected) > 0:
                    remove_idx = np.random.choice(selected_idx)
                    add_idx = np.random.choice(unselected)
                    
                    # Transfer weight to new asset
                    individual[add_idx] = individual[remove_idx]
                    individual[remove_idx] = 0
                    # Renormalize
                    individual /= individual.sum()
            else:
                # Adjust weights WITHOUT eliminating any asset
                if len(selected_idx) > 0:
                    # Add small perturbation
                    perturbation = np.random.normal(0, 0.05, len(selected_idx))
                    individual[selected_idx] += perturbation
                    
                    # Ensure all selected assets keep positive weight (minimum 1%)
                    min_weight = 0.01
                    individual[selected_idx] = np.maximum(individual[selected_idx], min_weight)
                    
                    # Renormalize to sum to 1
                    individual /= individual.sum()
                    
                    # Enforce max_weight constraint
                    individual = self._enforce_max_weight(individual)
        
        return individual
    
    def _enforce_cardinality(self, individual: np.ndarray) -> np.ndarray:
        """
        Enforce cardinality constraint by ensuring exactly K assets are selected.
        """
        selected_idx = np.where(individual > 0)[0]
        n_selected = len(selected_idx)
        
        if n_selected == self.cardinality:
            return individual
        
        new_individual = np.zeros(self.n_assets)
        
        if n_selected > self.cardinality:
            # Too many assets: keep top K by weight
            top_k_idx = selected_idx[np.argsort(individual[selected_idx])[-self.cardinality:]]
            new_individual[top_k_idx] = individual[top_k_idx]
        else:
            # Too few assets: add random ones
            new_individual[selected_idx] = individual[selected_idx]
            n_needed = self.cardinality - n_selected
            unselected = np.where(individual == 0)[0]
            if len(unselected) >= n_needed:
                additional_idx = np.random.choice(unselected, n_needed, replace=False)
                new_individual[additional_idx] = 1.0 / n_needed
        
        # Normalize
        if new_individual.sum() > 0:
            new_individual /= new_individual.sum()
        
        return new_individual
    
    def optimize(
        self,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 10
    ) -> Tuple[np.ndarray, dict]:
        """
        Run genetic algorithm optimization.
        
        Parameters
        ----------
        population_size : int
            Size of population
        generations : int
            Number of generations
        mutation_rate : float
            Probability of mutation
        crossover_rate : float
            Probability of crossover
        elite_size : int
            Number of top individuals to preserve
            
        Returns
        -------
        best_weights : np.ndarray
            Best portfolio found
        info : dict
            Performance metrics and history
        """
        # Initialize population
        population = [self.create_individual() for _ in range(population_size)]
        
        best_fitness_history = []
        avg_fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = np.array([self.fitness(ind) for ind in population])
            
            # Track progress
            best_fitness_history.append(np.max(fitness_scores))
            avg_fitness_history.append(np.mean(fitness_scores))
            
            if gen % 50 == 0:
                logger.info(f"Generation {gen}: Best Sharpe={np.max(fitness_scores):.4f}, "
                           f"Avg Sharpe={np.mean(fitness_scores):.4f}")
            
            # Select elite
            elite_idx = np.argsort(fitness_scores)[-elite_size:]
            elites = [population[i] for i in elite_idx]
            
            # Create next generation
            new_population = elites.copy()
            
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 5
                tournament_idx = np.random.choice(population_size, tournament_size, replace=False)
                tournament_fitness = fitness_scores[tournament_idx]
                winner1_idx = tournament_idx[np.argmax(tournament_fitness)]
                
                tournament_idx = np.random.choice(population_size, tournament_size, replace=False)
                tournament_fitness = fitness_scores[tournament_idx]
                winner2_idx = tournament_idx[np.argmax(tournament_fitness)]
                
                parent1 = population[winner1_idx]
                parent2 = population[winner2_idx]
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        # Final evaluation
        fitness_scores = np.array([self.fitness(ind) for ind in population])
        best_idx = np.argmax(fitness_scores)
        best_weights = population[best_idx]
        
        # Calculate metrics
        portfolio_return = np.dot(best_weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(best_weights, np.dot(self.cov_matrix, best_weights)))
        
        info = {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': fitness_scores[best_idx],
            'generations': generations,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'n_selected_assets': np.sum(best_weights > 0.001)
        }
        
        logger.info(f"GA optimization complete: Sharpe={info['sharpe']:.3f}, "
                   f"Assets selected={info['n_selected_assets']}")
        
        return best_weights, info


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing for portfolio optimization.
    
    Probabilistic optimization method inspired by metallurgy.
    Good for escaping local optima.
    """
    
    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.04,
        cardinality: Optional[int] = None
    ):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(mean_returns)
        self.cardinality = cardinality if cardinality else self.n_assets
    
    def energy(self, weights: np.ndarray) -> float:
        """
        Energy function (negative Sharpe ratio).
        
        Lower energy is better.
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            return 1e10
        
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return -sharpe  # Negative because we minimize energy
    
    def neighbor(self, weights: np.ndarray) -> np.ndarray:
        """Generate neighboring solution."""
        new_weights = weights.copy()
        selected_idx = np.where(new_weights > 0)[0]
        
        if len(selected_idx) > 0:
            # Small random perturbation
            idx = np.random.choice(selected_idx)
            new_weights[idx] += np.random.normal(0, 0.05)
            new_weights[idx] = max(0, new_weights[idx])
            
            # Renormalize
            if new_weights.sum() > 0:
                new_weights /= new_weights.sum()
        
        return new_weights
    
    def optimize(
        self,
        initial_temp: float = 1000,
        cooling_rate: float = 0.95,
        iterations: int = 5000
    ) -> Tuple[np.ndarray, dict]:
        """
        Run simulated annealing optimization.
        
        Parameters
        ----------
        initial_temp : float
            Initial temperature
        cooling_rate : float
            Temperature reduction factor (< 1)
        iterations : int
            Number of iterations
            
        Returns
        -------
        best_weights : np.ndarray
            Best portfolio found
        info : dict
            Performance metrics
        """
        # Initialize random solution
        current_weights = np.zeros(self.n_assets)
        selected_idx = np.random.choice(self.n_assets, self.cardinality, replace=False)
        random_weights = np.random.random(self.cardinality)
        current_weights[selected_idx] = random_weights / random_weights.sum()
        
        current_energy = self.energy(current_weights)
        best_weights = current_weights.copy()
        best_energy = current_energy
        
        temperature = initial_temp
        energy_history = []
        
        for i in range(iterations):
            # Generate neighbor
            new_weights = self.neighbor(current_weights)
            new_energy = self.energy(new_weights)
            
            # Acceptance criterion
            delta_energy = new_energy - current_energy
            
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                current_weights = new_weights
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_weights = current_weights.copy()
                    best_energy = current_energy
            
            # Cool down
            temperature *= cooling_rate
            energy_history.append(-best_energy)  # Convert back to Sharpe
            
            if i % 1000 == 0:
                logger.info(f"Iteration {i}: Best Sharpe={-best_energy:.4f}, Temp={temperature:.2f}")
        
        # Calculate metrics
        portfolio_return = np.dot(best_weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(best_weights, np.dot(self.cov_matrix, best_weights)))
        
        info = {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': -best_energy,
            'iterations': iterations,
            'energy_history': energy_history,
            'final_temp': temperature
        }
        
        logger.info(f"SA optimization complete: Sharpe={info['sharpe']:.3f}")
        
        return best_weights, info


if __name__ == "__main__":
    # Test GA optimizer
    np.random.seed(42)
    
    n_assets = 10
    mean_returns = np.random.uniform(0.05, 0.20, n_assets)
    
    # Random covariance
    corr = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    vols = np.random.uniform(0.15, 0.35, n_assets)
    cov_matrix = np.outer(vols, vols) * corr
    
    # Run GA
    ga = GeneticAlgorithmOptimizer(mean_returns, cov_matrix, cardinality=5)
    weights, info = ga.optimize(population_size=100, generations=100)
    
    print("Genetic Algorithm Result:")
    print(f"Selected assets: {np.where(weights > 0.001)[0]}")
    print(f"Weights: {weights[weights > 0.001]}")
    print(f"Sharpe: {info['sharpe']:.3f}")
