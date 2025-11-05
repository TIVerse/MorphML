"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for NAS.

CMA-ES is a state-of-the-art evolutionary strategy that adapts the full covariance
matrix of the search distribution. It's particularly effective for ill-conditioned
and non-separable problems.

Key Features:
- Adaptive covariance matrix (learns problem structure)
- Self-adaptive step-size control
- Invariant to rotations and scalings
- No gradient information needed
- Proven convergence properties

Reference:
    Hansen, N., and Ostermeier, A. "Completely Derandomized Self-Adaptation
    in Evolution Strategies." Evolutionary Computation, 2001.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.search import Individual
from morphml.logging_config import get_logger
from morphml.optimizers.evolutionary.encoding import ArchitectureEncoder

logger = get_logger(__name__)


class CMAES:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES maintains and adapts a multivariate normal distribution
    N(m, σ² C) where:
    - m: Mean vector (center of search)
    - σ: Step-size (scale of search)
    - C: Covariance matrix (shape of search distribution)
    
    The algorithm:
    1. Samples offspring from N(m, σ² C)
    2. Evaluates and ranks offspring
    3. Updates m using weighted recombination of best offspring
    4. Updates evolution paths (momentum-like)
    5. Adapts C based on successful steps
    6. Adapts σ to maintain proper step-size
    
    Key Components:
    - **Weighted Recombination:** Best offspring weighted by log-rank
    - **Cumulative Step-size Adaptation:** Adapts σ based on path length
    - **Rank-μ Update:** Updates C using μ best offspring
    - **Evolution Paths:** Track successful mutation directions
    
    Configuration:
        population_size: Offspring per generation (default: 4+⌊3*ln(n)⌋)
        sigma: Initial step-size (default: 0.3)
        max_generations: Maximum generations (default: 100)
        
    Example:
        >>> from morphml.optimizers.evolutionary import CMAES
        >>> optimizer = CMAES(
        ...     search_space=space,
        ...     config={'sigma': 0.3, 'population_size': 20}
        ... )
        >>> best = optimizer.optimize(evaluator)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            search_space: SearchSpace for architecture sampling
            config: Configuration dictionary
        """
        self.search_space = search_space
        self.config = config or {}
        
        # Architecture encoding
        self.max_nodes = self.config.get('max_nodes', 20)
        self.encoder = ArchitectureEncoder(search_space, self.max_nodes)
        self.dim = self.encoder.get_dimension()
        
        # Population parameters
        self.lambda_ = self.config.get(
            'population_size',
            4 + int(3 * np.log(self.dim))
        )
        self.mu = self.lambda_ // 2  # Number of parents
        
        # Recombination weights (log-rank weighted)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()  # Variance effective selection mass
        
        # Step-size control parameters
        self.sigma = self.config.get('sigma', 0.3)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Covariance matrix adaptation parameters
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)  # Rank-one update
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff)
        )  # Rank-μ update
        
        # Dynamic strategy parameters
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # Initialize distribution
        self.mean = np.random.rand(self.dim)  # Start at random point
        self.C = np.eye(self.dim)  # Covariance matrix (initially identity)
        self.pc = np.zeros(self.dim)  # Evolution path for C
        self.ps = np.zeros(self.dim)  # Evolution path for sigma
        
        # Eigendecomposition (for efficient sampling)
        self.B = np.eye(self.dim)  # Eigenvectors
        self.D = np.ones(self.dim)  # Eigenvalues
        self.BD = self.B * self.D  # B * D for efficient sampling
        
        self.eigeneval_count = 0
        self.eigeneval_interval = int(1 / (self.c1 + self.cmu) / self.dim / 10)
        
        # Optimization state
        self.max_generations = self.config.get('max_generations', 100)
        self.generation = 0
        
        self.best_individual: Optional[Individual] = None
        self.best_fitness: float = -np.inf
        self.best_vector: Optional[np.ndarray] = None
        
        # History
        self.history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized CMA-ES: dim={self.dim}, lambda={self.lambda_}, "
            f"mu={self.mu}, sigma={self.sigma:.3f}"
        )
    
    def sample_offspring(self) -> np.ndarray:
        """
        Sample offspring from N(m, σ² C).
        
        Uses eigendecomposition for efficient sampling:
        x = m + σ * B * D * z  where z ~ N(0, I)
        
        Returns:
            Offspring vector
        """
        z = np.random.randn(self.dim)
        y = self.BD @ z  # B * D * z
        x = self.mean + self.sigma * y
        
        # Clamp to bounds [0, 1]
        x = np.clip(x, 0.0, 1.0)
        
        return x
    
    def update_distribution(self, offspring_vectors: List[np.ndarray]) -> None:
        """
        Update mean, evolution paths, covariance matrix, and step-size.
        
        Args:
            offspring_vectors: Sorted offspring vectors (best first)
        """
        # Store old mean
        old_mean = self.mean.copy()
        
        # Recombination: weighted average of best μ offspring
        self.mean = sum(w * x for w, x in zip(self.weights, offspring_vectors[:self.mu]))
        
        # Cumulation: update evolution paths
        mean_shift = (self.mean - old_mean) / self.sigma
        
        # C^(-1/2) * mean_shift for ps
        self.ps = (
            (1 - self.cs) * self.ps +
            np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.B @ (mean_shift / self.D))
        )
        
        # Cumulation for pc (covariance path)
        hsig = (
            np.linalg.norm(self.ps) /
            np.sqrt(1 - (1 - self.cs) ** (2 * self.generation + 2)) <
            self.chiN * (1.4 + 2 / (self.dim + 1))
        )
        
        self.pc = (
            (1 - self.cc) * self.pc +
            hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_shift
        )
        
        # Adapt covariance matrix
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-μ update
        rank_mu = sum(
            w * np.outer((x - old_mean) / self.sigma, (x - old_mean) / self.sigma)
            for w, x in zip(self.weights, offspring_vectors[:self.mu])
        )
        
        self.C = (
            (1 - self.c1 - self.cmu) * self.C +
            self.c1 * rank_one +
            self.cmu * rank_mu
        )
        
        # Adapt step-size using cumulative step-size adaptation (CSA)
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )
        
        # Update eigendecomposition
        if self.generation - self.eigeneval_count > self.eigeneval_interval:
            self.eigeneval_count = self.generation
            
            # Enforce symmetry
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            
            # Eigendecomposition
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 0))  # Ensure positive
            self.BD = self.B * self.D
    
    def optimize(self, evaluator: Callable) -> Individual:
        """
        Run CMA-ES optimization.
        
        Args:
            evaluator: Function that evaluates ModelGraph -> fitness
            
        Returns:
            Best Individual found
            
        Example:
            >>> def my_evaluator(graph):
            ...     return train_and_evaluate(graph)
            >>> best = optimizer.optimize(my_evaluator)
            >>> print(f"Best fitness: {best.fitness:.4f}")
        """
        logger.info(
            f"Starting CMA-ES optimization for {self.max_generations} generations"
        )
        
        # Main CMA-ES loop
        for generation in range(self.max_generations):
            self.generation = generation + 1
            
            # Sample offspring
            offspring_vectors = [self.sample_offspring() for _ in range(self.lambda_)]
            
            # Evaluate offspring
            offspring_fitnesses = []
            offspring_individuals = []
            
            for vector in offspring_vectors:
                graph = self.encoder.decode(vector)
                fitness = evaluator(graph)
                
                offspring_fitnesses.append(fitness)
                
                individual = Individual(graph)
                individual.fitness = fitness
                offspring_individuals.append(individual)
                
                # Update global best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_vector = vector.copy()
                    self.best_individual = individual
            
            # Sort by fitness (descending)
            sorted_indices = np.argsort(offspring_fitnesses)[::-1]
            sorted_vectors = [offspring_vectors[i] for i in sorted_indices]
            sorted_fitnesses = [offspring_fitnesses[i] for i in sorted_indices]
            
            # Update distribution
            self.update_distribution(sorted_vectors)
            
            # Record history
            avg_fitness = np.mean(offspring_fitnesses)
            
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
                'sigma': self.sigma,
                'condition_number': np.max(self.D) / np.min(self.D) if np.min(self.D) > 0 else np.inf
            })
            
            # Logging
            if generation % 10 == 0 or generation == self.max_generations - 1:
                logger.info(
                    f"Generation {generation}/{self.max_generations}: "
                    f"best={self.best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}, "
                    f"sigma={self.sigma:.4f}"
                )
        
        logger.info(
            f"CMA-ES complete. Best fitness: {self.best_fitness:.4f}"
        )
        
        return self.best_individual
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.history
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot CMA-ES convergence.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot")
            return
        
        if not self.history:
            logger.warning("No history to plot")
            return
        
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        sigma = [h['sigma'] for h in self.history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Fitness plot
        ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        ax1.plot(generations, avg_fitness, 'r--', linewidth=2, label='Average Fitness')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('CMA-ES Fitness Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sigma plot
        ax2.plot(generations, sigma, 'g-', linewidth=2)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Step-size (σ)', fontsize=12)
        ax2.set_title('Step-size Adaptation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CMAES("
            f"dim={self.dim}, "
            f"lambda={self.lambda_}, "
            f"mu={self.mu}, "
            f"sigma={self.sigma:.3f})"
        )


# Convenience function
def optimize_with_cmaes(
    search_space: SearchSpace,
    evaluator: Callable,
    population_size: Optional[int] = None,
    max_generations: int = 100,
    sigma: float = 0.3,
    verbose: bool = True
) -> Individual:
    """
    Quick CMA-ES optimization with sensible defaults.
    
    Args:
        search_space: SearchSpace to optimize over
        evaluator: Fitness evaluation function
        population_size: Offspring per generation (None = auto)
        max_generations: Maximum generations
        sigma: Initial step-size
        verbose: Print progress
        
    Returns:
        Best Individual found
        
    Example:
        >>> from morphml.optimizers.evolutionary import optimize_with_cmaes
        >>> best = optimize_with_cmaes(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     max_generations=50,
        ...     sigma=0.3
        ... )
    """
    config = {
        'max_generations': max_generations,
        'sigma': sigma
    }
    
    if population_size is not None:
        config['population_size'] = population_size
    
    optimizer = CMAES(search_space=search_space, config=config)
    
    best = optimizer.optimize(evaluator)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CMA-ES Optimization Complete")
        print(f"{'='*60}")
        print(f"Best Fitness: {best.fitness:.4f}")
        print(f"Generations: {max_generations}")
        print(f"Population Size: {optimizer.lambda_}")
        print(f"Final σ: {optimizer.sigma:.4f}")
        print(f"{'='*60}\n")
    
    return best
