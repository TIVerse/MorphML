"""Gaussian Process-based Bayesian optimization for neural architecture search.

This module implements Bayesian optimization using Gaussian Process (GP) as the
surrogate model. GP provides a probabilistic model of the fitness function,
enabling intelligent exploration-exploitation trade-offs through acquisition functions.

Key Features:
- Multiple kernel options (Matern, RBF, etc.)
- Multiple acquisition functions (EI, UCB, PI)
- Efficient architecture encoding for GP modeling
- Multi-restart optimization for acquisition maximization
- Automatic hyperparameter tuning for GP

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    RBF,
    WhiteKernel,
)

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.core.search import Individual
from morphml.logging_config import get_logger
from morphml.optimizers.bayesian.acquisition import (
    AcquisitionOptimizer,
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)
from morphml.optimizers.bayesian.base import BaseBayesianOptimizer

logger = get_logger(__name__)


class GaussianProcessOptimizer(BaseBayesianOptimizer):
    """
    Gaussian Process-based Bayesian optimization.
    
    Uses a Gaussian Process as a surrogate model to approximate the
    fitness function. The GP provides both mean predictions (exploitation)
    and uncertainty estimates (exploration), enabling intelligent sampling
    through acquisition functions.
    
    Algorithm:
    1. Initialize with random samples
    2. Fit GP on observed (x, y) pairs
    3. Optimize acquisition function to find next x
    4. Evaluate fitness at x
    5. Update GP and repeat
    
    Configuration:
        acquisition: 'ei', 'ucb', 'pi' (default: 'ei')
        kernel: 'matern', 'rbf', 'matern52' (default: 'matern')
        n_initial_points: Random samples before GP (default: 10)
        xi: Exploration parameter for EI/PI (default: 0.01)
        kappa: Exploration parameter for UCB (default: 2.576)
        acq_optimizer: 'lbfgs', 'de', 'random' (default: 'lbfgs')
        normalize_y: Normalize fitness values (default: True)
        n_restarts: GP hyperparameter optimization restarts (default: 5)
        
    Example:
        >>> from morphml.optimizers.bayesian import GaussianProcessOptimizer
        >>> from morphml.core.dsl import create_cnn_space
        >>> 
        >>> space = create_cnn_space(num_classes=10)
        >>> optimizer = GaussianProcessOptimizer(
        ...     search_space=space,
        ...     config={
        ...         'acquisition': 'ei',
        ...         'kernel': 'matern',
        ...         'n_initial_points': 10
        ...     }
        ... )
        >>> 
        >>> def evaluate(graph):
        ...     return train_and_evaluate(graph)
        >>> 
        >>> best = optimizer.optimize(evaluate, max_evaluations=100)
        >>> print(f"Best fitness: {best.fitness:.4f}")
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Gaussian Process optimizer.
        
        Args:
            search_space: SearchSpace defining architecture options
            config: Configuration dictionary with optional keys:
                - acquisition: Acquisition function type
                - kernel: GP kernel type
                - n_initial_points: Initial random samples
                - xi: EI/PI exploration parameter
                - kappa: UCB exploration parameter
                - acq_optimizer: Acquisition optimization method
                - normalize_y: Whether to normalize targets
                - n_restarts: GP hyperparameter optimization restarts
        """
        super().__init__(search_space, config or {})
        
        # Acquisition function configuration
        self.acquisition_type = self.config.get('acquisition', 'ei')
        self.xi = self.config.get('xi', 0.01)
        self.kappa = self.config.get('kappa', 2.576)
        
        # GP configuration
        self.kernel_type = self.config.get('kernel', 'matern')
        self.normalize_y = self.config.get('normalize_y', True)
        self.n_restarts_optimizer = self.config.get('n_restarts', 5)
        self.alpha = self.config.get('alpha', 1e-6)  # Noise level
        
        # Acquisition optimizer configuration
        self.acq_optimizer_method = self.config.get('acq_optimizer', 'lbfgs')
        self.acq_n_restarts = self.config.get('acq_n_restarts', 10)
        
        # Initialize kernel
        self.kernel = self._create_kernel()
        
        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        
        # Initialize acquisition optimizer
        self.acq_optimizer = AcquisitionOptimizer(
            method=self.acq_optimizer_method,
            n_restarts=self.acq_n_restarts,
            random_state=self.random_state
        )
        
        # Observation storage
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        
        # Track whether GP is fitted
        self._gp_fitted = False
        
        logger.info(
            f"Initialized GaussianProcessOptimizer with "
            f"kernel={self.kernel_type}, acquisition={self.acquisition_type}"
        )
    
    def _create_kernel(self) -> Kernel:
        """
        Create GP kernel based on configuration.
        
        Returns:
            Configured kernel for Gaussian Process
            
        Raises:
            ValueError: If kernel type is unknown
        """
        if self.kernel_type == 'matern':
            # Matern kernel with nu=2.5 (twice differentiable)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5
            )
        
        elif self.kernel_type == 'matern52':
            # Matern kernel with nu=5/2
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5
            )
        
        elif self.kernel_type == 'matern32':
            # Matern kernel with nu=3/2
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=1.5
            )
        
        elif self.kernel_type == 'rbf':
            # Radial Basis Function (squared exponential)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2)
            )
        
        else:
            raise ValueError(
                f"Unknown kernel type: {self.kernel_type}. "
                f"Choose from: 'matern', 'matern52', 'matern32', 'rbf'"
            )
        
        # Add white noise kernel for numerical stability
        kernel = kernel + WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(1e-10, 1e-1)
        )
        
        return kernel
    
    def ask(self) -> List[ModelGraph]:
        """
        Generate next candidate architecture using acquisition function.
        
        During initial phase (< n_initial_points), samples randomly.
        After that, uses GP + acquisition function to select promising candidates.
        
        Returns:
            List containing single ModelGraph candidate
        """
        # Random exploration during initialization
        if len(self.y_observed) < self.n_initial_points:
            candidate = self.search_space.sample()
            logger.debug(
                f"Random sampling ({len(self.y_observed)}/{self.n_initial_points})"
            )
            return [candidate]
        
        # Fit GP on all observations
        self._fit_gp()
        
        # Optimize acquisition function
        x_next = self._optimize_acquisition()
        
        # Decode to architecture
        candidate = self._decode_architecture(x_next)
        
        logger.debug(
            f"Selected candidate via {self.acquisition_type} acquisition "
            f"(iteration {len(self.y_observed)})"
        )
        
        return [candidate]
    
    def tell(self, results: List[Tuple[ModelGraph, float]]) -> None:
        """
        Update GP with new evaluation results.
        
        Args:
            results: List of (graph, fitness) tuples from evaluation
        """
        for graph, fitness in results:
            # Encode architecture
            x = self._encode_architecture(graph)
            
            # Store observation
            self.X_observed.append(x)
            self.y_observed.append(fitness)
            
            # Update history
            self.history.append({
                'generation': self.generation,
                'genome': graph,
                'fitness': fitness,
                'encoding': x
            })
            
            logger.debug(f"Added observation: fitness={fitness:.4f}")
        
        # Mark GP as needing refit
        self._gp_fitted = False
        
        self.generation += 1
    
    def _fit_gp(self) -> None:
        """
        Fit Gaussian Process on observed data.
        
        Fits the GP surrogate model on all (X, y) observations,
        including hyperparameter optimization via maximum likelihood.
        """
        if self._gp_fitted and len(self.X_observed) > 0:
            return  # Already fitted with current data
        
        if len(self.X_observed) == 0:
            logger.warning("No observations to fit GP")
            return
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        logger.debug(f"Fitting GP on {len(X)} observations")
        
        try:
            self.gp.fit(X, y)
            self._gp_fitted = True
            
            # Log learned hyperparameters
            logger.debug(f"GP kernel: {self.gp.kernel_}")
            logger.debug(f"GP log-likelihood: {self.gp.log_marginal_likelihood():.2f}")
            
        except Exception as e:
            logger.error(f"GP fitting failed: {e}")
            raise
    
    def _optimize_acquisition(self) -> np.ndarray:
        """
        Find architecture encoding that maximizes acquisition function.
        
        Uses the acquisition optimizer to find the point with highest
        acquisition value (most promising for evaluation).
        
        Returns:
            Optimal architecture encoding
        """
        # Get current best fitness
        f_best = max(self.y_observed)
        
        # Create acquisition function
        def acquisition_fn(x: np.ndarray) -> np.ndarray:
            """
            Evaluate acquisition at given point(s).
            
            Args:
                x: Architecture encoding(s), shape (n_samples, n_features)
                
            Returns:
                Acquisition values, shape (n_samples,)
            """
            # Ensure 2D
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Get GP predictions
            mu, sigma = self.gp.predict(x, return_std=True)
            
            # Compute acquisition values
            if self.acquisition_type == 'ei':
                acq_values = expected_improvement(mu, sigma, f_best, self.xi)
            elif self.acquisition_type == 'ucb':
                acq_values = upper_confidence_bound(mu, sigma, self.kappa)
            elif self.acquisition_type == 'pi':
                acq_values = probability_of_improvement(mu, sigma, f_best, self.xi)
            else:
                raise ValueError(f"Unknown acquisition: {self.acquisition_type}")
            
            return acq_values
        
        # Get bounds for optimization
        bounds = self._get_encoding_bounds()
        
        # Optimize acquisition
        try:
            x_next = self.acq_optimizer.optimize(
                acquisition_fn=acquisition_fn,
                bounds=bounds,
                n_candidates=1
            )
            
            # Extract single candidate
            if x_next.ndim == 2:
                x_next = x_next[0]
            
            # Log acquisition value
            acq_value = acquisition_fn(x_next.reshape(1, -1))[0]
            logger.debug(f"Acquisition optimum: {acq_value:.6f}")
            
            return x_next
            
        except Exception as e:
            logger.warning(f"Acquisition optimization failed: {e}. Using random sample.")
            # Fallback to random sample
            return np.array([
                np.random.uniform(low, high) for low, high in bounds
            ])
    
    def predict(
        self,
        graphs: List[ModelGraph],
        return_std: bool = False
    ) -> np.ndarray:
        """
        Predict fitness for given architectures using GP.
        
        Args:
            graphs: List of ModelGraph instances
            return_std: If True, return (mean, std), else just mean
            
        Returns:
            Predicted fitness values (and optionally standard deviations)
            
        Example:
            >>> graphs = [space.sample() for _ in range(5)]
            >>> predictions = optimizer.predict(graphs, return_std=True)
            >>> means, stds = predictions
        """
        # Ensure GP is fitted
        if not self._gp_fitted:
            self._fit_gp()
        
        # Encode architectures
        X = np.array([self._encode_architecture(g) for g in graphs])
        
        # Predict
        if return_std:
            mu, sigma = self.gp.predict(X, return_std=True)
            return mu, sigma
        else:
            mu = self.gp.predict(X, return_std=False)
            return mu
    
    def get_best_predicted(self, n_samples: int = 100) -> ModelGraph:
        """
        Sample architectures and return the one with highest predicted fitness.
        
        Useful for suggesting good architectures without evaluation.
        
        Args:
            n_samples: Number of random samples to evaluate
            
        Returns:
            Architecture with highest predicted mean fitness
        """
        # Sample random architectures
        candidates = [self.search_space.sample() for _ in range(n_samples)]
        
        # Predict fitness
        predictions = self.predict(candidates, return_std=False)
        
        # Return best
        best_idx = np.argmax(predictions)
        return candidates[best_idx]
    
    def get_uncertainty_map(
        self,
        n_samples: int = 100
    ) -> Tuple[List[ModelGraph], np.ndarray, np.ndarray]:
        """
        Sample architectures and get prediction uncertainty.
        
        Useful for understanding which regions of search space are uncertain.
        
        Args:
            n_samples: Number of random samples
            
        Returns:
            (graphs, means, stds) tuple
        """
        candidates = [self.search_space.sample() for _ in range(n_samples)]
        means, stds = self.predict(candidates, return_std=True)
        return candidates, means, stds
    
    def get_gp_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the fitted GP.
        
        Returns:
            Dictionary with GP information
        """
        if not self._gp_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'n_observations': len(self.X_observed),
            'kernel': str(self.gp.kernel_),
            'log_marginal_likelihood': self.gp.log_marginal_likelihood(),
            'best_observed': max(self.y_observed) if self.y_observed else None,
            'mean_observed': np.mean(self.y_observed) if self.y_observed else None,
            'std_observed': np.std(self.y_observed) if self.y_observed else None,
        }
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence.
        
        Shows best fitness over iterations.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
            return
        
        if not self.y_observed:
            logger.warning("No observations to plot")
            return
        
        # Compute best-so-far
        best_so_far = []
        current_best = -np.inf
        for y in self.y_observed:
            if y > current_best:
                current_best = y
            best_so_far.append(current_best)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(best_so_far, 'b-', linewidth=2, label='Best fitness')
        plt.plot(self.y_observed, 'ko', alpha=0.3, markersize=4, label='Observations')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('Bayesian Optimization Convergence', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_acquisition_landscape(
        self,
        n_samples: int = 1000,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize acquisition function landscape (2D projection).
        
        Projects high-dimensional acquisition to 2D for visualization.
        
        Args:
            n_samples: Number of points to sample
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
            return
        
        if not self._gp_fitted:
            logger.warning("GP not fitted, cannot plot acquisition")
            return
        
        # Sample random architectures
        candidates = [self.search_space.sample() for _ in range(n_samples)]
        X = np.array([self._encode_architecture(g) for g in candidates])
        
        # Get GP predictions
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # Compute acquisition values
        f_best = max(self.y_observed)
        if self.acquisition_type == 'ei':
            acq = expected_improvement(mu, sigma, f_best, self.xi)
        elif self.acquisition_type == 'ucb':
            acq = upper_confidence_bound(mu, sigma, self.kappa)
        else:
            acq = probability_of_improvement(mu, sigma, f_best, self.xi)
        
        # Project to 2D using PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Mean predictions
        scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=mu, cmap='viridis', s=20)
        axes[0].set_title('GP Mean Predictions')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Uncertainty
        scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=sigma, cmap='plasma', s=20)
        axes[1].set_title('GP Uncertainty (σ)')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Acquisition
        scatter3 = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=acq, cmap='coolwarm', s=20)
        axes[2].set_title(f'Acquisition ({self.acquisition_type.upper()})')
        axes[2].set_xlabel('PC1')
        axes[2].set_ylabel('PC2')
        plt.colorbar(scatter3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Acquisition landscape saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GaussianProcessOptimizer("
            f"kernel={self.kernel_type}, "
            f"acquisition={self.acquisition_type}, "
            f"n_obs={len(self.y_observed)})"
        )


# Convenience function for quick GP optimization
def optimize_with_gp(
    search_space: SearchSpace,
    evaluator: Any,
    n_iterations: int = 50,
    n_initial: int = 10,
    acquisition: str = 'ei',
    kernel: str = 'matern',
    verbose: bool = True
) -> Individual:
    """
    Quick Gaussian Process optimization with sensible defaults.
    
    Args:
        search_space: SearchSpace to optimize over
        evaluator: Fitness evaluation function
        n_iterations: Total number of evaluations
        n_initial: Random samples before GP
        acquisition: Acquisition function ('ei', 'ucb', 'pi')
        kernel: GP kernel ('matern', 'rbf')
        verbose: Print progress
        
    Returns:
        Best Individual found
        
    Example:
        >>> from morphml.core.dsl import create_cnn_space
        >>> space = create_cnn_space(num_classes=10)
        >>> best = optimize_with_gp(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     n_iterations=50,
        ...     acquisition='ei'
        ... )
    """
    optimizer = GaussianProcessOptimizer(
        search_space=search_space,
        config={
            'n_initial_points': n_initial,
            'acquisition': acquisition,
            'kernel': kernel,
            'max_iterations': n_iterations
        }
    )
    
    def callback(iteration: int, best: Individual, history: List) -> None:
        if verbose:
            print(f"Iteration {iteration}: Best fitness = {best.fitness:.4f}")
    
    best = optimizer.optimize(
        evaluator=evaluator,
        max_evaluations=n_iterations,
        callback=callback if verbose else None
    )
    
    return best
