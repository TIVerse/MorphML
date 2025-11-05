"""SMAC (Sequential Model-based Algorithm Configuration) optimizer.

SMAC uses Random Forest instead of Gaussian Process as the surrogate model,
making it more scalable and robust for neural architecture search with mixed
continuous/categorical spaces.

Key advantages:
- Scales to high dimensions better than GP
- Handles categorical variables natively
- More robust to noisy evaluations
- Efficient with limited data

Reference:
    Hutter, F., et al. "Sequential Model-Based Optimization for General
    Algorithm Configuration." LION 2011.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.core.search import Individual
from morphml.logging_config import get_logger
from morphml.optimizers.bayesian.acquisition import (
    AcquisitionOptimizer,
    expected_improvement,
)
from morphml.optimizers.bayesian.base import BaseBayesianOptimizer

logger = get_logger(__name__)


class SMACOptimizer(BaseBayesianOptimizer):
    """
    Sequential Model-based Algorithm Configuration optimizer.
    
    SMAC uses Random Forest as a surrogate model instead of Gaussian Process.
    The Random Forest provides:
    - Predictions via ensemble averaging
    - Uncertainty via variance across trees
    - Scalability to high dimensions
    - Natural handling of mixed spaces
    
    Algorithm:
    1. Initialize with random samples
    2. Fit Random Forest on observed (x, y) pairs
    3. Optimize acquisition function (typically EI)
    4. Evaluate selected architecture
    5. Update forest and repeat
    
    Configuration:
        n_initial_points: Random samples before RF (default: 15)
        n_estimators: Number of trees in forest (default: 50)
        max_depth: Maximum tree depth (default: 10)
        min_samples_split: Min samples to split node (default: 2)
        acquisition: Acquisition function (default: 'ei')
        xi: EI exploration parameter (default: 0.01)
        acq_optimizer: Acquisition optimization method (default: 'random')
        
    Example:
        >>> from morphml.optimizers.bayesian import SMACOptimizer
        >>> optimizer = SMACOptimizer(
        ...     search_space=space,
        ...     config={
        ...         'n_estimators': 50,
        ...         'max_depth': 10,
        ...         'acquisition': 'ei'
        ...     }
        ... )
        >>> best = optimizer.optimize(evaluator, max_evaluations=100)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SMAC optimizer.
        
        Args:
            search_space: SearchSpace defining architecture options
            config: Configuration dictionary with optional keys:
                - n_initial_points: Initial random samples
                - n_estimators: Number of RF trees
                - max_depth: Maximum tree depth
                - min_samples_split: Min samples for split
                - acquisition: Acquisition function type
                - xi: EI exploration parameter
                - acq_optimizer: Acquisition optimization method
        """
        super().__init__(search_space, config or {})
        
        # Random Forest configuration
        self.n_estimators = self.config.get('n_estimators', 50)
        self.max_depth = self.config.get('max_depth', 10)
        self.min_samples_split = self.config.get('min_samples_split', 2)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 1)
        
        # Acquisition configuration
        self.acquisition_type = self.config.get('acquisition', 'ei')
        self.xi = self.config.get('xi', 0.01)
        
        # Override n_initial (SMAC typically needs fewer than TPE)
        self.n_initial_points = self.config.get('n_initial_points', 15)
        
        # Acquisition optimizer (random search works well with RF)
        self.acq_optimizer_method = self.config.get('acq_optimizer', 'random')
        self.acq_n_samples = self.config.get('acq_n_samples', 1000)
        
        # Initialize Random Forest
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Initialize acquisition optimizer
        self.acq_optimizer = AcquisitionOptimizer(
            method=self.acq_optimizer_method,
            n_samples=self.acq_n_samples,
            random_state=self.random_state
        )
        
        # Observation storage
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        
        # Track whether RF is fitted
        self._rf_fitted = False
        
        logger.info(
            f"Initialized SMACOptimizer with "
            f"n_estimators={self.n_estimators}, max_depth={self.max_depth}"
        )
    
    def ask(self) -> List[ModelGraph]:
        """
        Generate next candidate using Random Forest + acquisition.
        
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
        
        # Fit Random Forest
        self._fit_rf()
        
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
        Update Random Forest with new evaluation results.
        
        Args:
            results: List of (graph, fitness) tuples
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
        
        # Mark RF as needing refit
        self._rf_fitted = False
        
        self.generation += 1
    
    def _fit_rf(self) -> None:
        """
        Fit Random Forest on observed data.
        
        Trains the Random Forest surrogate on all (X, y) observations.
        """
        if self._rf_fitted and len(self.X_observed) > 0:
            return  # Already fitted with current data
        
        if len(self.X_observed) == 0:
            logger.warning("No observations to fit Random Forest")
            return
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        logger.debug(f"Fitting Random Forest on {len(X)} observations")
        
        try:
            self.rf.fit(X, y)
            self._rf_fitted = True
            
            # Log forest statistics
            if hasattr(self.rf, 'estimators_'):
                avg_nodes = np.mean([
                    tree.tree_.node_count for tree in self.rf.estimators_
                ])
                logger.debug(f"RF fitted: {len(self.rf.estimators_)} trees, "
                           f"avg {avg_nodes:.0f} nodes per tree")
            
        except Exception as e:
            logger.error(f"Random Forest fitting failed: {e}")
            raise
    
    def _optimize_acquisition(self) -> np.ndarray:
        """
        Find architecture encoding that maximizes acquisition function.
        
        Returns:
            Optimal architecture encoding
        """
        # Get current best fitness
        f_best = max(self.y_observed)
        
        # Create acquisition function
        def acquisition_fn(x: np.ndarray) -> np.ndarray:
            """
            Evaluate acquisition at given point(s).
            
            For Random Forest, uncertainty is estimated from variance
            across trees in the ensemble.
            
            Args:
                x: Architecture encoding(s), shape (n_samples, n_features)
                
            Returns:
                Acquisition values, shape (n_samples,)
            """
            # Ensure 2D
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Get RF predictions
            mu, sigma = self._predict_with_uncertainty(x)
            
            # Compute acquisition values (EI)
            acq_values = expected_improvement(mu, sigma, f_best, self.xi)
            
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
    
    def _predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty using Random Forest.
        
        Uncertainty is estimated as the standard deviation of predictions
        across all trees in the forest.
        
        Args:
            X: Input points, shape (n_samples, n_features)
            
        Returns:
            (means, stds) tuple of shape (n_samples,)
        """
        # Get predictions from all trees
        predictions = np.array([
            tree.predict(X) for tree in self.rf.estimators_
        ])
        
        # Mean across trees
        mu = np.mean(predictions, axis=0)
        
        # Standard deviation across trees
        sigma = np.std(predictions, axis=0)
        
        # Add small constant to avoid zero uncertainty
        sigma = sigma + 1e-6
        
        return mu, sigma
    
    def predict(
        self,
        graphs: List[ModelGraph],
        return_std: bool = False
    ) -> np.ndarray:
        """
        Predict fitness for given architectures using Random Forest.
        
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
        # Ensure RF is fitted
        if not self._rf_fitted:
            self._fit_rf()
        
        # Encode architectures
        X = np.array([self._encode_architecture(g) for g in graphs])
        
        # Predict
        if return_std:
            mu, sigma = self._predict_with_uncertainty(X)
            return mu, sigma
        else:
            mu = self.rf.predict(X)
            return mu
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from Random Forest.
        
        Useful for understanding which architecture components matter most.
        
        Returns:
            Feature importance array
        """
        if not self._rf_fitted:
            logger.warning("RF not fitted, no feature importances available")
            return np.array([])
        
        return self.rf.feature_importances_
    
    def get_rf_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the fitted Random Forest.
        
        Returns:
            Dictionary with RF information
        """
        if not self._rf_fitted:
            return {'fitted': False}
        
        # Tree statistics
        tree_depths = [tree.tree_.max_depth for tree in self.rf.estimators_]
        tree_nodes = [tree.tree_.node_count for tree in self.rf.estimators_]
        
        return {
            'fitted': True,
            'n_observations': len(self.X_observed),
            'n_estimators': len(self.rf.estimators_),
            'avg_tree_depth': np.mean(tree_depths),
            'max_tree_depth': max(tree_depths),
            'avg_tree_nodes': np.mean(tree_nodes),
            'best_observed': max(self.y_observed) if self.y_observed else None,
            'mean_observed': np.mean(self.y_observed) if self.y_observed else None,
            'std_observed': np.std(self.y_observed) if self.y_observed else None,
        }
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence.
        
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
        plt.plot(self.y_observed, 'ro', alpha=0.3, markersize=4, label='Observations')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('SMAC Optimization Convergence', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        top_k: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importances from Random Forest.
        
        Args:
            top_k: Number of top features to show
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
            return
        
        importances = self.get_feature_importances()
        
        if len(importances) == 0:
            logger.warning("No feature importances available")
            return
        
        # Get top-k features
        indices = np.argsort(importances)[-top_k:][::-1]
        values = importances[indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), values)
        plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_k} Feature Importances (Random Forest)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SMACOptimizer("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"n_obs={len(self.y_observed)})"
        )


# Convenience function for quick SMAC optimization
def optimize_with_smac(
    search_space: SearchSpace,
    evaluator: Any,
    n_iterations: int = 100,
    n_initial: int = 15,
    n_estimators: int = 50,
    verbose: bool = True
) -> Individual:
    """
    Quick SMAC optimization with sensible defaults.
    
    Args:
        search_space: SearchSpace to optimize over
        evaluator: Fitness evaluation function
        n_iterations: Total number of evaluations
        n_initial: Random samples before SMAC
        n_estimators: Number of trees in Random Forest
        verbose: Print progress
        
    Returns:
        Best Individual found
        
    Example:
        >>> from morphml.core.dsl import create_cnn_space
        >>> space = create_cnn_space(num_classes=10)
        >>> best = optimize_with_smac(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     n_iterations=100,
        ...     n_estimators=50
        ... )
    """
    optimizer = SMACOptimizer(
        search_space=search_space,
        config={
            'n_initial_points': n_initial,
            'n_estimators': n_estimators,
            'max_iterations': n_iterations
        }
    )
    
    def callback(iteration: int, best: Individual, history: List) -> None:
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Best fitness = {best.fitness:.4f}")
    
    best = optimizer.optimize(
        evaluator=evaluator,
        max_evaluations=n_iterations,
        callback=callback if verbose else None
    )
    
    return best
