"""Acquisition functions for Bayesian optimization.

Acquisition functions balance exploration (uncertain regions) and exploitation
(promising regions) by quantifying the value of sampling at a given point.

Common acquisition functions:
- Expected Improvement (EI): Expected gain over current best
- Upper Confidence Bound (UCB): Optimistic estimate
- Probability of Improvement (PI): Probability of beating current best

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Expected Improvement acquisition function.
    
    EI balances exploration and exploitation by computing the expected
    amount by which a point improves over the current best.
    
    Formula:
        EI(x) = E[max(f(x) - f*, 0)]
              = (μ - f* - ξ) * Φ(Z) + σ * φ(Z)
        where Z = (μ - f* - ξ) / σ
    
    Args:
        mu: Predicted mean(s)
        sigma: Predicted standard deviation(s)
        f_best: Current best fitness value
        xi: Exploration parameter (higher = more exploration)
        
    Returns:
        Expected improvement value(s)
        
    Example:
        >>> mu = np.array([0.5, 0.7, 0.3])
        >>> sigma = np.array([0.1, 0.2, 0.05])
        >>> f_best = 0.6
        >>> ei = expected_improvement(mu, sigma, f_best, xi=0.01)
    """
    with np.errstate(divide='warn', invalid='warn'):
        # Compute improvement
        imp = mu - f_best - xi
        
        # Compute Z-score
        Z = imp / sigma
        
        # Expected improvement
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Handle zero sigma (no uncertainty)
        ei[sigma == 0.0] = 0.0
        
    return ei


def upper_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float = 2.576
) -> np.ndarray:
    """
    Upper Confidence Bound acquisition function.
    
    UCB provides an optimistic estimate by adding a multiple of the
    uncertainty to the predicted mean. The kappa parameter controls
    the exploration-exploitation trade-off.
    
    Formula:
        UCB(x) = μ(x) + κ * σ(x)
    
    Common kappa values:
    - 1.96: 95% confidence (moderate exploration)
    - 2.576: 99% confidence (high exploration)
    - 1.0: Balanced
    
    Args:
        mu: Predicted mean(s)
        sigma: Predicted standard deviation(s)
        kappa: Exploration parameter (higher = more exploration)
        
    Returns:
        UCB value(s)
        
    Example:
        >>> mu = np.array([0.5, 0.7])
        >>> sigma = np.array([0.1, 0.2])
        >>> ucb = upper_confidence_bound(mu, sigma, kappa=2.0)
    """
    return mu + kappa * sigma


def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Probability of Improvement acquisition function.
    
    PI computes the probability that a point will improve over the
    current best. More conservative than EI.
    
    Formula:
        PI(x) = P(f(x) > f*)
              = Φ((μ - f* - ξ) / σ)
    
    Args:
        mu: Predicted mean(s)
        sigma: Predicted standard deviation(s)
        f_best: Current best fitness value
        xi: Exploration parameter
        
    Returns:
        Probability of improvement value(s)
        
    Example:
        >>> mu = np.array([0.5, 0.7])
        >>> sigma = np.array([0.1, 0.2])
        >>> f_best = 0.6
        >>> pi = probability_of_improvement(mu, sigma, f_best)
    """
    with np.errstate(divide='warn', invalid='warn'):
        Z = (mu - f_best - xi) / sigma
        pi = norm.cdf(Z)
        
        # Handle zero sigma
        pi[sigma == 0.0] = 0.0
        
    return pi


def thompson_sampling(
    mu: np.ndarray,
    sigma: np.ndarray,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Thompson Sampling for acquisition.
    
    Sample from the posterior distribution and select the point
    with the highest sample. Naturally balances exploration/exploitation.
    
    Args:
        mu: Predicted mean(s)
        sigma: Predicted standard deviation(s)
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled values from posterior
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples = np.random.normal(mu, sigma)
    return samples


class AcquisitionOptimizer:
    """
    Optimizer for acquisition functions.
    
    Finds the point that maximizes the acquisition function value,
    which determines where to sample next.
    
    Attributes:
        method: Optimization method ('lbfgs', 'de', 'random')
        n_restarts: Number of random restarts for local optimization
        n_samples: Number of random samples for 'random' method
        
    Example:
        >>> def my_acquisition(x):
        ...     return expected_improvement(gp.predict(x), f_best=0.8)
        >>> optimizer = AcquisitionOptimizer(method='lbfgs', n_restarts=10)
        >>> x_next = optimizer.optimize(my_acquisition, bounds)
    """
    
    def __init__(
        self,
        method: str = 'lbfgs',
        n_restarts: int = 10,
        n_samples: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize acquisition optimizer.
        
        Args:
            method: Optimization method ('lbfgs', 'de', 'random')
            n_restarts: Number of random restarts for 'lbfgs'
            n_samples: Number of samples for 'random' method
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.n_restarts = n_restarts
        self.n_samples = n_samples
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def optimize(
        self,
        acquisition_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_candidates: int = 1
    ) -> np.ndarray:
        """
        Find point(s) that maximize acquisition function.
        
        Args:
            acquisition_fn: Function to maximize (takes array, returns scalar)
            bounds: List of (min, max) tuples for each dimension
            n_candidates: Number of candidates to return
            
        Returns:
            Best point(s) as numpy array of shape (n_candidates, n_dims)
            
        Raises:
            ValueError: If method is unknown
        """
        if self.method == 'lbfgs':
            return self._optimize_lbfgs(acquisition_fn, bounds, n_candidates)
        elif self.method == 'de':
            return self._optimize_differential_evolution(acquisition_fn, bounds)
        elif self.method == 'random':
            return self._optimize_random_search(acquisition_fn, bounds, n_candidates)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def _optimize_lbfgs(
        self,
        acquisition_fn: Callable,
        bounds: List[Tuple[float, float]],
        n_candidates: int
    ) -> np.ndarray:
        """
        Multi-start L-BFGS-B optimization.
        
        Performs multiple local optimizations from random starting points
        and returns the best result.
        """
        best_x = None
        best_value = -np.inf
        
        for _ in range(self.n_restarts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high)
                for low, high in bounds
            ])
            
            # Minimize negative (to maximize)
            try:
                result = minimize(
                    lambda x: -acquisition_fn(x.reshape(1, -1))[0],
                    x0=x0,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )
                
                value = -result.fun
                if value > best_value:
                    best_value = value
                    best_x = result.x
                    
            except Exception as e:
                logger.warning(f"L-BFGS-B optimization failed: {e}")
                continue
        
        if best_x is None:
            # Fallback to random sample
            best_x = np.array([np.random.uniform(low, high) for low, high in bounds])
        
        return best_x.reshape(1, -1) if n_candidates == 1 else best_x
    
    def _optimize_differential_evolution(
        self,
        acquisition_fn: Callable,
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Global optimization using Differential Evolution.
        
        More robust than L-BFGS but slower. Good for multimodal acquisitions.
        """
        try:
            result = differential_evolution(
                lambda x: -acquisition_fn(x.reshape(1, -1))[0],
                bounds=bounds,
                maxiter=100,
                seed=self.random_state,
                workers=1,
                polish=True
            )
            return result.x.reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"Differential evolution failed: {e}")
            # Fallback to random
            return np.array([
                [np.random.uniform(low, high) for low, high in bounds]
            ])
    
    def _optimize_random_search(
        self,
        acquisition_fn: Callable,
        bounds: List[Tuple[float, float]],
        n_candidates: int
    ) -> np.ndarray:
        """
        Random search: sample many points and pick best.
        
        Simple but surprisingly effective baseline.
        """
        # Generate random candidates
        n_dims = len(bounds)
        candidates = np.zeros((self.n_samples, n_dims))
        
        for i, (low, high) in enumerate(bounds):
            candidates[:, i] = np.random.uniform(low, high, self.n_samples)
        
        # Evaluate all candidates
        values = np.array([acquisition_fn(x.reshape(1, -1))[0] for x in candidates])
        
        # Return top n_candidates
        top_indices = np.argsort(values)[-n_candidates:][::-1]
        
        return candidates[top_indices]


def get_acquisition_function(
    name: str,
    **kwargs
) -> Callable:
    """
    Factory function for acquisition functions.
    
    Args:
        name: Acquisition function name ('ei', 'ucb', 'pi', 'ts')
        **kwargs: Additional parameters for the acquisition function
        
    Returns:
        Acquisition function
        
    Example:
        >>> acq_fn = get_acquisition_function('ei', f_best=0.8, xi=0.01)
        >>> value = acq_fn(mu=0.9, sigma=0.1)
    """
    if name.lower() == 'ei':
        f_best = kwargs.get('f_best', 0.0)
        xi = kwargs.get('xi', 0.01)
        return lambda mu, sigma: expected_improvement(mu, sigma, f_best, xi)
    
    elif name.lower() == 'ucb':
        kappa = kwargs.get('kappa', 2.576)
        return lambda mu, sigma: upper_confidence_bound(mu, sigma, kappa)
    
    elif name.lower() == 'pi':
        f_best = kwargs.get('f_best', 0.0)
        xi = kwargs.get('xi', 0.01)
        return lambda mu, sigma: probability_of_improvement(mu, sigma, f_best, xi)
    
    elif name.lower() == 'ts':
        random_state = kwargs.get('random_state', None)
        return lambda mu, sigma: thompson_sampling(mu, sigma, random_state)
    
    else:
        raise ValueError(
            f"Unknown acquisition function: {name}. "
            f"Choose from: 'ei', 'ucb', 'pi', 'ts'"
        )
