"""Tree-structured Parzen Estimator (TPE) for Bayesian optimization.

TPE is a sequential model-based optimization algorithm that models p(x|y) instead
of p(y|x). It splits observations into "good" and "bad" based on a quantile,
then models the density of x for each group separately.

Key advantages over GP:
- Scales better to high dimensions
- Handles categorical variables naturally
- Computationally efficient
- Works well for neural architecture search

Reference:
    Bergstra, J., et al. "Algorithms for Hyper-Parameter Optimization." NIPS 2011.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde

from morphml.core.dsl import SearchSpace
from morphml.core.graph import GraphMutator, ModelGraph
from morphml.core.search import Individual
from morphml.logging_config import get_logger
from morphml.optimizers.bayesian.base import BaseBayesianOptimizer

logger = get_logger(__name__)


class TPEOptimizer(BaseBayesianOptimizer):
    """
    Tree-structured Parzen Estimator for Bayesian optimization.

    TPE takes a different approach than GP-based BO:
    1. Split observations into "good" (top γ quantile) and "bad" (rest)
    2. Model p(x|y=good) using kernel density estimation
    3. Model p(x|y=bad) similarly
    4. Select x that maximizes p(x|y=good) / p(x|y=bad)

    This approach:
    - Is more scalable than GP (O(n) vs O(n³))
    - Handles mixed continuous/discrete spaces naturally
    - Performs well on neural architecture search

    Configuration:
        n_initial_points: Random samples before TPE (default: 20)
        gamma: Quantile for good/bad split (default: 0.25)
        n_ei_candidates: Candidates to evaluate EI on (default: 24)
        bandwidth: KDE bandwidth (default: 'scott')
        prior_weight: Weight of prior in density estimation (default: 1.0)

    Example:
        >>> from morphml.optimizers.bayesian import TPEOptimizer
        >>> optimizer = TPEOptimizer(
        ...     search_space=space,
        ...     config={
        ...         'n_initial_points': 20,
        ...         'gamma': 0.25,
        ...         'n_ei_candidates': 24
        ...     }
        ... )
        >>> best = optimizer.optimize(evaluator, max_evaluations=100)
    """

    def __init__(self, search_space: SearchSpace, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TPE optimizer.

        Args:
            search_space: SearchSpace defining architecture options
            config: Configuration dictionary with optional keys:
                - n_initial_points: Initial random samples
                - gamma: Quantile for splitting good/bad
                - n_ei_candidates: Number of EI candidates
                - bandwidth: KDE bandwidth method
                - prior_weight: Prior weight for densities
        """
        super().__init__(search_space, config or {})

        # TPE-specific configuration
        self.gamma = self.config.get("gamma", 0.25)
        self.n_ei_candidates = self.config.get("n_ei_candidates", 24)
        self.bandwidth = self.config.get("bandwidth", "scott")
        self.prior_weight = self.config.get("prior_weight", 1.0)

        # Override n_initial_points (TPE typically needs more)
        self.n_initial_points = self.config.get("n_initial_points", 20)

        # Mutation for candidate generation
        self.mutator = GraphMutator()

        # Observation storage
        self.observations: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized TPEOptimizer with gamma={self.gamma}, "
            f"n_ei_candidates={self.n_ei_candidates}"
        )

    def ask(self) -> List[ModelGraph]:
        """
        Generate next candidate using TPE.

        Returns:
            List containing single ModelGraph candidate
        """
        # Random exploration during initialization
        if len(self.observations) < self.n_initial_points:
            candidate = self.search_space.sample()
            logger.debug(f"Random sampling ({len(self.observations)}/{self.n_initial_points})")
            return [candidate]

        # Split observations into good and bad
        good_obs, bad_obs = self._split_observations()

        logger.debug(f"TPE split: {len(good_obs)} good, {len(bad_obs)} bad observations")

        # Generate candidates and select best by EI
        best_candidate = None
        best_ei = -float("inf")

        for _ in range(self.n_ei_candidates):
            # Sample from good distribution
            candidate = self._sample_from_good(good_obs)

            # Compute expected improvement
            ei = self._compute_ei(candidate, good_obs, bad_obs)

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        logger.debug(f"Selected candidate with EI={best_ei:.4f}")

        return [best_candidate]

    def tell(self, results: List[Tuple[ModelGraph, float]]) -> None:
        """
        Update observations with new results.

        Args:
            results: List of (graph, fitness) tuples
        """
        for graph, fitness in results:
            # Encode architecture
            x = self._encode_architecture(graph)

            # Store observation
            self.observations.append(
                {"graph": graph, "fitness": fitness, "encoding": x, "generation": self.generation}
            )

            # Update history
            self.history.append(
                {"generation": self.generation, "genome": graph, "fitness": fitness, "encoding": x}
            )

            logger.debug(f"Added observation: fitness={fitness:.4f}")

        self.generation += 1

    def _split_observations(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Split observations into good and bad based on quantile.

        Returns:
            (good_observations, bad_observations) tuple
        """
        # Sort by fitness (descending)
        sorted_obs = sorted(self.observations, key=lambda x: x["fitness"], reverse=True)

        # Split at gamma quantile
        n_good = max(1, int(self.gamma * len(sorted_obs)))

        good_obs = sorted_obs[:n_good]
        bad_obs = sorted_obs[n_good:]

        return good_obs, bad_obs

    def _sample_from_good(self, good_obs: List[Dict]) -> ModelGraph:
        """
        Sample architecture from "good" distribution.

        Strategy: Pick random good observation and apply small mutations.

        Args:
            good_obs: List of good observations

        Returns:
            Sampled ModelGraph
        """
        if not good_obs:
            return self.search_space.sample()

        # Pick random good architecture as template
        template = random.choice(good_obs)["graph"]

        # Clone and mutate slightly
        candidate = self._mutate_slightly(template)

        return candidate

    def _mutate_slightly(self, graph: ModelGraph, n_mutations: int = 1) -> ModelGraph:
        """
        Apply small mutations to architecture.

        Args:
            graph: Template graph
            n_mutations: Number of mutations to apply

        Returns:
            Mutated graph
        """
        try:
            # Clone graph
            mutated = graph.clone()

            # Apply mutations
            for _ in range(n_mutations):
                mutation_type = random.choice(["modify_node", "add_node", "remove_node"])

                if mutation_type == "modify_node" and len(mutated.nodes) > 2:
                    self.mutator.mutate_node_params(mutated)
                elif mutation_type == "add_node":
                    self.mutator.add_node(mutated)
                elif mutation_type == "remove_node" and len(mutated.nodes) > 3:
                    self.mutator.remove_node(mutated)

            # Validate
            if mutated.is_valid_dag():
                return mutated
            else:
                return graph.clone()

        except Exception as e:
            logger.warning(f"Mutation failed: {e}. Returning original.")
            return graph.clone()

    def _compute_ei(
        self, candidate: ModelGraph, good_obs: List[Dict], bad_obs: List[Dict]
    ) -> float:
        """
        Compute expected improvement as ratio of densities.

        EI(x) ≈ p(x|y=good) / p(x|y=bad)

        Args:
            candidate: Candidate architecture
            good_obs: Good observations
            bad_obs: Bad observations

        Returns:
            Expected improvement value
        """
        # Encode candidate
        x = self._encode_architecture(candidate)

        # Estimate densities
        p_good = self._estimate_density(x, good_obs)
        p_bad = self._estimate_density(x, bad_obs)

        # Compute EI
        if p_bad == 0:
            return float("inf") if p_good > 0 else 0.0

        ei = p_good / p_bad

        return ei

    def _estimate_density(self, x: np.ndarray, observations: List[Dict]) -> float:
        """
        Estimate probability density at point x.

        Uses kernel density estimation (KDE) with Gaussian kernel.

        Args:
            x: Point to evaluate density at
            observations: List of observations for density estimation

        Returns:
            Estimated density value
        """
        if not observations:
            return 1e-10  # Small positive value

        # Extract encodings
        X_obs = np.array([obs["encoding"] for obs in observations])

        # Handle single observation
        if len(X_obs) == 1:
            # Gaussian centered at the single observation
            dist = np.linalg.norm(x - X_obs[0])
            return np.exp(-0.5 * dist**2)

        try:
            # Kernel density estimation
            kde = gaussian_kde(X_obs.T, bw_method=self.bandwidth)

            density = kde(x.reshape(-1, 1))[0]

            # Add prior weight
            prior_density = 1.0 / (len(x) ** 0.5)  # Rough prior
            density = (density + self.prior_weight * prior_density) / (1 + self.prior_weight)

            return max(density, 1e-10)  # Avoid zero density

        except Exception as e:
            logger.warning(f"KDE failed: {e}. Using fallback.")
            # Fallback: distance-based density
            distances = np.linalg.norm(X_obs - x, axis=1)
            mean_dist = np.mean(distances)
            return np.exp(-mean_dist)

    def get_good_architectures(self, n: int = 10) -> List[ModelGraph]:
        """
        Get top-n architectures by fitness.

        Args:
            n: Number of architectures to return

        Returns:
            List of best ModelGraph instances
        """
        sorted_obs = sorted(self.observations, key=lambda x: x["fitness"], reverse=True)

        return [obs["graph"] for obs in sorted_obs[:n]]

    def get_density_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about density estimation.

        Returns:
            Dictionary with density statistics
        """
        if len(self.observations) < self.n_initial_points:
            return {"status": "initializing", "n_obs": len(self.observations)}

        good_obs, bad_obs = self._split_observations()

        return {
            "status": "active",
            "n_observations": len(self.observations),
            "n_good": len(good_obs),
            "n_bad": len(bad_obs),
            "gamma": self.gamma,
            "best_fitness": max(obs["fitness"] for obs in self.observations),
            "good_threshold": good_obs[-1]["fitness"] if good_obs else None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TPEOptimizer("
            f"gamma={self.gamma}, "
            f"n_ei_candidates={self.n_ei_candidates}, "
            f"n_obs={len(self.observations)})"
        )


# Convenience function for quick TPE optimization
def optimize_with_tpe(
    search_space: SearchSpace,
    evaluator: Any,
    n_iterations: int = 100,
    n_initial: int = 20,
    gamma: float = 0.25,
    verbose: bool = True,
) -> Individual:
    """
    Quick TPE optimization with sensible defaults.

    Args:
        search_space: SearchSpace to optimize over
        evaluator: Fitness evaluation function
        n_iterations: Total number of evaluations
        n_initial: Random samples before TPE
        gamma: Quantile for good/bad split
        verbose: Print progress

    Returns:
        Best Individual found

    Example:
        >>> from morphml.core.dsl import create_cnn_space
        >>> space = create_cnn_space(num_classes=10)
        >>> best = optimize_with_tpe(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     n_iterations=100,
        ...     gamma=0.25
        ... )
    """
    optimizer = TPEOptimizer(
        search_space=search_space,
        config={"n_initial_points": n_initial, "gamma": gamma, "max_iterations": n_iterations},
    )

    def callback(iteration: int, best: Individual, history: List) -> None:
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Best fitness = {best.fitness:.4f}")

    best = optimizer.optimize(
        evaluator=evaluator, max_evaluations=n_iterations, callback=callback if verbose else None
    )

    return best
