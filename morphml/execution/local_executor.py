"""Local executor for running experiments on a single machine.

Provides LocalExecutor class for orchestrating the complete
experiment lifecycle on a single machine.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from morphml.config import get_config
from morphml.core.dsl.search_space import SearchSpace
from morphml.core.search.search_engine import SearchEngine
from morphml.evaluation.heuristic import HeuristicEvaluator
from morphml.logging_config import get_logger
from morphml.utils.checkpoint import Checkpoint

logger = get_logger(__name__)


class LocalExecutor:
    """
    Executes experiments on local machine.

    Manages the complete experiment lifecycle:
    1. Initialize search from search space
    2. Run optimizer for multiple generations
    3. Evaluate candidates using evaluator
    4. Track results and save checkpoints
    5. Return best model

    Example:
        >>> executor = LocalExecutor()
        >>> results = executor.run(
        ...     search_space=space,
        ...     optimizer=ga,
        ...     evaluator=evaluator,
        ...     max_evaluations=1000
        ... )
        >>> best_model = results['best_graph']
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize local executor.

        Args:
            config: Configuration dictionary (uses global config if None)
        """
        self.config = config or get_config()
        self.logger = logger
        self.checkpoint_dir = Path(self.config.get("checkpoints_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        search_space: SearchSpace,
        optimizer: SearchEngine,
        evaluator: Optional[Callable] = None,
        max_evaluations: int = 1000,
        checkpoint_interval: int = 100,
        callbacks: Optional[list] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete experiment.

        Args:
            search_space: Search space definition
            optimizer: Optimizer instance (e.g., GeneticAlgorithm)
            evaluator: Evaluator for fitness computation (defaults to HeuristicEvaluator)
            max_evaluations: Budget in number of evaluations
            checkpoint_interval: Save checkpoint every N evaluations
            callbacks: Optional callbacks to call each generation
            verbose: Print progress information

        Returns:
            Dictionary with results:
                - best_graph: Best architecture found
                - best_fitness: Best fitness value
                - num_evaluations: Total evaluations performed
                - elapsed_time: Total time in seconds
                - final_generation: Final generation number
                - statistics: Final statistics
                - history: Complete history

        Example:
            >>> executor = LocalExecutor()
            >>> results = executor.run(
            ...     search_space=space,
            ...     optimizer=GeneticAlgorithm(space, config),
            ...     max_evaluations=500
            ... )
        """
        if verbose:
            self.logger.info("=" * 70)
            self.logger.info("Starting Experiment")
            self.logger.info("=" * 70)
            self.logger.info(f"Search space: {search_space.name}")
            self.logger.info(f"Optimizer: {optimizer.__class__.__name__}")
            self.logger.info(f"Budget: {max_evaluations} evaluations")

        # Use heuristic evaluator if none provided
        if evaluator is None:
            if verbose:
                self.logger.info("No evaluator provided, using HeuristicEvaluator")
            evaluator = HeuristicEvaluator()

        # Configure optimizer
        optimizer.config["max_evaluations"] = max_evaluations

        # Track progress
        start_time = time.time()
        num_evaluations = 0
        last_checkpoint = 0

        # Initialize population
        if verbose:
            self.logger.info("Initializing population...")

        population = optimizer.initialize_population(optimizer.config.get("population_size", 50))

        if verbose:
            self.logger.info(f"Population initialized with {len(population)} individuals")

        # Evaluate initial population
        for individual in population.individuals:
            if individual.fitness is None:
                individual.set_fitness(evaluator(individual.graph))
                num_evaluations += 1

        # Track best
        optimizer._update_best(population)
        optimizer._record_history(population)

        if verbose:
            self.logger.info(
                f"Initial population evaluated. Best fitness: {population.best_fitness():.4f}"
            )
            self.logger.info("")
            self.logger.info("Starting evolution...")

        # Main evolution loop
        generation = 0
        callbacks = callbacks or []

        while num_evaluations < max_evaluations and not optimizer.should_stop():
            # Evolution step
            population = optimizer.step(population, evaluator)

            # Evaluate new individuals
            for individual in population.individuals:
                if individual.fitness is None and num_evaluations < max_evaluations:
                    individual.set_fitness(evaluator(individual.graph))
                    num_evaluations += 1

            # Update tracking
            optimizer._update_best(population)
            optimizer._record_history(population)
            optimizer.generation = generation
            optimizer.num_evaluations = num_evaluations

            # Call callbacks
            for callback in callbacks:
                callback(optimizer, population)

            # Checkpoint
            if num_evaluations - last_checkpoint >= checkpoint_interval:
                self._save_checkpoint(optimizer, num_evaluations)
                last_checkpoint = num_evaluations

            # Log progress
            if verbose and generation % 10 == 0:
                stats = optimizer.get_statistics()
                elapsed = time.time() - start_time
                evals_per_sec = num_evaluations / elapsed if elapsed > 0 else 0

                self.logger.info(
                    f"Gen {generation:3d}: "
                    f"best={stats['best_fitness']:.4f}, "
                    f"mean={stats['mean_fitness']:.4f}, "
                    f"evals={num_evaluations}/{max_evaluations}, "
                    f"({evals_per_sec:.1f} eval/s)"
                )

            generation += 1

        # Finalize
        elapsed_time = time.time() - start_time
        best_graph = optimizer.get_best()
        final_stats = optimizer.get_statistics()

        results = {
            "best_graph": best_graph,
            "best_fitness": final_stats["best_fitness"],
            "num_evaluations": num_evaluations,
            "elapsed_time": elapsed_time,
            "final_generation": generation,
            "statistics": final_stats,
            "history": optimizer.get_history(),
        }

        if verbose:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("Experiment Complete")
            self.logger.info("=" * 70)
            self.logger.info(f"Best fitness: {final_stats['best_fitness']:.4f}")
            self.logger.info(f"Evaluations: {num_evaluations}")
            self.logger.info(f"Time: {elapsed_time:.2f}s")
            self.logger.info(f"Avg time per evaluation: {elapsed_time/num_evaluations*1000:.2f}ms")
            self.logger.info("=" * 70)

        return results

    def _save_checkpoint(self, optimizer: SearchEngine, num_evals: int) -> None:
        """
        Save optimizer state to checkpoint.

        Args:
            optimizer: Optimizer to checkpoint
            num_evals: Current number of evaluations
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{num_evals}.pkl"

        try:
            Checkpoint.save(optimizer, str(checkpoint_path))
            self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> SearchEngine:
        """
        Load optimizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Restored optimizer

        Example:
            >>> executor = LocalExecutor()
            >>> optimizer = executor.load_checkpoint("checkpoint_500.pkl")
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        optimizer = Checkpoint.load(str(checkpoint_path))

        self.logger.info(
            f"Checkpoint loaded: generation {optimizer.generation}, "
            f"{optimizer.num_evaluations} evaluations"
        )

        return optimizer

    def resume(
        self,
        checkpoint_path: str,
        search_space: SearchSpace,
        evaluator: Optional[Callable] = None,
        max_additional_evaluations: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Resume experiment from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            search_space: Search space (must match checkpointed experiment)
            evaluator: Evaluator for fitness computation
            max_additional_evaluations: Additional evaluation budget
            verbose: Print progress information

        Returns:
            Results dictionary

        Example:
            >>> executor = LocalExecutor()
            >>> results = executor.resume(
            ...     "checkpoint_500.pkl",
            ...     search_space=space,
            ...     max_additional_evaluations=500
            ... )
        """
        # Load checkpoint
        optimizer = self.load_checkpoint(checkpoint_path)

        # Calculate remaining budget
        prev_evals = optimizer.num_evaluations
        total_budget = prev_evals + max_additional_evaluations

        if verbose:
            self.logger.info(f"Resuming from {prev_evals} evaluations")
            self.logger.info(f"Additional budget: {max_additional_evaluations}")
            self.logger.info(f"Total budget: {total_budget}")

        # Continue execution
        return self.run(
            search_space=search_space,
            optimizer=optimizer,
            evaluator=evaluator,
            max_evaluations=total_budget,
            verbose=verbose,
        )


# Convenience function
def run_experiment(
    search_space: SearchSpace,
    optimizer: SearchEngine,
    evaluator: Optional[Callable] = None,
    max_evaluations: int = 1000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run experiment.

    Args:
        search_space: Search space definition
        optimizer: Optimizer instance
        evaluator: Evaluator function
        max_evaluations: Evaluation budget
        **kwargs: Additional arguments for LocalExecutor.run()

    Returns:
        Results dictionary

    Example:
        >>> results = run_experiment(
        ...     search_space=space,
        ...     optimizer=ga,
        ...     max_evaluations=1000
        ... )
    """
    executor = LocalExecutor()
    return executor.run(
        search_space=search_space,
        optimizer=optimizer,
        evaluator=evaluator,
        max_evaluations=max_evaluations,
        **kwargs,
    )
