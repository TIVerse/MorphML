"""Particle Swarm Optimization (PSO) for neural architecture search.

PSO is a swarm intelligence algorithm inspired by social behavior of bird flocking
and fish schooling. Particles move through search space guided by their personal
best positions and the global best position.

Key Features:
- Swarm intelligence
- Velocity-based movement
- Cognitive and social components
- No gradient information needed
- Works in continuous spaces

Reference:
    Kennedy, J., and Eberhart, R. "Particle Swarm Optimization." 
    IEEE International Conference on Neural Networks, 1995.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.search import Individual
from morphml.logging_config import get_logger
from morphml.optimizers.evolutionary.encoding import ArchitectureEncoder

logger = get_logger(__name__)


@dataclass
class Particle:
    """
    Particle in PSO swarm.
    
    Each particle has:
    - Current position in search space
    - Current velocity
    - Personal best position found so far
    - Personal best fitness
    
    Attributes:
        position: Current position vector
        velocity: Current velocity vector
        best_position: Personal best position
        best_fitness: Personal best fitness value
        fitness: Current fitness
        
    Example:
        >>> particle = Particle(
        ...     position=np.random.rand(50),
        ...     velocity=np.random.rand(50) * 0.1,
        ...     best_position=np.random.rand(50),
        ...     best_fitness=-np.inf
        ... )
    """
    
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = -np.inf
    fitness: float = 0.0


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for architecture search.
    
    PSO uses a swarm of particles that move through continuous search space.
    Each particle:
    1. Remembers its personal best position (cognitive component)
    2. Is attracted to the global best position (social component)
    3. Has inertia that maintains its previous direction
    
    Update Equations:
        v_i(t+1) = w*v_i(t) + c1*r1*(p_i - x_i(t)) + c2*r2*(g - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
    
    where:
    - v_i: velocity of particle i
    - x_i: position of particle i  
    - p_i: personal best of particle i
    - g: global best position
    - w: inertia weight (controls exploration vs exploitation)
    - c1: cognitive coefficient (attraction to personal best)
    - c2: social coefficient (attraction to global best)
    - r1, r2: random numbers in [0,1]
    
    Configuration:
        num_particles: Swarm size (default: 30)
        max_iterations: Maximum iterations (default: 100)
        w: Inertia weight (default: 0.7)
        w_min: Minimum inertia (default: 0.4)
        w_max: Maximum inertia (default: 0.9)
        c1: Cognitive coefficient (default: 1.5)
        c2: Social coefficient (default: 1.5)
        max_velocity: Velocity clamping (default: 0.5)
        
    Example:
        >>> from morphml.optimizers.evolutionary import ParticleSwarmOptimizer
        >>> optimizer = ParticleSwarmOptimizer(
        ...     search_space=space,
        ...     config={
        ...         'num_particles': 30,
        ...         'max_iterations': 100,
        ...         'w': 0.7,
        ...         'c1': 1.5,
        ...         'c2': 1.5
        ...     }
        ... )
        >>> best = optimizer.optimize(evaluator)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PSO optimizer.
        
        Args:
            search_space: SearchSpace for architecture sampling
            config: Configuration dictionary
        """
        self.search_space = search_space
        self.config = config or {}
        
        # PSO parameters
        self.num_particles = self.config.get('num_particles', 30)
        self.max_iterations = self.config.get('max_iterations', 100)
        
        # Inertia weight (can be adaptive)
        self.w = self.config.get('w', 0.7)
        self.w_min = self.config.get('w_min', 0.4)
        self.w_max = self.config.get('w_max', 0.9)
        self.adaptive_inertia = self.config.get('adaptive_inertia', True)
        
        # Cognitive and social coefficients
        self.c1 = self.config.get('c1', 1.5)  # Personal best attraction
        self.c2 = self.config.get('c2', 1.5)  # Global best attraction
        
        # Velocity clamping
        self.max_velocity = self.config.get('max_velocity', 0.5)
        
        # Architecture encoding
        self.max_nodes = self.config.get('max_nodes', 20)
        self.encoder = ArchitectureEncoder(search_space, self.max_nodes)
        self.dim = self.encoder.get_dimension()
        
        # Swarm state
        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = -np.inf
        self.global_best_individual: Optional[Individual] = None
        
        # History
        self.iteration = 0
        self.history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized PSO: {self.num_particles} particles, "
            f"dimension={self.dim}, max_iterations={self.max_iterations}"
        )
    
    def initialize_swarm(self) -> None:
        """Initialize swarm with random particles."""
        self.particles = []
        
        for i in range(self.num_particles):
            # Random position in [0, 1]^dim
            position = np.random.rand(self.dim)
            
            # Random velocity (small initial values)
            velocity = (np.random.rand(self.dim) - 0.5) * 0.1
            
            # Create particle
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=-np.inf
            )
            
            self.particles.append(particle)
        
        logger.debug(f"Initialized swarm of {len(self.particles)} particles")
    
    def evaluate_particle(self, particle: Particle, evaluator: Callable) -> None:
        """
        Evaluate fitness of a particle.
        
        Args:
            particle: Particle to evaluate
            evaluator: Fitness evaluation function
        """
        # Decode position to architecture
        graph = self.encoder.decode(particle.position)
        
        # Evaluate
        fitness = evaluator(graph)
        particle.fitness = fitness
        
        # Update personal best
        if fitness > particle.best_fitness:
            particle.best_position = particle.position.copy()
            particle.best_fitness = fitness
        
        # Update global best
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()
            
            # Store as Individual
            individual = Individual(graph)
            individual.fitness = fitness
            self.global_best_individual = individual
    
    def update_velocity(self, particle: Particle, iteration: int) -> np.ndarray:
        """
        Update particle velocity using PSO equations.
        
        Args:
            particle: Particle to update
            iteration: Current iteration number
            
        Returns:
            New velocity vector
        """
        # Adaptive inertia weight (linearly decreasing)
        if self.adaptive_inertia:
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations
        else:
            w = self.w
        
        # Random coefficients for stochasticity
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        
        # Cognitive component (personal best attraction)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        
        # Social component (global best attraction)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        
        # New velocity
        new_velocity = w * particle.velocity + cognitive + social
        
        # Velocity clamping to prevent explosion
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        
        return new_velocity
    
    def update_position(self, particle: Particle) -> np.ndarray:
        """
        Update particle position.
        
        Args:
            particle: Particle to update
            
        Returns:
            New position vector
        """
        new_position = particle.position + particle.velocity
        
        # Clamp to bounds [0, 1]
        new_position = np.clip(new_position, 0.0, 1.0)
        
        return new_position
    
    def optimize(self, evaluator: Callable) -> Individual:
        """
        Run PSO optimization.
        
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
        logger.info(f"Starting PSO optimization for {self.max_iterations} iterations")
        
        # Initialize swarm
        self.initialize_swarm()
        
        # Evaluate initial swarm
        for particle in self.particles:
            self.evaluate_particle(particle, evaluator)
        
        # Main PSO loop
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            # Update each particle
            for particle in self.particles:
                # Update velocity
                particle.velocity = self.update_velocity(particle, iteration)
                
                # Update position
                particle.position = self.update_position(particle)
                
                # Evaluate new position
                self.evaluate_particle(particle, evaluator)
            
            # Record history
            avg_fitness = np.mean([p.fitness for p in self.particles])
            best_fitness = max(p.fitness for p in self.particles)
            
            self.history.append({
                'iteration': iteration,
                'global_best_fitness': self.global_best_fitness,
                'avg_fitness': avg_fitness,
                'best_fitness': best_fitness
            })
            
            # Logging
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                logger.info(
                    f"Iteration {iteration}/{self.max_iterations}: "
                    f"global_best={self.global_best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}"
                )
        
        logger.info(
            f"PSO complete. Best fitness: {self.global_best_fitness:.4f}"
        )
        
        return self.global_best_individual
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.history
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot PSO convergence.
        
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
        
        iterations = [h['iteration'] for h in self.history]
        global_best = [h['global_best_fitness'] for h in self.history]
        avg_fitness = [h['avg_fitness'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, global_best, 'b-', linewidth=2, label='Global Best')
        plt.plot(iterations, avg_fitness, 'r--', linewidth=2, label='Average Fitness')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('PSO Convergence', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
            f"ParticleSwarmOptimizer("
            f"particles={self.num_particles}, "
            f"w={self.w:.2f}, "
            f"c1={self.c1:.2f}, "
            f"c2={self.c2:.2f})"
        )


# Convenience function
def optimize_with_pso(
    search_space: SearchSpace,
    evaluator: Callable,
    num_particles: int = 30,
    max_iterations: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    verbose: bool = True
) -> Individual:
    """
    Quick PSO optimization with sensible defaults.
    
    Args:
        search_space: SearchSpace to optimize over
        evaluator: Fitness evaluation function
        num_particles: Swarm size
        max_iterations: Maximum iterations
        w: Inertia weight
        c1: Cognitive coefficient
        c2: Social coefficient
        verbose: Print progress
        
    Returns:
        Best Individual found
        
    Example:
        >>> from morphml.optimizers.evolutionary import optimize_with_pso
        >>> best = optimize_with_pso(
        ...     search_space=space,
        ...     evaluator=my_evaluator,
        ...     num_particles=30,
        ...     max_iterations=100
        ... )
    """
    optimizer = ParticleSwarmOptimizer(
        search_space=search_space,
        config={
            'num_particles': num_particles,
            'max_iterations': max_iterations,
            'w': w,
            'c1': c1,
            'c2': c2
        }
    )
    
    best = optimizer.optimize(evaluator)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PSO Optimization Complete")
        print(f"{'='*60}")
        print(f"Best Fitness: {best.fitness:.4f}")
        print(f"Iterations: {max_iterations}")
        print(f"Swarm Size: {num_particles}")
        print(f"{'='*60}\n")
    
    return best
