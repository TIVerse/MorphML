# Component 4: Advanced Evolutionary Algorithms

**Duration:** Week 7-8  
**LOC Target:** ~5,000  
**Dependencies:** Phase 1

---

## 🎯 Objective

Implement advanced evolutionary and swarm-based optimization algorithms beyond standard GA:
1. **Differential Evolution (DE)** - Mutation via vector differences
2. **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) - Adaptive step-size control
3. **Particle Swarm Optimization (PSO)** - Swarm intelligence
4. **Evolutionary Programming (EP)** - Self-adaptive mutations

These algorithms excel in different scenarios and provide alternatives to genetic algorithms.

---

## 📋 Files to Create

### 1. `evolutionary/differential_evolution.py` (~1,500 LOC)

**`DifferentialEvolution` class:**

```python
class DifferentialEvolution(BaseOptimizer):
    """
    Differential Evolution (DE).
    
    DE uses vector differences for mutation:
        mutant = x_r1 + F * (x_r2 - x_r3)
    
    where x_r1, x_r2, x_r3 are random individuals and F is scaling factor.
    
    Variants:
    - DE/rand/1: mutant = x_r1 + F * (x_r2 - x_r3)
    - DE/best/1: mutant = x_best + F * (x_r2 - x_r3)
    - DE/rand/2: mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    
    Config:
        population_size: Population size (default: 50)
        F: Scaling factor for mutation (default: 0.8)
        CR: Crossover probability (default: 0.9)
        strategy: 'rand/1', 'best/1', 'rand/2' (default: 'rand/1')
        max_generations: Maximum generations (default: 100)
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        self.pop_size = config.get('population_size', 50)
        self.F = config.get('F', 0.8)  # Differential weight
        self.CR = config.get('CR', 0.9)  # Crossover probability
        self.strategy = config.get('strategy', 'rand/1')
        self.max_generations = config.get('max_generations', 100)
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
    
    def initialize(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.pop_size):
            genome = self.search_space.sample()
            individual = Individual(genome)
            self.population.append(individual)
    
    def mutate_de_rand_1(self, idx: int) -> ModelGraph:
        """
        DE/rand/1 mutation.
        
        Select 3 random individuals (different from idx):
            mutant = x_r1 + F * (x_r2 - x_r3)
        """
        # Select 3 random indices
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2, r3 = random.sample(candidates, 3)
        
        x_r1 = self.population[r1].genome
        x_r2 = self.population[r2].genome
        x_r3 = self.population[r3].genome
        
        # Compute mutant via graph operations
        mutant = self._differential_mutation(x_r1, x_r2, x_r3)
        
        return mutant
    
    def mutate_de_best_1(self, idx: int) -> ModelGraph:
        """
        DE/best/1 mutation.
        
        mutant = x_best + F * (x_r1 - x_r2)
        """
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = random.sample(candidates, 2)
        
        x_best = self.best_individual.genome
        x_r1 = self.population[r1].genome
        x_r2 = self.population[r2].genome
        
        mutant = self._differential_mutation(x_best, x_r1, x_r2)
        
        return mutant
    
    def _differential_mutation(
        self,
        base: ModelGraph,
        diff1: ModelGraph,
        diff2: ModelGraph
    ) -> ModelGraph:
        """
        Apply differential mutation on graphs.
        
        This is non-trivial for discrete structures like graphs.
        We use a heuristic approach:
        1. Clone base graph
        2. Identify differences between diff1 and diff2
        3. Apply fraction F of those differences to base
        """
        mutant = base.clone()
        
        # Find nodes in diff1 but not in diff2
        diff1_nodes = set(n.operation for n in diff1.nodes.values())
        diff2_nodes = set(n.operation for n in diff2.nodes.values())
        
        diff_ops = diff1_nodes - diff2_nodes
        
        # Apply F fraction of differences
        for node in mutant.nodes.values():
            if random.random() < self.F:
                if diff_ops:
                    # Replace with operation from difference
                    node.operation = random.choice(list(diff_ops))
        
        return mutant
    
    def crossover(
        self,
        target: ModelGraph,
        mutant: ModelGraph
    ) -> ModelGraph:
        """
        Binomial crossover.
        
        For each gene, inherit from mutant with probability CR,
        otherwise from target.
        """
        trial = target.clone()
        
        for node_id in trial.nodes:
            if random.random() < self.CR:
                # Inherit from mutant
                if node_id in mutant.nodes:
                    trial.nodes[node_id] = mutant.nodes[node_id].clone()
        
        return trial
    
    def selection(
        self,
        target: Individual,
        trial: Individual
    ) -> Individual:
        """
        Greedy selection.
        
        Select better of target and trial.
        """
        if trial.fitness > target.fitness:
            return trial
        else:
            return target
    
    def optimize(self) -> Individual:
        """Run DE optimization."""
        logger.info(f"Starting DE with strategy={self.strategy}")
        
        # Initialize
        self.initialize()
        
        # Evaluate initial population
        for individual in self.population:
            individual.fitness = self.evaluate(individual.genome)
        
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        
        # Evolution
        for generation in range(self.max_generations):
            new_population = []
            
            for i in range(self.pop_size):
                target = self.population[i]
                
                # Mutation
                if self.strategy == 'rand/1':
                    mutant_genome = self.mutate_de_rand_1(i)
                elif self.strategy == 'best/1':
                    mutant_genome = self.mutate_de_best_1(i)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                
                # Crossover
                trial_genome = self.crossover(target.genome, mutant_genome)
                
                # Evaluate trial
                trial = Individual(trial_genome)
                trial.fitness = self.evaluate(trial_genome)
                
                # Selection
                selected = self.selection(target, trial)
                new_population.append(selected)
                
                # Update best
                if selected.fitness > self.best_individual.fitness:
                    self.best_individual = selected
            
            self.population = new_population
            
            # Logging
            if generation % 10 == 0:
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                logger.info(
                    f"Gen {generation}: "
                    f"best={self.best_individual.fitness:.4f}, "
                    f"avg={avg_fitness:.4f}"
                )
        
        return self.best_individual
```

---

### 2. `evolutionary/cma_es.py` (~1,800 LOC)

**`CMAES` class:**

```python
class CMAES(BaseOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy.
    
    CMA-ES is a state-of-the-art continuous optimizer that:
    - Adapts covariance matrix to shape search distribution
    - Self-adapts step-size (sigma)
    - Invariant to rotations and scalings
    
    Key Components:
    - Mean vector (m): Center of search distribution
    - Covariance matrix (C): Shape of distribution
    - Step-size (sigma): Scale of distribution
    - Evolution paths: Track successful steps
    
    Config:
        population_size: Offspring per generation (default: 4 + ⌊3*ln(n)⌋)
        sigma: Initial step-size (default: 0.3)
        max_generations: Maximum generations (default: 100)
    
    Note: CMA-ES works on continuous spaces. For NAS, we need to
    encode architectures as continuous vectors.
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        # Dimension of search space (encoded)
        self.dim = config.get('dimension', 50)
        
        # Population size
        self.lambda_ = config.get('population_size', 4 + int(3 * np.log(self.dim)))
        self.mu = self.lambda_ // 2  # Parents
        
        # Weights for recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1 / (self.weights ** 2).sum()
        
        # Step-size control
        self.sigma = config.get('sigma', 0.3)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Covariance matrix adaptation
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        
        # Initialize
        self.mean = np.random.randn(self.dim)
        self.C = np.eye(self.dim)  # Covariance matrix
        self.pc = np.zeros(self.dim)  # Evolution path for C
        self.ps = np.zeros(self.dim)  # Evolution path for sigma
        
        self.eigenvalues = np.ones(self.dim)
        self.eigenvectors = np.eye(self.dim)
        
        self.max_generations = config.get('max_generations', 100)
    
    def sample(self) -> np.ndarray:
        """Sample from N(m, sigma^2 * C)."""
        z = np.random.randn(self.dim)
        y = self.eigenvectors @ (np.sqrt(self.eigenvalues) * z)
        x = self.mean + self.sigma * y
        return x
    
    def optimize(self) -> Individual:
        """Run CMA-ES optimization."""
        logger.info(f"Starting CMA-ES (dim={self.dim})")
        
        best_fitness = -float('inf')
        best_individual = None
        
        for generation in range(self.max_generations):
            # Sample offspring
            offspring = []
            for _ in range(self.lambda_):
                x = self.sample()
                
                # Decode to architecture
                genome = self._decode_architecture(x)
                individual = Individual(genome)
                individual.fitness = self.evaluate(genome)
                individual.x = x  # Store encoding
                
                offspring.append(individual)
            
            # Sort by fitness
            offspring.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Select parents
            parents = offspring[:self.mu]
            
            # Update best
            if parents[0].fitness > best_fitness:
                best_fitness = parents[0].fitness
                best_individual = parents[0]
            
            # Recombination (update mean)
            old_mean = self.mean.copy()
            self.mean = sum(w * ind.x for w, ind in zip(self.weights, parents))
            
            # Update evolution paths
            self.update_evolution_paths(old_mean, parents)
            
            # Adapt covariance matrix
            self.update_covariance_matrix(parents)
            
            # Adapt step-size
            self.update_step_size()
            
            # Eigendecomposition (for sampling)
            if generation % 10 == 0:
                self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
                self.eigenvalues = np.maximum(self.eigenvalues, 1e-10)
            
            # Logging
            if generation % 10 == 0:
                logger.info(
                    f"Gen {generation}: "
                    f"best={best_fitness:.4f}, "
                    f"sigma={self.sigma:.4f}"
                )
        
        return best_individual
    
    def update_evolution_paths(self, old_mean: np.ndarray, parents: List[Individual]):
        """Update evolution paths for step-size and covariance."""
        # ... Implementation details
        pass
    
    def update_covariance_matrix(self, parents: List[Individual]):
        """Update covariance matrix."""
        # ... Implementation details
        pass
    
    def update_step_size(self):
        """Adapt step-size sigma."""
        # ... Implementation details
        pass
```

---

### 3. `evolutionary/particle_swarm.py` (~1,200 LOC)

**`ParticleSwarmOptimizer` class:**

```python
@dataclass
class Particle:
    """PSO particle."""
    position: np.ndarray  # Current position
    velocity: np.ndarray  # Current velocity
    best_position: np.ndarray  # Personal best
    best_fitness: float  # Personal best fitness
    genome: Optional[ModelGraph] = None
    fitness: float = 0.0


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO).
    
    Swarm intelligence algorithm where particles:
    1. Move through search space
    2. Remember their best position (cognitive)
    3. Are attracted to global best (social)
    
    Update equations:
        v_i = w*v_i + c1*r1*(p_i - x_i) + c2*r2*(g - x_i)
        x_i = x_i + v_i
    
    where:
    - v_i: velocity of particle i
    - x_i: position of particle i
    - p_i: personal best of particle i
    - g: global best
    - w: inertia weight
    - c1, c2: cognitive and social coefficients
    - r1, r2: random numbers in [0,1]
    
    Config:
        num_particles: Swarm size (default: 30)
        w: Inertia weight (default: 0.7)
        c1: Cognitive coefficient (default: 1.5)
        c2: Social coefficient (default: 1.5)
        max_iterations: Maximum iterations (default: 100)
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        self.num_particles = config.get('num_particles', 30)
        self.w = config.get('w', 0.7)  # Inertia
        self.c1 = config.get('c1', 1.5)  # Cognitive
        self.c2 = config.get('c2', 1.5)  # Social
        self.max_iterations = config.get('max_iterations', 100)
        
        self.dim = config.get('dimension', 50)
        
        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = -float('inf')
    
    def initialize(self) -> None:
        """Initialize swarm."""
        for _ in range(self.num_particles):
            position = np.random.randn(self.dim)
            velocity = np.random.randn(self.dim) * 0.1
            
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=-float('inf')
            )
            
            self.particles.append(particle)
    
    def update_velocity(self, particle: Particle) -> np.ndarray:
        """Update particle velocity."""
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        
        new_velocity = self.w * particle.velocity + cognitive + social
        
        # Velocity clamping
        max_velocity = 0.5
        new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
        
        return new_velocity
    
    def optimize(self) -> Individual:
        """Run PSO."""
        logger.info(f"Starting PSO with {self.num_particles} particles")
        
        self.initialize()
        
        # Evaluate initial swarm
        for particle in self.particles:
            genome = self._decode_architecture(particle.position)
            particle.genome = genome
            particle.fitness = self.evaluate(genome)
            
            particle.best_position = particle.position.copy()
            particle.best_fitness = particle.fitness
            
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # Iterations
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                # Update velocity and position
                particle.velocity = self.update_velocity(particle)
                particle.position += particle.velocity
                
                # Evaluate
                genome = self._decode_architecture(particle.position)
                particle.genome = genome
                particle.fitness = self.evaluate(genome)
                
                # Update personal best
                if particle.fitness > particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Logging
            if iteration % 10 == 0:
                avg_fitness = np.mean([p.fitness for p in self.particles])
                logger.info(
                    f"Iter {iteration}: "
                    f"global_best={self.global_best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}"
                )
        
        # Return best
        best_genome = self._decode_architecture(self.global_best_position)
        best_individual = Individual(best_genome)
        best_individual.fitness = self.global_best_fitness
        
        return best_individual
```

---

### 4. `evolutionary/utils.py` (~500 LOC)

**Architecture encoding/decoding utilities:**

```python
class ArchitectureEncoder:
    """
    Encode/decode architectures for continuous optimizers.
    
    Maps ModelGraph ↔ continuous vector.
    """
    
    @staticmethod
    def encode(graph: ModelGraph, dim: int = 50) -> np.ndarray:
        """Encode graph as fixed-length vector."""
        # Simplified encoding
        vector = np.zeros(dim)
        
        nodes = list(graph.nodes.values())
        for i, node in enumerate(nodes[:dim//2]):
            vector[i] = hash(node.operation) % 100 / 100.0
        
        return vector
    
    @staticmethod
    def decode(vector: np.ndarray, search_space: SearchSpace) -> ModelGraph:
        """Decode vector to graph."""
        # Simplified decoding: sample and modify
        graph = search_space.sample()
        
        # Modify based on vector values
        # ... Implementation details
        
        return graph
```

---

## 🧪 Tests

**`test_differential_evolution.py`:**
```python
def test_de_on_toy_problem():
    """Test DE on simple optimization."""
    space = SearchSpace(...)
    de = DifferentialEvolution(space, {
        'population_size': 20,
        'F': 0.8,
        'CR': 0.9,
        'max_generations': 50
    })
    
    best = de.optimize()
    
    assert best.fitness > 0.7  # Reasonable solution
```

---

## ✅ Deliverables

- [ ] Differential Evolution optimizer
- [ ] CMA-ES optimizer
- [ ] Particle Swarm optimizer
- [ ] Architecture encoding/decoding for continuous spaces
- [ ] Tests showing convergence

---

**Next:** `05_benchmarking_visualization.md`
