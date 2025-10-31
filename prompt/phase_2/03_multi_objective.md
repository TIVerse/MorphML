# Component 3: Multi-Objective Optimization

**Duration:** Week 5-6  
**LOC Target:** ~4,000  
**Dependencies:** Phase 1, Phase 2 Components 1-2

---

## 🎯 Objective

Implement multi-objective optimization to simultaneously optimize multiple competing objectives:
- **Accuracy vs Latency** - High accuracy, low inference time
- **Accuracy vs Model Size** - High accuracy, few parameters
- **Accuracy vs Energy** - High accuracy, low power consumption

Use **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) to find Pareto-optimal architectures.

---

## 📋 Core Concepts

### Pareto Dominance
Architecture A dominates B if:
- A is better in at least one objective
- A is no worse in all other objectives

### Pareto Front
Set of non-dominated solutions - no solution is strictly better than another.

### NSGA-II Algorithm
1. **Fast Non-dominated Sorting** - Rank solutions by domination
2. **Crowding Distance** - Maintain diversity in objective space
3. **Selection** - Prefer lower rank, higher crowding distance
4. **Crossover & Mutation** - Generate offspring
5. **Elitism** - Combine parent + offspring, select best

---

## 📋 Files to Create

### 1. `multi_objective/nsga2.py` (~2,000 LOC)

**`NSGA2Optimizer` class:**

```python
from typing import List, Dict, Tuple, Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class MultiObjectiveIndividual:
    """Individual with multiple fitness values."""
    genome: ModelGraph
    objectives: Dict[str, float]  # {'accuracy': 0.95, 'latency': 12.5, 'params': 2.1e6}
    rank: int = 0
    crowding_distance: float = 0.0
    
    def dominates(self, other: 'MultiObjectiveIndividual') -> bool:
        """Check if this individual dominates another."""
        better_in_any = False
        
        for obj_name, obj_value in self.objectives.items():
            other_value = other.objectives[obj_name]
            
            # Assume maximization (negate for minimization)
            if obj_value < other_value:
                return False  # Worse in this objective
            elif obj_value > other_value:
                better_in_any = True
        
        return better_in_any


class NSGA2Optimizer(BaseOptimizer):
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.
    
    Multi-objective evolutionary algorithm with:
    - Fast non-dominated sorting (O(MN²))
    - Crowding distance for diversity
    - Elitism (parent + offspring selection)
    
    Config:
        population_size: Population size (default: 100)
        num_generations: Number of generations (default: 100)
        crossover_rate: Crossover probability (default: 0.9)
        mutation_rate: Mutation probability (default: 0.1)
        objectives: List of objective functions
            Example: [
                {'name': 'accuracy', 'maximize': True},
                {'name': 'latency', 'maximize': False},
                {'name': 'params', 'maximize': False}
            ]
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        self.pop_size = config.get('population_size', 100)
        self.num_generations = config.get('num_generations', 100)
        self.crossover_rate = config.get('crossover_rate', 0.9)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        
        # Objective specifications
        self.objectives = config.get('objectives', [
            {'name': 'accuracy', 'maximize': True},
            {'name': 'latency', 'maximize': False}
        ])
        
        self.population: List[MultiObjectiveIndividual] = []
        self.pareto_fronts: List[List[MultiObjectiveIndividual]] = []
    
    def initialize(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.pop_size):
            genome = self.search_space.sample()
            individual = MultiObjectiveIndividual(
                genome=genome,
                objectives={}
            )
            self.population.append(individual)
    
    def evaluate(self, individual: MultiObjectiveIndividual) -> None:
        """
        Evaluate all objectives for an individual.
        
        Objectives might include:
        - accuracy: Validation accuracy
        - latency: Inference time (ms)
        - params: Number of parameters
        - flops: Floating point operations
        - energy: Power consumption
        """
        # Evaluate architecture
        results = self._evaluate_architecture(individual.genome)
        
        # Extract objective values
        for obj_spec in self.objectives:
            obj_name = obj_spec['name']
            maximize = obj_spec['maximize']
            
            value = results.get(obj_name, 0.0)
            
            # Negate if minimizing (for consistent comparison)
            if not maximize:
                value = -value
            
            individual.objectives[obj_name] = value
    
    def fast_non_dominated_sort(
        self,
        population: List[MultiObjectiveIndividual]
    ) -> List[List[MultiObjectiveIndividual]]:
        """
        Fast non-dominated sorting.
        
        Returns:
            List of Pareto fronts (F0, F1, F2, ...)
            F0 = non-dominated set
        """
        fronts = [[]]
        
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def calculate_crowding_distance(
        self,
        front: List[MultiObjectiveIndividual]
    ) -> None:
        """
        Calculate crowding distance for individuals in a front.
        
        Crowding distance measures density of solutions around
        an individual in objective space.
        """
        if len(front) == 0:
            return
        
        # Initialize distances
        for individual in front:
            individual.crowding_distance = 0.0
        
        # For each objective
        for obj_name in self.objectives[0]['name']:
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_name])
            
            # Infinite distance for boundary points
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Normalize objective range
            obj_min = front[0].objectives[obj_name]
            obj_max = front[-1].objectives[obj_name]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate distances
            for i in range(1, len(front) - 1):
                distance = (
                    front[i + 1].objectives[obj_name] -
                    front[i - 1].objectives[obj_name]
                ) / obj_range
                
                front[i].crowding_distance += distance
    
    def tournament_selection(
        self,
        k: int = 2
    ) -> MultiObjectiveIndividual:
        """
        Binary tournament selection.
        
        Select k individuals, return one with:
        1. Better rank (lower is better)
        2. If same rank, better crowding distance (higher is better)
        """
        contestants = random.sample(self.population, k)
        
        # Sort by rank, then crowding distance
        contestants.sort(
            key=lambda x: (x.rank, -x.crowding_distance)
        )
        
        return contestants[0]
    
    def evolve(self) -> List[MultiObjectiveIndividual]:
        """
        Generate offspring population via crossover and mutation.
        
        Returns:
            Offspring population
        """
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self._crossover(
                    parent1.genome,
                    parent2.genome
                )
            else:
                child1_genome = parent1.genome.clone()
                child2_genome = parent2.genome.clone()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1_genome = self._mutate(child1_genome)
            if random.random() < self.mutation_rate:
                child2_genome = self._mutate(child2_genome)
            
            # Create offspring
            child1 = MultiObjectiveIndividual(genome=child1_genome, objectives={})
            child2 = MultiObjectiveIndividual(genome=child2_genome, objectives={})
            
            offspring.extend([child1, child2])
        
        return offspring[:self.pop_size]
    
    def environmental_selection(
        self,
        combined_population: List[MultiObjectiveIndividual]
    ) -> List[MultiObjectiveIndividual]:
        """
        Select next generation from combined parent + offspring.
        
        Elitist selection preserving best solutions.
        """
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined_population)
        
        # Calculate crowding distances
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select individuals
        next_population = []
        for front in fronts:
            if len(next_population) + len(front) <= self.pop_size:
                next_population.extend(front)
            else:
                # Sort by crowding distance and fill remaining slots
                remaining = self.pop_size - len(next_population)
                front.sort(key=lambda x: -x.crowding_distance)
                next_population.extend(front[:remaining])
                break
        
        return next_population
    
    def optimize(self) -> List[MultiObjectiveIndividual]:
        """
        Run NSGA-II optimization.
        
        Returns:
            Pareto front (non-dominated solutions)
        """
        logger.info(f"Starting NSGA-II with {self.pop_size} individuals")
        
        # Initialize
        self.initialize()
        
        # Evaluate initial population
        for individual in self.population:
            self.evaluate(individual)
        
        # Evolution
        for generation in range(self.num_generations):
            # Generate offspring
            offspring = self.evolve()
            
            # Evaluate offspring
            for individual in offspring:
                self.evaluate(individual)
            
            # Combine parent + offspring
            combined = self.population + offspring
            
            # Environmental selection
            self.population = self.environmental_selection(combined)
            
            # Logging
            if generation % 10 == 0:
                pareto_front = [ind for ind in self.population if ind.rank == 0]
                logger.info(
                    f"Generation {generation}: "
                    f"Pareto front size = {len(pareto_front)}"
                )
                
                # Log Pareto front statistics
                self._log_pareto_statistics(pareto_front)
        
        # Return final Pareto front
        pareto_front = [ind for ind in self.population if ind.rank == 0]
        self.pareto_fronts = [pareto_front]
        
        logger.info(f"NSGA-II complete. Pareto front size: {len(pareto_front)}")
        
        return pareto_front
    
    def _log_pareto_statistics(self, pareto_front: List[MultiObjectiveIndividual]) -> None:
        """Log statistics of Pareto front."""
        for obj_spec in self.objectives:
            obj_name = obj_spec['name']
            values = [ind.objectives[obj_name] for ind in pareto_front]
            
            logger.info(
                f"  {obj_name}: "
                f"min={min(values):.4f}, "
                f"max={max(values):.4f}, "
                f"mean={np.mean(values):.4f}"
            )
```

---

### 2. `multi_objective/objectives.py` (~1,000 LOC)

**Objective function implementations:**

```python
class ObjectiveEvaluator:
    """Evaluates multiple objectives for an architecture."""
    
    @staticmethod
    def evaluate_accuracy(
        graph: ModelGraph,
        dataset: DataLoader,
        num_epochs: int = 10
    ) -> float:
        """Evaluate validation accuracy."""
        model = build_model(graph)
        trainer = Trainer(model, dataset)
        results = trainer.train(num_epochs)
        return results['val_accuracy']
    
    @staticmethod
    def evaluate_latency(
        graph: ModelGraph,
        input_shape: Tuple[int, ...],
        num_trials: int = 100
    ) -> float:
        """
        Measure inference latency (ms).
        
        Average over multiple trials on GPU.
        """
        model = build_model(graph).cuda().eval()
        dummy_input = torch.randn(1, *input_shape).cuda()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(dummy_input)
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_trials):
            with torch.no_grad():
                model(dummy_input)
        
        torch.cuda.synchronize()
        end = time.time()
        
        latency_ms = (end - start) / num_trials * 1000
        return latency_ms
    
    @staticmethod
    def evaluate_params(graph: ModelGraph) -> float:
        """Count model parameters (millions)."""
        model = build_model(graph)
        num_params = sum(p.numel() for p in model.parameters())
        return num_params / 1e6
    
    @staticmethod
    def evaluate_flops(
        graph: ModelGraph,
        input_shape: Tuple[int, ...]
    ) -> float:
        """
        Estimate FLOPs (GFLOPs).
        
        Use thop or fvcore to count operations.
        """
        from thop import profile
        
        model = build_model(graph)
        dummy_input = torch.randn(1, *input_shape)
        
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops / 1e9  # GFLOPs
```

---

### 3. `multi_objective/visualization.py` (~1,000 LOC)

**Pareto front visualization:**

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ParetoVisualizer:
    """Visualize Pareto fronts."""
    
    @staticmethod
    def plot_2d(
        pareto_front: List[MultiObjectiveIndividual],
        obj1_name: str,
        obj2_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D Pareto front.
        
        Args:
            pareto_front: List of Pareto-optimal individuals
            obj1_name: Name of first objective (x-axis)
            obj2_name: Name of second objective (y-axis)
            save_path: Optional path to save figure
        """
        x = [ind.objectives[obj1_name] for ind in pareto_front]
        y = [ind.objectives[obj2_name] for ind in pareto_front]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=100, alpha=0.6, edgecolors='black')
        plt.xlabel(obj1_name.capitalize(), fontsize=14)
        plt.ylabel(obj2_name.capitalize(), fontsize=14)
        plt.title(f'Pareto Front: {obj1_name} vs {obj2_name}', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_3d(
        pareto_front: List[MultiObjectiveIndividual],
        obj1_name: str,
        obj2_name: str,
        obj3_name: str,
        save_path: Optional[str] = None
    ):
        """Plot 3D Pareto front."""
        x = [ind.objectives[obj1_name] for ind in pareto_front]
        y = [ind.objectives[obj2_name] for ind in pareto_front]
        z = [ind.objectives[obj3_name] for ind in pareto_front]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x, y, z, s=100, c=z, cmap='viridis', alpha=0.6, edgecolors='black')
        
        ax.set_xlabel(obj1_name.capitalize(), fontsize=12)
        ax.set_ylabel(obj2_name.capitalize(), fontsize=12)
        ax.set_zlabel(obj3_name.capitalize(), fontsize=12)
        ax.set_title(f'3D Pareto Front', fontsize=16)
        
        plt.colorbar(scatter, label=obj3_name.capitalize())
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_parallel_coordinates(
        pareto_front: List[MultiObjectiveIndividual],
        save_path: Optional[str] = None
    ):
        """
        Parallel coordinates plot for many objectives.
        
        Each line represents a solution, axes represent objectives.
        """
        import pandas as pd
        from pandas.plotting import parallel_coordinates
        
        # Convert to DataFrame
        data = []
        for ind in pareto_front:
            row = ind.objectives.copy()
            data.append(row)
        
        df = pd.DataFrame(data)
        df['solution'] = range(len(df))
        
        plt.figure(figsize=(14, 6))
        parallel_coordinates(df, 'solution', colormap='viridis', alpha=0.5)
        plt.title('Pareto Front - Parallel Coordinates', fontsize=16)
        plt.legend().remove()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
```

---

## 🧪 Tests

**`test_nsga2.py`:**
```python
def test_dominance():
    """Test Pareto dominance logic."""
    ind1 = MultiObjectiveIndividual(
        genome=None,
        objectives={'acc': 0.95, 'latency': -10.0}
    )
    ind2 = MultiObjectiveIndividual(
        genome=None,
        objectives={'acc': 0.90, 'latency': -15.0}
    )
    
    assert ind1.dominates(ind2)  # Better in both
    assert not ind2.dominates(ind1)


def test_nsga2_optimization():
    """Test NSGA-II on toy problem."""
    space = SearchSpace(...)
    
    nsga2 = NSGA2Optimizer(space, {
        'population_size': 50,
        'num_generations': 20,
        'objectives': [
            {'name': 'f1', 'maximize': True},
            {'name': 'f2', 'maximize': True}
        ]
    })
    
    pareto_front = nsga2.optimize()
    
    assert len(pareto_front) > 0
    # All solutions should be non-dominated
    for ind1 in pareto_front:
        for ind2 in pareto_front:
            if ind1 != ind2:
                assert not ind1.dominates(ind2)
```

---

## ✅ Deliverables

- [ ] NSGA-II optimizer implementation
- [ ] Fast non-dominated sorting
- [ ] Crowding distance calculation
- [ ] Multiple objective evaluators (accuracy, latency, params, FLOPs)
- [ ] 2D/3D Pareto front visualization
- [ ] Parallel coordinates plot
- [ ] Tests validating Pareto optimality

---

**Next:** `04_advanced_evolutionary.md`
