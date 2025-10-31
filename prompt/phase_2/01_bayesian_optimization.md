# Component 1: Bayesian Optimization

**Duration:** Weeks 1-2  
**LOC Target:** ~5,000  
**Dependencies:** Phase 1, scipy, scikit-optimize

---

## 🎯 Objective

Implement Bayesian optimization for sample-efficient architecture search using surrogate models (Gaussian Process, TPE) and acquisition functions.

---

## 📋 Files to Create

### 1. `bayesian/gaussian_process.py` (~1,500 LOC)

**`GaussianProcessOptimizer` class:**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

class GaussianProcessOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Gaussian Process surrogate.
    
    The GP models fitness as a function of architecture parameters,
    allowing intelligent exploration-exploitation trade-off through
    acquisition functions.
    
    Config:
        acquisition: 'ei' (expected improvement), 'ucb', 'pi'
        kernel: 'matern', 'rbf'
        n_initial_points: Random samples before GP fitting
        xi: Exploration parameter for EI
        kappa: Exploration parameter for UCB
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        self.acquisition_type = config.get('acquisition', 'ei')
        self.n_initial = config.get('n_initial_points', 10)
        self.xi = config.get('xi', 0.01)
        self.kappa = config.get('kappa', 2.576)
        
        # Kernel selection
        kernel_type = config.get('kernel', 'matern')
        if kernel_type == 'matern':
            kernel = Matern(nu=2.5)
        elif kernel_type == 'rbf':
            kernel = RBF()
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
    
    def initialize(self) -> Population:
        """Random initialization."""
        individuals = []
        for _ in range(self.n_initial):
            genome = self.search_space.sample()
            individuals.append(Individual(genome))
        return Population(individuals)
    
    def ask(self) -> List[ModelGraph]:
        """Generate next candidate using acquisition function."""
        if len(self.y_observed) < self.n_initial:
            # Random exploration
            return [self.search_space.sample()]
        
        # Fit GP on observed data
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        # Optimize acquisition function
        best_x = self._optimize_acquisition()
        best_graph = self._decode_architecture(best_x)
        
        return [best_graph]
    
    def tell(self, results: List[Tuple[ModelGraph, float]]) -> None:
        """Update GP with new observations."""
        for graph, fitness in results:
            x = self._encode_architecture(graph)
            self.X_observed.append(x)
            self.y_observed.append(fitness)
            
            self.history.append({
                'generation': self.generation,
                'genome': graph,
                'fitness': fitness
            })
        
        self.generation += 1
    
    def _optimize_acquisition(self) -> np.ndarray:
        """Find architecture that maximizes acquisition function."""
        from scipy.optimize import differential_evolution
        
        # Define bounds for continuous encoding
        bounds = self._get_bounds()
        
        # Negative because we maximize (scipy minimizes)
        result = differential_evolution(
            lambda x: -self._acquisition_function(x),
            bounds=bounds,
            maxiter=100,
            seed=42
        )
        
        return result.x
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Compute acquisition value for architecture encoding.
        
        Expected Improvement (EI):
            EI(x) = E[max(f(x) - f_best, 0)]
                  = (μ - f_best - ξ) * Φ(Z) + σ * φ(Z)
        
        Upper Confidence Bound (UCB):
            UCB(x) = μ(x) + κ * σ(x)
        """
        x = x.reshape(1, -1)
        
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if self.acquisition_type == 'ei':
            # Expected Improvement
            f_best = max(self.y_observed)
            with np.errstate(divide='ignore'):
                Z = (mu - f_best - self.xi) / sigma
                ei = (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
        
        elif self.acquisition_type == 'ucb':
            # Upper Confidence Bound
            return mu + self.kappa * sigma
        
        elif self.acquisition_type == 'pi':
            # Probability of Improvement
            f_best = max(self.y_observed)
            Z = (mu - f_best - self.xi) / sigma
            return norm.cdf(Z)
    
    def _encode_architecture(self, graph: ModelGraph) -> np.ndarray:
        """
        Encode graph as fixed-length vector for GP.
        
        Encoding scheme:
        - One-hot for operations at each position
        - Continuous for hyperparameters
        - Pad/truncate to fixed length
        """
        encoding = []
        
        topo_order = graph.topological_sort()
        max_depth = 20  # Fixed encoding length
        
        for i in range(max_depth):
            if i < len(topo_order):
                node = topo_order[i]
                # One-hot operation type (simplified)
                op_idx = OPERATION_TYPES.index(node.operation) if node.operation in OPERATION_TYPES else 0
                encoding.append(op_idx)
                
                # Key hyperparameters
                if node.operation == 'conv2d':
                    encoding.append(node.get_param('filters', 32))
                elif node.operation == 'dense':
                    encoding.append(node.get_param('units', 128))
                else:
                    encoding.append(0)
            else:
                # Padding
                encoding.extend([0, 0])
        
        return np.array(encoding, dtype=float)
    
    def _decode_architecture(self, x: np.ndarray) -> ModelGraph:
        """Decode vector back to ModelGraph."""
        # Simplified: sample similar architecture
        # In practice, use more sophisticated decoding
        return self.search_space.sample()
    
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for optimization."""
        # Bounds for encoding dimensions
        max_depth = 20
        bounds = []
        for _ in range(max_depth):
            bounds.append((0, len(OPERATION_TYPES)))  # Operation
            bounds.append((0, 512))  # Hyperparameter
        return bounds
```

---

### 2. `bayesian/tpe.py` (~1,200 LOC)

**`TPEOptimizer` class:**

```python
class TPEOptimizer(BaseOptimizer):
    """
    Tree-structured Parzen Estimator.
    
    TPE models p(x|y) instead of p(y|x) by:
    1. Splitting observations into good (y > threshold) and bad (y ≤ threshold)
    2. Modeling p(x|y=good) and p(x|y=bad) with tree-structured densities
    3. Selecting x that maximizes p(x|y=good) / p(x|y=bad)
    
    More scalable than GP for high-dimensional spaces.
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        self.n_initial = config.get('n_initial_points', 20)
        self.gamma = config.get('gamma', 0.25)  # Fraction of good samples
        self.n_ei_candidates = config.get('n_ei_candidates', 24)
        
        self.observations: List[Dict[str, Any]] = []
    
    def ask(self) -> List[ModelGraph]:
        """Sample using TPE."""
        if len(self.observations) < self.n_initial:
            return [self.search_space.sample()]
        
        # Split into good and bad
        sorted_obs = sorted(self.observations, key=lambda x: x['fitness'], reverse=True)
        n_good = max(1, int(self.gamma * len(sorted_obs)))
        
        good_samples = sorted_obs[:n_good]
        bad_samples = sorted_obs[n_good:]
        
        # Sample candidates and select best EI
        best_graph = None
        best_ei = -float('inf')
        
        for _ in range(self.n_ei_candidates):
            # Sample from good distribution
            candidate = self._sample_from_good(good_samples)
            
            # Compute EI
            ei = self._compute_ei(candidate, good_samples, bad_samples)
            
            if ei > best_ei:
                best_ei = ei
                best_graph = candidate
        
        return [best_graph]
    
    def _sample_from_good(self, good_samples: List[Dict]) -> ModelGraph:
        """Sample architecture similar to good samples."""
        # Simplified: pick random good sample and mutate
        template = random.choice(good_samples)['genome']
        mutated = self._mutate_slightly(template)
        return mutated
    
    def _compute_ei(self, candidate, good_samples, bad_samples) -> float:
        """Compute expected improvement as ratio of densities."""
        # Simplified EI approximation
        p_good = self._estimate_density(candidate, good_samples)
        p_bad = self._estimate_density(candidate, bad_samples)
        
        if p_bad == 0:
            return float('inf')
        return p_good / p_bad
```

---

### 3. `bayesian/acquisition.py` (~800 LOC)

**Acquisition function implementations:**

```python
def expected_improvement(mu: float, sigma: float, f_best: float, xi: float = 0.01) -> float:
    """Expected Improvement acquisition."""
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (mu - f_best - xi) / sigma
        ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = 0.0 if sigma == 0.0 else ei
    return ei

def upper_confidence_bound(mu: float, sigma: float, kappa: float = 2.576) -> float:
    """UCB acquisition."""
    return mu + kappa * sigma

def probability_of_improvement(mu: float, sigma: float, f_best: float, xi: float = 0.01) -> float:
    """PI acquisition."""
    Z = (mu - f_best - xi) / sigma
    return norm.cdf(Z)

class AcquisitionOptimizer:
    """Optimizes acquisition functions."""
    
    def optimize(
        self,
        acquisition_fn: Callable,
        bounds: List[Tuple[float, float]],
        n_restarts: int = 10
    ) -> np.ndarray:
        """Multi-start optimization of acquisition."""
        best_x = None
        best_value = -float('inf')
        
        for _ in range(n_restarts):
            x0 = [random.uniform(low, high) for low, high in bounds]
            
            result = minimize(
                lambda x: -acquisition_fn(x),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if -result.fun > best_value:
                best_value = -result.fun
                best_x = result.x
        
        return best_x
```

---

### 4. `bayesian/smac.py` (~1,500 LOC)

**`SMACOptimizer` class:**

```python
class SMACOptimizer(BaseOptimizer):
    """
    Sequential Model-based Algorithm Configuration.
    
    Uses Random Forest instead of GP for scalability.
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        from sklearn.ensemble import RandomForestRegressor
        
        self.rf = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 50),
            max_depth=config.get('max_depth', 10),
            random_state=42
        )
        
        self.X_train = []
        self.y_train = []
    
    # Similar structure to GP but using Random Forest
```

---

## 🧪 Tests

**`test_gaussian_process.py`:**
```python
def test_gp_converges_on_toy_problem():
    """GP should find optimum of simple function."""
    def toy_fitness(graph):
        # Fitness = number of conv2d layers (optimal = 5)
        count = sum(1 for n in graph.nodes.values() if n.operation == 'conv2d')
        return -abs(count - 5)
    
    space = SearchSpace(layers=[Layer.conv2d([32])])
    gp = GaussianProcessOptimizer(space, {'n_initial_points': 5})
    
    # Run optimization
    for _ in range(20):
        candidates = gp.ask()
        results = [(g, toy_fitness(g)) for g in candidates]
        gp.tell(results)
    
    best = gp.get_best()
    assert toy_fitness(best) >= -1  # Close to optimum
```

---

## ✅ Deliverables

- [ ] GP optimizer with acquisition functions
- [ ] TPE optimizer
- [ ] SMAC optimizer
- [ ] Architecture encoding/decoding
- [ ] Tests showing convergence
- [ ] Example usage

---

**Next:** `02_gradient_based_nas.md`
