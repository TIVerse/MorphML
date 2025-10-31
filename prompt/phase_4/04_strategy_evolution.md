# Component 4: Strategy Evolution & Adaptation

**Duration:** Weeks 6-7  
**LOC Target:** ~3,000  
**Dependencies:** Components 1-3

---

## 🎯 Objective

Learn which optimization strategies work best:
1. **Multi-Armed Bandit** - Select among optimizer strategies
2. **Meta-RL** - Learn strategy selection policy
3. **Adaptive Hyperparameters** - Auto-tune optimizer settings
4. **Portfolio Optimization** - Combine multiple strategies

---

## 📋 Files to Create

### 1. `meta_learning/strategy_selector.py` (~1,500 LOC)

```python
from typing import List, Dict, Optional
import numpy as np

class StrategySelector:
    """
    Adaptive strategy selection using multi-armed bandits.
    
    Strategies to select from:
    - Genetic Algorithm
    - Bayesian Optimization
    - DARTS
    - Differential Evolution
    - etc.
    
    Uses UCB (Upper Confidence Bound) to balance exploration/exploitation.
    """
    
    def __init__(self, strategies: List[str], config: Dict[str, Any]):
        self.strategies = strategies
        self.num_strategies = len(strategies)
        
        # Statistics per strategy
        self.counts = np.zeros(self.num_strategies)
        self.rewards = np.zeros(self.num_strategies)
        
        # Hyperparameters
        self.exploration_factor = config.get('exploration_factor', 2.0)
    
    def select_strategy(self) -> str:
        """
        Select strategy using UCB algorithm.
        
        UCB(i) = mean_reward(i) + c * sqrt(log(total_pulls) / pulls(i))
        """
        total_pulls = self.counts.sum()
        
        if total_pulls < self.num_strategies:
            # Pull each arm at least once
            return self.strategies[int(total_pulls)]
        
        # Compute UCB scores
        mean_rewards = self.rewards / (self.counts + 1e-8)
        exploration_bonus = self.exploration_factor * np.sqrt(
            np.log(total_pulls) / (self.counts + 1e-8)
        )
        
        ucb_scores = mean_rewards + exploration_bonus
        
        # Select best
        best_idx = np.argmax(ucb_scores)
        
        return self.strategies[best_idx]
    
    def update(self, strategy: str, reward: float):
        """
        Update statistics after using a strategy.
        
        Args:
            strategy: Strategy used
            reward: Reward obtained (e.g., best fitness improvement)
        """
        idx = self.strategies.index(strategy)
        
        self.counts[idx] += 1
        self.rewards[idx] += reward
        
        logger.debug(
            f"Updated {strategy}: "
            f"pulls={self.counts[idx]}, "
            f"avg_reward={self.rewards[idx]/self.counts[idx]:.4f}"
        )
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all strategies."""
        stats = {}
        
        for i, strategy in enumerate(self.strategies):
            if self.counts[i] > 0:
                stats[strategy] = {
                    'pulls': int(self.counts[i]),
                    'total_reward': float(self.rewards[i]),
                    'avg_reward': float(self.rewards[i] / self.counts[i])
                }
        
        return stats


class AdaptiveOptimizer:
    """
    Adaptive optimizer that switches strategies during search.
    
    Uses StrategySelector to choose best optimizer at each generation.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        optimizers: Dict[str, BaseOptimizer],
        config: Dict[str, Any]
    ):
        self.search_space = search_space
        self.optimizers = optimizers
        
        self.selector = StrategySelector(
            strategies=list(optimizers.keys()),
            config=config
        )
        
        self.current_strategy = None
        self.best_fitness = -float('inf')
    
    def optimize(self, num_generations: int = 100) -> Individual:
        """
        Run adaptive optimization.
        
        Switches between strategies based on performance.
        """
        logger.info("Starting adaptive optimization")
        
        for generation in range(num_generations):
            # Select strategy
            strategy = self.selector.select_strategy()
            optimizer = self.optimizers[strategy]
            
            logger.info(f"Gen {generation}: Using {strategy}")
            
            # Run one step
            population = optimizer.step()
            
            # Evaluate
            for ind in population:
                ind.fitness = self.evaluate(ind.genome)
            
            # Track improvement
            gen_best = max(ind.fitness for ind in population)
            improvement = gen_best - self.best_fitness
            
            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
            
            # Update strategy selector (reward = improvement)
            self.selector.update(strategy, improvement)
            
            # Log
            if generation % 10 == 0:
                stats = self.selector.get_statistics()
                logger.info(f"Strategy stats: {stats}")
        
        # Return best
        best_optimizer = max(
            self.optimizers.values(),
            key=lambda opt: max(ind.fitness for ind in opt.population)
        )
        
        return max(best_optimizer.population, key=lambda ind: ind.fitness)
```

---

### 2. `meta_learning/hyperparameter_tuning.py` (~1,000 LOC)

```python
class MetaHyperparameterTuner:
    """
    Automatically tune optimizer hyperparameters.
    
    Uses Bayesian optimization to tune:
    - Population size
    - Mutation rate
    - Crossover rate
    - Learning rates
    - etc.
    """
    
    def __init__(self, optimizer_class, search_space: SearchSpace):
        self.optimizer_class = optimizer_class
        self.search_space = search_space
        
        # Hyperparameter search space
        self.hyperparam_space = self._define_hyperparam_space()
        
        # BO for tuning
        from skopt import BayesSearchCV
        self.tuner = None
    
    def tune(self, budget: int = 50) -> Dict[str, Any]:
        """
        Tune hyperparameters.
        
        Returns:
            Best hyperparameter configuration
        """
        best_config = None
        best_performance = -float('inf')
        
        for trial in range(budget):
            # Sample hyperparameters
            config = self._sample_hyperparams()
            
            # Create optimizer
            optimizer = self.optimizer_class(self.search_space, config)
            
            # Run short optimization
            best = optimizer.optimize(num_generations=20)
            
            # Evaluate
            if best.fitness > best_performance:
                best_performance = best.fitness
                best_config = config
            
            logger.info(
                f"Trial {trial}: config={config}, "
                f"performance={best.fitness:.4f}"
            )
        
        return best_config
```

---

### 3. `meta_learning/portfolio.py` (~500 LOC)

```python
class PortfolioOptimizer:
    """
    Run multiple optimizers in parallel and combine results.
    
    Allocates computational budget across strategies.
    """
    
    def __init__(self, optimizers: List[BaseOptimizer]):
        self.optimizers = optimizers
    
    def optimize(self, total_budget: int) -> List[Individual]:
        """
        Run portfolio of optimizers.
        
        Args:
            total_budget: Total architecture evaluations
        
        Returns:
            Combined best architectures
        """
        # Allocate budget equally
        budget_per_optimizer = total_budget // len(self.optimizers)
        
        all_results = []
        
        for optimizer in self.optimizers:
            results = optimizer.optimize(budget=budget_per_optimizer)
            all_results.extend(results)
        
        # Return top-k overall
        all_results.sort(key=lambda x: x.fitness, reverse=True)
        
        return all_results[:10]
```

---

## 🧪 Tests

```python
def test_strategy_selection():
    """Test UCB strategy selection."""
    selector = StrategySelector(['GA', 'BO', 'DARTS'], {})
    
    # Simulate rewards
    for _ in range(100):
        strategy = selector.select_strategy()
        reward = {'GA': 0.1, 'BO': 0.5, 'DARTS': 0.3}[strategy]
        selector.update(strategy, reward)
    
    stats = selector.get_statistics()
    
    # BO should have highest average reward
    assert stats['BO']['avg_reward'] > stats['GA']['avg_reward']
```

---

## ✅ Deliverables

- [ ] Multi-armed bandit strategy selector
- [ ] Adaptive optimizer switching
- [ ] Hyperparameter auto-tuning
- [ ] Portfolio optimization
- [ ] Outperforms fixed strategy by 20%+

---

**Next:** `05_transfer_learning.md`
