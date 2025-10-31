# Component 6: Execution Engine & CLI

**Duration:** Week 8  
**LOC Target:** ~3,000  
**Dependencies:** Components 1-5

---

## 🎯 Objective

Implement local execution engine, model evaluator, result logging, and CLI interface to tie all components together into a usable system.

---

## 📋 Files to Create

### 1. `execution/local_executor.py` (~800 LOC)

**`LocalExecutor` class:**

```python
class LocalExecutor:
    """
    Executes experiments on local machine.
    
    Manages the complete experiment lifecycle:
    1. Initialize search from search space
    2. Run optimizer (GA) for multiple generations
    3. Evaluate candidates using ModelEvaluator
    4. Track results and checkpoints
    5. Return best model
    """
    
    def __init__(self, config: Config = None):
        self.config = config or global_config
        self.logger = get_logger(__name__)
        self.checkpoint_dir = Path(self.config.get('execution.checkpoint_dir'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        search_space: SearchSpace,
        optimizer: BaseOptimizer,
        evaluator: 'ModelEvaluator',
        max_evaluations: int = 1000,
        checkpoint_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Run complete experiment.
        
        Args:
            search_space: Search space definition
            optimizer: Optimizer instance (e.g., GeneticAlgorithm)
            evaluator: Evaluator for fitness computation
            max_evaluations: Budget in number of evaluations
            checkpoint_interval: Save checkpoint every N evaluations
        
        Returns:
            Dictionary with results and best model
        """
        self.logger.info("Starting experiment")
        self.logger.info(f"Search space size: ~{search_space.size()}")
        self.logger.info(f"Budget: {max_evaluations} evaluations")
        
        # Initialize
        population = optimizer.initialize()
        num_evaluations = 0
        start_time = time.time()
        
        # Progress tracking with Rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Evolving architectures...",
                total=max_evaluations
            )
            
            while num_evaluations < max_evaluations and not optimizer.should_stop():
                # Get candidates
                candidates = optimizer.ask()
                
                # Evaluate
                results = []
                for graph in candidates:
                    if num_evaluations >= max_evaluations:
                        break
                    
                    fitness = evaluator.evaluate(graph)
                    results.append((graph, fitness))
                    num_evaluations += 1
                    
                    progress.update(task, advance=1)
                
                # Update optimizer
                optimizer.tell(results)
                
                # Evolve next generation
                population = optimizer.evolve()
                
                # Checkpoint
                if num_evaluations % checkpoint_interval == 0:
                    self._save_checkpoint(optimizer, num_evaluations)
                
                # Log statistics
                stats = optimizer.get_statistics()
                self.logger.info(
                    f"Gen {stats['generation']}: "
                    f"best={stats['best_fitness']:.4f}, "
                    f"mean={stats['mean_fitness']:.4f}, "
                    f"evals={num_evaluations}"
                )
        
        # Finalize
        elapsed_time = time.time() - start_time
        best_graph = optimizer.get_best()
        final_stats = optimizer.get_statistics()
        
        results = {
            'best_graph': best_graph,
            'best_fitness': final_stats['best_fitness'],
            'num_evaluations': num_evaluations,
            'elapsed_time': elapsed_time,
            'final_generation': final_stats['generation'],
            'statistics': final_stats,
            'history': optimizer.history
        }
        
        self.logger.info(f"Experiment complete in {elapsed_time:.2f}s")
        self.logger.info(f"Best fitness: {final_stats['best_fitness']:.4f}")
        
        return results
    
    def _save_checkpoint(self, optimizer: BaseOptimizer, num_evals: int) -> None:
        """Save optimizer state."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{num_evals}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'optimizer_state': optimizer.__dict__,
                'num_evaluations': num_evals,
                'timestamp': time.time()
            }, f)
        
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint to resume experiment."""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
```

---

### 2. `execution/evaluator.py` (~500 LOC)

**`ModelEvaluator` class:**

```python
class ModelEvaluator:
    """
    Evaluates model architectures.
    
    For Phase 1, uses a simplified evaluation:
    - Graph complexity metrics (depth, width, parameters)
    - Heuristic scoring based on known good practices
    
    Future phases will add actual training and validation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self._cache: Dict[str, float] = {}
    
    def evaluate(self, graph: ModelGraph) -> float:
        """
        Evaluate architecture and return fitness score.
        
        Args:
            graph: Model architecture to evaluate
        
        Returns:
            Fitness score (higher is better)
        """
        # Check cache
        graph_hash = self._hash_graph(graph)
        if graph_hash in self._cache:
            self.logger.debug(f"Cache hit for graph {graph_hash[:8]}")
            return self._cache[graph_hash]
        
        # Validate graph
        try:
            graph.validate()
        except GraphError as e:
            self.logger.warning(f"Invalid graph: {e}")
            return 0.0
        
        # Compute fitness
        fitness = self._compute_fitness(graph)
        
        # Cache result
        self._cache[graph_hash] = fitness
        
        return fitness
    
    def _compute_fitness(self, graph: ModelGraph) -> float:
        """
        Compute fitness from graph properties.
        
        Heuristic scoring:
        - Depth: moderate depth is good (5-15 layers)
        - Width: reasonable parameter count
        - Diversity: variety of operation types
        - Structure: presence of key components (conv, pool, etc.)
        """
        metrics = self._compute_metrics(graph)
        
        # Depth score (5-15 is optimal)
        depth = metrics['depth']
        depth_score = 1.0 - abs(depth - 10) / 10.0
        depth_score = max(0, min(1, depth_score))
        
        # Width score (prefer moderate width)
        width = metrics['width']
        width_score = 1.0 if width < 1e6 else 1.0 / (width / 1e6)
        
        # Diversity score
        diversity_score = len(metrics['operation_types']) / len(OPERATION_TYPES)
        
        # Structure score (bonus for good patterns)
        structure_score = 0.0
        ops = metrics['operation_types']
        
        # Bonus for conv + pool combination
        if 'conv2d' in ops and any(p in ops for p in ['max_pool', 'avg_pool']):
            structure_score += 0.2
        
        # Bonus for normalization
        if any(n in ops for n in ['batch_norm', 'layer_norm']):
            structure_score += 0.1
        
        # Bonus for regularization
        if 'dropout' in ops:
            structure_score += 0.1
        
        # Weighted combination
        fitness = (
            0.3 * depth_score +
            0.2 * width_score +
            0.2 * diversity_score +
            0.3 * structure_score
        )
        
        self.logger.debug(
            f"Fitness: {fitness:.4f} "
            f"(depth={depth_score:.2f}, width={width_score:.2f}, "
            f"div={diversity_score:.2f}, struct={structure_score:.2f})"
        )
        
        return fitness
    
    def _compute_metrics(self, graph: ModelGraph) -> Dict[str, Any]:
        """Compute graph metrics."""
        topo_order = graph.topological_sort()
        
        # Depth: number of layers
        depth = len(topo_order)
        
        # Width: estimated parameter count
        width = 0
        for node in topo_order:
            if node.operation == 'conv2d':
                filters = node.get_param('filters', 32)
                kernel = node.get_param('kernel_size', 3)
                width += filters * kernel * kernel
            elif node.operation == 'dense':
                units = node.get_param('units', 128)
                width += units
        
        # Operation diversity
        operation_types = {node.operation for node in topo_order}
        
        return {
            'depth': depth,
            'width': width,
            'operation_types': operation_types,
            'num_nodes': len(graph.nodes),
            'num_edges': len(graph.edges)
        }
    
    def _hash_graph(self, graph: ModelGraph) -> str:
        """Create hash of graph for caching."""
        import hashlib
        graph_str = json.dumps(graph.to_dict(), sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()
```

---

### 3. `execution/result_logger.py` (~300 LOC)

**`ResultLogger` class:**

```python
class ResultLogger:
    """
    Logs experiment results to files.
    
    Creates:
    - JSON summary with best model and statistics
    - CSV with per-evaluation results
    - Visualizations (fitness curve, best architecture)
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """Save all results."""
        self._save_summary(results)
        self._save_history(results['history'])
        self._save_best_model(results['best_graph'])
        self._plot_fitness_curve(results['history'])
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _save_summary(self, results: Dict[str, Any]) -> None:
        """Save JSON summary."""
        summary_path = self.output_dir / "summary.json"
        
        summary = {
            'best_fitness': results['best_fitness'],
            'num_evaluations': results['num_evaluations'],
            'elapsed_time': results['elapsed_time'],
            'final_generation': results['final_generation'],
            'statistics': results['statistics']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        """Save evaluation history as CSV."""
        csv_path = self.output_dir / "history.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'fitness', 'num_nodes', 'num_edges'])
            writer.writeheader()
            
            for entry in history:
                writer.writerow({
                    'generation': entry['generation'],
                    'fitness': entry['fitness'],
                    'num_nodes': len(entry['genome'].nodes),
                    'num_edges': len(entry['genome'].edges)
                })
    
    def _save_best_model(self, graph: ModelGraph) -> None:
        """Save best model graph."""
        save_graph(graph, self.output_dir / "best_model.json", format='json')
    
    def _plot_fitness_curve(self, history: List[Dict[str, Any]]) -> None:
        """Plot fitness over generations."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")
            return
        
        # Group by generation
        generations = {}
        for entry in history:
            gen = entry['generation']
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(entry['fitness'])
        
        # Compute statistics per generation
        gen_nums = sorted(generations.keys())
        best_fitness = [max(generations[g]) for g in gen_nums]
        avg_fitness = [sum(generations[g]) / len(generations[g]) for g in gen_nums]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(gen_nums, best_fitness, label='Best', marker='o')
        plt.plot(gen_nums, avg_fitness, label='Average', marker='s')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / "fitness_curve.png", dpi=150)
        plt.close()
```

---

### 4. `cli/main.py` (~400 LOC)

**CLI entry point with Click:**

```python
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
@click.version_option(version=__version__)
def cli():
    """
    MorphML - Evolutionary AutoML Construction Kit
    
    Use 'morphml COMMAND --help' for more information on a command.
    """
    pass

@cli.command()
@click.argument('experiment_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./results', help='Output directory')
@click.option('--checkpoint-dir', '-c', default=None, help='Checkpoint directory')
@click.option('--resume', '-r', type=click.Path(exists=True), help='Resume from checkpoint')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def run(experiment_file, output_dir, checkpoint_dir, resume, verbose):
    """
    Run an experiment from a Python file.
    
    Example:
        morphml run experiment.py --output-dir ./results
    """
    # Setup logging
    setup_logging(verbose=verbose)
    logger = get_logger(__name__)
    
    console.print(f"[bold cyan]MorphML v{__version__}[/bold cyan]")
    console.print(f"Running experiment: {experiment_file}\n")
    
    # Load experiment definition
    try:
        spec = importlib.util.spec_from_file_location("experiment", experiment_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        console.print(f"[bold red]Error loading experiment:[/bold red] {e}")
        raise click.Abort()
    
    # Extract components
    if not hasattr(module, 'search_space'):
        console.print("[bold red]Error:[/bold red] experiment file must define 'search_space'")
        raise click.Abort()
    
    search_space = module.search_space
    optimizer_config = getattr(module, 'optimizer_config', {})
    max_evaluations = getattr(module, 'max_evaluations', 1000)
    
    # Initialize components
    optimizer = GeneticAlgorithm(search_space, optimizer_config)
    evaluator = ModelEvaluator()
    executor = LocalExecutor()
    
    # Run experiment
    try:
        results = executor.run(
            search_space=search_space,
            optimizer=optimizer,
            evaluator=evaluator,
            max_evaluations=max_evaluations
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise click.Abort()
    
    # Save results
    result_logger = ResultLogger(output_dir)
    result_logger.log_results(results)
    
    # Display summary
    _display_results_summary(results)

def _display_results_summary(results: Dict[str, Any]):
    """Display results in terminal."""
    table = Table(title="Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Best Fitness", f"{results['best_fitness']:.4f}")
    table.add_row("Evaluations", str(results['num_evaluations']))
    table.add_row("Time (seconds)", f"{results['elapsed_time']:.2f}")
    table.add_row("Generations", str(results['final_generation']))
    
    console.print(table)

@cli.command()
@click.argument('results_dir', type=click.Path(exists=True))
def status(results_dir):
    """Show status of an experiment."""
    summary_path = Path(results_dir) / "summary.json"
    
    if not summary_path.exists():
        console.print(f"[red]No results found in {results_dir}[/red]")
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    table = Table(title=f"Experiment Status: {results_dir}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in summary.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)

@cli.command()
def config():
    """Show current configuration."""
    table = Table(title="MorphML Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    
    config_dict = global_config.to_dict()
    
    def add_rows(d, prefix=""):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                add_rows(value, full_key)
            else:
                table.add_row(full_key, str(value))
    
    add_rows(config_dict)
    console.print(table)

if __name__ == '__main__':
    cli()
```

---

### 5. `cli/commands/` - Command modules

Split commands into separate files for better organization:

**`cli/commands/run.py`** - Run command implementation  
**`cli/commands/status.py`** - Status command  
**`cli/commands/config.py`** - Config command

---

## 🧪 Tests

**`tests/integration/test_end_to_end.py`** (~600 LOC):

```python
def test_full_experiment_workflow():
    """
    Integration test: complete workflow from DSL to results.
    """
    # Define search space
    space = SearchSpace(
        layers=[
            Layer.conv2d(filters=[32, 64], kernel_size=[3]),
            Layer.dense(units=[128, 256])
        ]
    )
    
    # Configure optimizer
    config = {
        'population_size': 10,
        'max_generations': 3  # Short for testing
    }
    optimizer = GeneticAlgorithm(space, config)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Run experiment
    executor = LocalExecutor()
    results = executor.run(
        search_space=space,
        optimizer=optimizer,
        evaluator=evaluator,
        max_evaluations=30
    )
    
    # Validate results
    assert 'best_graph' in results
    assert 'best_fitness' in results
    assert results['num_evaluations'] <= 30
    assert results['best_graph'].is_valid_dag()

def test_cli_run_command(tmp_path):
    """Test CLI run command."""
    # Create experiment file
    experiment_file = tmp_path / "experiment.py"
    experiment_file.write_text("""
from morphml import SearchSpace, Layer

search_space = SearchSpace(
    layers=[Layer.dense([128])]
)

optimizer_config = {
    'population_size': 5,
    'max_generations': 2
}

max_evaluations = 10
""")
    
    # Run CLI
    runner = CliRunner()
    result = runner.invoke(cli, ['run', str(experiment_file), '--output-dir', str(tmp_path)])
    
    assert result.exit_code == 0
    assert (tmp_path / "summary.json").exists()
```

---

## 📝 Example Experiment File

**`examples/quickstart.py`:**

```python
"""
MorphML Quickstart Example

This example demonstrates basic usage:
1. Define search space
2. Configure optimizer
3. Run experiment
"""

from morphml import SearchSpace, Layer

# Define what architectures to search over
search_space = SearchSpace(
    layers=[
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.batch_norm(),
        Layer.dense(units=[128, 256, 512]),
        Layer.dropout(rate=[0.1, 0.3, 0.5])
    ]
)

# Configure genetic algorithm
optimizer_config = {
    'population_size': 20,
    'elite_size': 2,
    'mutation_rate': 0.15,
    'crossover_rate': 0.7,
    'max_generations': 50,
    'selection_strategy': 'tournament',
    'tournament_size': 3
}

# Budget
max_evaluations = 500
```

**Run with:**
```bash
morphml run examples/quickstart.py --output-dir ./results
```

---

## ✅ Deliverables

- [ ] LocalExecutor with progress tracking
- [ ] ModelEvaluator with heuristic scoring
- [ ] ResultLogger with CSV/JSON/plots
- [ ] CLI with run/status/config commands
- [ ] Rich terminal output with progress bars
- [ ] Checkpointing system
- [ ] End-to-end integration test
- [ ] Example experiment files
- [ ] All components work together seamlessly

---

## 🎉 Phase 1 Complete!

After completing this component, you'll have a fully functional foundation for MorphML:
- ✅ DSL for defining search spaces
- ✅ Graph-based model representation
- ✅ Genetic algorithm optimizer
- ✅ Local execution engine
- ✅ Professional CLI
- ✅ Comprehensive tests

**Usage:**
```bash
# Install
poetry install

# Run example
morphml run examples/quickstart.py

# Check results
morphml status ./results
```

---

**Next Phase:** Phase 2 will add advanced optimizers (Bayesian, DARTS, multi-objective).
