# Component 5: Benchmarking & Visualization

**Duration:** Week 9-10  
**LOC Target:** ~5,000  
**Dependencies:** All Phase 2 components

---

## 🎯 Objective

Create comprehensive benchmarking and visualization tools to:
1. **Benchmark optimizers** on standard NAS benchmarks (NAS-Bench-201, DARTS Search Space)
2. **Compare performance** across different algorithms
3. **Visualize search progress** with interactive plots
4. **Generate reports** with metrics and insights
5. **Track experiments** with MLflow integration

---

## 📋 Files to Create

### 1. `benchmarking/nas_bench.py` (~2,000 LOC)

**`NASBenchmark` class:**

```python
class NASBenchmark:
    """
    Interface to NAS-Bench-201 and other benchmarks.
    
    NAS-Bench-201 is a tabular benchmark with pre-computed results
    for 15,625 architectures on CIFAR-10, CIFAR-100, and ImageNet-16.
    
    Benefits:
    - No training required (instant fitness lookup)
    - Reproducible comparisons
    - Fast benchmarking of optimizers
    
    Usage:
        bench = NASBenchmark('nasbench201')
        accuracy = bench.query(architecture, dataset='cifar10')
    """
    
    def __init__(self, benchmark_name: str, data_path: Optional[str] = None):
        """
        Initialize benchmark.
        
        Args:
            benchmark_name: 'nasbench201', 'nasbench101', 'darts'
            data_path: Path to benchmark data
        """
        self.benchmark_name = benchmark_name
        
        if benchmark_name == 'nasbench201':
            from nas_201_api import NASBench201API
            self.api = NASBench201API(data_path or 'NAS-Bench-201-v1_1-096897.pth')
        elif benchmark_name == 'nasbench101':
            from nasbench import api as nb_api
            self.api = nb_api.NASBench(data_path or 'nasbench_only108.tfrecord')
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    def query(
        self,
        architecture: ModelGraph,
        dataset: str = 'cifar10',
        metric: str = 'val_accuracy'
    ) -> float:
        """
        Query benchmark for architecture performance.
        
        Args:
            architecture: Architecture to query
            dataset: 'cifar10', 'cifar100', 'ImageNet16-120'
            metric: 'val_accuracy', 'test_accuracy', 'train_time'
        
        Returns:
            Performance metric value
        """
        if self.benchmark_name == 'nasbench201':
            # Convert ModelGraph to NAS-Bench-201 string format
            arch_str = self._convert_to_nasbench201(architecture)
            
            # Query API
            index = self.api.query_index_by_arch(arch_str)
            info = self.api.get_more_info(index, dataset, hp='200')
            
            if metric == 'val_accuracy':
                return info['valid-accuracy']
            elif metric == 'test_accuracy':
                return info['test-accuracy']
            elif metric == 'train_time':
                return info['train-time']
        
        return 0.0
    
    def _convert_to_nasbench201(self, graph: ModelGraph) -> str:
        """
        Convert ModelGraph to NAS-Bench-201 string format.
        
        NAS-Bench-201 format: |op1~0|+|op2~0|op3~1|+|op4~0|op5~1|op6~2|
        
        Example: |avg_pool_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+...
        """
        # Simplified conversion logic
        # In practice, need to match graph topology to benchmark format
        ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        
        # Extract operations from graph
        arch_str = "|"
        for node in graph.nodes.values():
            op = node.operation if node.operation in ops else 'none'
            arch_str += f"{op}~0|+"
        
        return arch_str.rstrip('+')
    
    def sample_random(self) -> str:
        """Sample random architecture from benchmark."""
        if self.benchmark_name == 'nasbench201':
            import random
            index = random.randint(0, len(self.api) - 1)
            return self.api.arch(index)
        
        return ""


class BenchmarkComparison:
    """
    Compare multiple optimizers on benchmarks.
    
    Runs each optimizer multiple times and aggregates statistics.
    """
    
    def __init__(
        self,
        optimizers: List[BaseOptimizer],
        benchmark: NASBenchmark,
        num_runs: int = 10
    ):
        self.optimizers = optimizers
        self.benchmark = benchmark
        self.num_runs = num_runs
        self.results: Dict[str, List[Dict]] = {}
    
    def run(self, budget: int = 100) -> Dict[str, Any]:
        """
        Run comparison.
        
        Args:
            budget: Number of architecture evaluations per run
        
        Returns:
            Comparison results with statistics
        """
        logger.info(f"Running benchmark comparison with {len(self.optimizers)} optimizers")
        
        for optimizer in self.optimizers:
            optimizer_name = optimizer.__class__.__name__
            logger.info(f"Benchmarking {optimizer_name}")
            
            run_results = []
            
            for run_id in range(self.num_runs):
                logger.info(f"  Run {run_id + 1}/{self.num_runs}")
                
                # Run optimizer
                result = self._run_optimizer(optimizer, budget)
                run_results.append(result)
            
            self.results[optimizer_name] = run_results
        
        # Aggregate statistics
        stats = self._compute_statistics()
        
        return stats
    
    def _run_optimizer(self, optimizer: BaseOptimizer, budget: int) -> Dict:
        """Run single optimizer."""
        # Wrap benchmark query as fitness function
        def fitness_fn(graph: ModelGraph) -> float:
            return self.benchmark.query(graph, dataset='cifar10')
        
        optimizer.evaluate = fitness_fn
        
        # Run optimization
        best = optimizer.optimize()
        
        return {
            'best_fitness': best.fitness,
            'history': optimizer.history
        }
    
    def _compute_statistics(self) -> Dict[str, Dict]:
        """Compute statistics across runs."""
        stats = {}
        
        for optimizer_name, runs in self.results.items():
            best_fitnesses = [run['best_fitness'] for run in runs]
            
            stats[optimizer_name] = {
                'mean': np.mean(best_fitnesses),
                'std': np.std(best_fitnesses),
                'min': np.min(best_fitnesses),
                'max': np.max(best_fitnesses),
                'median': np.median(best_fitnesses)
            }
        
        return stats
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Plot comparison results."""
        visualizer = BenchmarkVisualizer(self.results)
        visualizer.plot_box_plot(save_path)
```

---

### 2. `visualization/experiment_tracker.py` (~1,500 LOC)

**`ExperimentTracker` class:**

```python
import mlflow
from typing import Dict, Any, List

class ExperimentTracker:
    """
    Track experiments with MLflow.
    
    Logs:
    - Hyperparameters
    - Metrics over time
    - Best architecture
    - Artifacts (plots, models)
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize tracker.
        
        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking server URI (default: local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.run = None
    
    def start_run(self, run_name: str, config: Dict[str, Any]):
        """Start MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)
        
        # Log config
        mlflow.log_params(config)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log metric."""
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_architecture(self, architecture: ModelGraph, name: str = "best_architecture"):
        """Log architecture as artifact."""
        # Save as JSON
        import json
        arch_dict = architecture.to_dict()
        
        with open(f"{name}.json", 'w') as f:
            json.dump(arch_dict, f, indent=2)
        
        mlflow.log_artifact(f"{name}.json")
    
    def log_figure(self, fig: plt.Figure, name: str):
        """Log matplotlib figure."""
        fig.savefig(f"{name}.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact(f"{name}.png")
    
    def end_run(self):
        """End MLflow run."""
        mlflow.end_run()


class ProgressVisualizer:
    """
    Real-time visualization of search progress.
    
    Creates interactive plots that update during search.
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def update(self, generation: int, metrics: Dict[str, float]):
        """Update with new metrics."""
        self.history.append({
            'generation': generation,
            **metrics
        })
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot convergence curve."""
        import pandas as pd
        
        df = pd.DataFrame(self.history)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'best_fitness' in df.columns:
            ax.plot(df['generation'], df['best_fitness'], 
                   label='Best', linewidth=2, color='red')
        
        if 'mean_fitness' in df.columns:
            ax.plot(df['generation'], df['mean_fitness'], 
                   label='Mean', linewidth=2, color='blue', alpha=0.7)
        
        ax.set_xlabel('Generation', fontsize=14)
        ax.set_ylabel('Fitness', fontsize=14)
        ax.set_title('Convergence Curve', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_diversity(self, save_path: Optional[str] = None):
        """Plot population diversity over time."""
        df = pd.DataFrame(self.history)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'diversity' in df.columns:
            ax.plot(df['generation'], df['diversity'], linewidth=2)
            ax.set_xlabel('Generation', fontsize=14)
            ax.set_ylabel('Diversity', fontsize=14)
            ax.set_title('Population Diversity', fontsize=16)
            ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
```

---

### 3. `visualization/architecture_viz.py` (~1,000 LOC)

**Interactive architecture visualization:**

```python
import networkx as nx
import plotly.graph_objects as go

class ArchitectureVisualizer:
    """
    Visualize neural architecture graphs.
    
    Supports:
    - Static plots (matplotlib)
    - Interactive plots (Plotly)
    - 3D layouts
    """
    
    @staticmethod
    def plot_static(
        graph: ModelGraph,
        save_path: Optional[str] = None,
        layout: str = 'hierarchical'
    ):
        """
        Plot architecture with matplotlib.
        
        Args:
            graph: Architecture to plot
            save_path: Path to save figure
            layout: 'hierarchical', 'spring', 'circular'
        """
        G = graph.to_networkx()
        
        # Layout
        if layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        elif layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes
        node_colors = [_get_operation_color(graph.nodes[n].operation) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.9, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, 
                              arrowsize=20, ax=ax)
        
        # Draw labels
        labels = {n: graph.nodes[n].operation for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        ax.set_title('Neural Architecture', fontsize=16)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_interactive(graph: ModelGraph, save_path: Optional[str] = None):
        """
        Plot interactive architecture with Plotly.
        
        Allows zooming, panning, and node inspection.
        """
        G = graph.to_networkx()
        pos = nx.spring_layout(G, dim=3)
        
        # Extract node positions
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_z = [pos[n][2] for n in G.nodes()]
        
        # Node trace
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=[_get_operation_color(graph.nodes[n].operation) for n in G.nodes()],
                line=dict(width=2, color='white')
            ),
            text=[graph.nodes[n].operation for n in G.nodes()],
            textposition='top center',
            hoverinfo='text'
        )
        
        # Edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            edge_trace = go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title='Interactive Neural Architecture',
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                zaxis=dict(showgrid=False, showticklabels=False)
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()


def _get_operation_color(operation: str) -> str:
    """Map operation to color."""
    colors = {
        'conv2d': '#FF6B6B',
        'maxpool': '#4ECDC4',
        'avgpool': '#45B7D1',
        'dense': '#FFA07A',
        'relu': '#98D8C8',
        'batchnorm': '#C7CEEA'
    }
    return colors.get(operation, '#CCCCCC')
```

---

### 4. `visualization/reports.py` (~500 LOC)

**Automated report generation:**

```python
from jinja2 import Template

class ReportGenerator:
    """
    Generate HTML/PDF reports for experiments.
    
    Includes:
    - Experiment summary
    - Performance metrics
    - Convergence plots
    - Best architecture visualization
    - Comparison tables
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.data: Dict[str, Any] = {}
    
    def add_section(self, section_name: str, content: Any):
        """Add section to report."""
        self.data[section_name] = content
    
    def generate_html(self, output_path: str):
        """Generate HTML report."""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ experiment_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>{{ experiment_name }}</h1>
            
            {% for section, content in data.items() %}
            <h2>{{ section }}</h2>
            <div>{{ content }}</div>
            {% endfor %}
        </body>
        </html>
        """)
        
        html = template.render(
            experiment_name=self.experiment_name,
            data=self.data
        )
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Report saved to {output_path}")
```

---

## 🧪 Tests

**`test_benchmarking.py`:**
```python
def test_nasbench201_query():
    """Test NAS-Bench-201 query."""
    bench = NASBenchmark('nasbench201', data_path='path/to/data')
    
    # Sample random architecture
    arch_str = bench.sample_random()
    
    # Query should return valid accuracy
    acc = bench.query(arch_str, dataset='cifar10')
    assert 0 <= acc <= 100


def test_benchmark_comparison():
    """Test optimizer comparison on benchmark."""
    bench = NASBenchmark('nasbench201')
    
    optimizers = [
        GeneticAlgorithm(space, {}),
        DifferentialEvolution(space, {})
    ]
    
    comparison = BenchmarkComparison(optimizers, bench, num_runs=3)
    stats = comparison.run(budget=50)
    
    assert 'GeneticAlgorithm' in stats
    assert 'mean' in stats['GeneticAlgorithm']
```

---

## ✅ Deliverables

- [ ] NAS-Bench-201 integration
- [ ] Benchmark comparison framework
- [ ] MLflow experiment tracking
- [ ] Convergence visualization
- [ ] Interactive architecture plots (Plotly)
- [ ] Automated HTML report generation
- [ ] Statistical comparison (box plots, significance tests)

---

## 📈 Example Usage

```python
# Benchmark comparison
from morphml.benchmarking import NASBenchmark, BenchmarkComparison
from morphml.optimizers import GeneticAlgorithm, NSGA2Optimizer, DARTSOptimizer

# Load benchmark
bench = NASBenchmark('nasbench201')

# Define optimizers
optimizers = [
    GeneticAlgorithm(search_space, {'population_size': 50}),
    NSGA2Optimizer(search_space, {'population_size': 100}),
    DARTSOptimizer(search_space, {'learning_rate': 3e-4})
]

# Run comparison
comparison = BenchmarkComparison(optimizers, bench, num_runs=10)
results = comparison.run(budget=500)

# Plot results
comparison.plot_comparison(save_path='comparison.png')

# Generate report
from morphml.visualization import ReportGenerator

report = ReportGenerator('NAS Benchmark Comparison')
report.add_section('Results', results)
report.generate_html('report.html')
```

---

**Phase 2 Complete!** 🎉

Total Phase 2 LOC: ~25,000 production code

**Next Phase:** `phase_3/01_master_worker.md`
