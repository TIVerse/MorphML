"""Architecture comparison utilities.

Compare multiple architectures across various metrics.

Example:
    >>> from morphml.utils.comparison import compare_architectures
    >>> 
    >>> comparison = compare_architectures([arch1, arch2, arch3])
    >>> comparison.print_table()
    >>> comparison.plot()
"""

from typing import List, Dict, Any, Optional
import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureComparison:
    """
    Compare multiple architectures across metrics.
    
    Attributes:
        architectures: List of ModelGraph instances
        names: Optional names for architectures
        
    Example:
        >>> comparison = ArchitectureComparison([arch1, arch2, arch3])
        >>> comparison.add_metric("custom", lambda g: len(g.nodes) * 2)
        >>> comparison.print_table()
    """
    
    def __init__(
        self,
        architectures: List[ModelGraph],
        names: Optional[List[str]] = None,
    ):
        """
        Initialize comparison.
        
        Args:
            architectures: List of architectures to compare
            names: Optional names for each architecture
        """
        self.architectures = architectures
        self.names = names or [f"Arch_{i+1}" for i in range(len(architectures))]
        self.custom_metrics = {}
        
        if len(self.architectures) != len(self.names):
            raise ValueError("Number of names must match number of architectures")
    
    def add_metric(self, name: str, func):
        """
        Add custom metric.
        
        Args:
            name: Metric name
            func: Function that takes ModelGraph and returns numeric value
            
        Example:
            >>> comparison.add_metric("complexity", lambda g: g.depth() * len(g.nodes))
        """
        self.custom_metrics[name] = func
    
    def compute_metrics(self) -> Dict[str, List[Any]]:
        """
        Compute all metrics for all architectures.
        
        Returns:
            Dictionary mapping metric names to lists of values
        """
        metrics = {
            "nodes": [],
            "edges": [],
            "parameters": [],
            "depth": [],
            "width": [],
        }
        
        # Add custom metrics
        for name in self.custom_metrics:
            metrics[name] = []
        
        # Compute for each architecture
        for arch in self.architectures:
            metrics["nodes"].append(len(arch.nodes))
            metrics["edges"].append(len(arch.edges))
            metrics["parameters"].append(arch.estimate_parameters())
            metrics["depth"].append(arch.depth())
            metrics["width"].append(arch.width())
            
            # Custom metrics
            for name, func in self.custom_metrics.items():
                try:
                    value = func(arch)
                    metrics[name].append(value)
                except Exception as e:
                    logger.warning(f"Failed to compute {name}: {e}")
                    metrics[name].append(None)
        
        return metrics
    
    def print_table(self):
        """Print comparison table."""
        metrics = self.compute_metrics()
        
        # Try to use rich for better formatting
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            table = Table(title="Architecture Comparison", show_header=True)
            
            table.add_column("Architecture", style="cyan")
            for metric_name in metrics.keys():
                table.add_column(metric_name.title(), style="green")
            
            for i, name in enumerate(self.names):
                row = [name]
                for metric_values in metrics.values():
                    value = metric_values[i]
                    if value is None:
                        row.append("N/A")
                    elif isinstance(value, float):
                        row.append(f"{value:.2f}")
                    elif isinstance(value, int) and value > 1000:
                        row.append(f"{value:,}")
                    else:
                        row.append(str(value))
                table.add_row(*row)
            
            console.print(table)
            
        except ImportError:
            # Fallback to simple print
            print("\n" + "="*80)
            print("Architecture Comparison")
            print("="*80)
            
            # Header
            header = f"{'Architecture':<20}"
            for metric_name in metrics.keys():
                header += f"{metric_name.title():<15}"
            print(header)
            print("-"*80)
            
            # Rows
            for i, name in enumerate(self.names):
                row = f"{name:<20}"
                for metric_values in metrics.values():
                    value = metric_values[i]
                    if value is None:
                        row += f"{'N/A':<15}"
                    elif isinstance(value, float):
                        row += f"{value:<15.2f}"
                    elif isinstance(value, int) and value > 1000:
                        row += f"{value:<15,}"
                    else:
                        row += f"{str(value):<15}"
                print(row)
            print("="*80 + "\n")
    
    def plot(self, output_file: Optional[str] = None):
        """
        Plot comparison charts.
        
        Args:
            output_file: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib required for plotting. Install with: pip install matplotlib")
            return
        
        metrics = self.compute_metrics()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Architecture Comparison", fontsize=16, fontweight="bold")
        
        plot_metrics = ["nodes", "edges", "parameters", "depth", "width"]
        
        for idx, metric_name in enumerate(plot_metrics):
            if idx >= 6:
                break
            
            ax = axes[idx // 3, idx % 3]
            values = metrics[metric_name]
            
            # Bar plot
            x = np.arange(len(self.names))
            ax.bar(x, values, color='skyblue', edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(self.names, rotation=45, ha='right')
            ax.set_title(metric_name.title())
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(plot_metrics) < 6:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {output_file}")
        else:
            plt.show()
    
    def get_best(self, metric: str = "parameters", minimize: bool = True) -> tuple:
        """
        Get best architecture by metric.
        
        Args:
            metric: Metric name
            minimize: Whether to minimize (True) or maximize (False)
            
        Returns:
            Tuple of (architecture, name, value)
        """
        metrics = self.compute_metrics()
        
        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        values = metrics[metric]
        
        if minimize:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return (
            self.architectures[best_idx],
            self.names[best_idx],
            values[best_idx]
        )
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        metrics = self.compute_metrics()
        summary = {}
        
        for metric_name, values in metrics.items():
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            
            if not valid_values:
                continue
            
            summary[metric_name] = {
                "mean": np.mean(valid_values),
                "std": np.std(valid_values),
                "min": np.min(valid_values),
                "max": np.max(valid_values),
            }
        
        return summary


def compare_architectures(
    architectures: List[ModelGraph],
    names: Optional[List[str]] = None,
    print_table: bool = True,
    plot: bool = False,
    output_file: Optional[str] = None,
) -> ArchitectureComparison:
    """
    Quick comparison of multiple architectures.
    
    Args:
        architectures: List of architectures
        names: Optional names
        print_table: Whether to print comparison table
        plot: Whether to plot comparison
        output_file: Optional file to save plot
        
    Returns:
        ArchitectureComparison instance
        
    Example:
        >>> comparison = compare_architectures([arch1, arch2, arch3])
        >>> best_arch, name, params = comparison.get_best("parameters")
        >>> print(f"Best: {name} with {params:,} parameters")
    """
    comparison = ArchitectureComparison(architectures, names)
    
    if print_table:
        comparison.print_table()
    
    if plot:
        comparison.plot(output_file)
    
    return comparison


def find_similar_architectures(
    target: ModelGraph,
    candidates: List[ModelGraph],
    top_k: int = 5,
    metric: str = "structure",
) -> List[tuple]:
    """
    Find architectures similar to target.
    
    Args:
        target: Target architecture
        candidates: List of candidate architectures
        top_k: Number of similar architectures to return
        metric: Similarity metric ("structure", "parameters", "depth")
        
    Returns:
        List of (architecture, similarity_score) tuples
        
    Example:
        >>> similar = find_similar_architectures(my_arch, all_archs, top_k=3)
        >>> for arch, score in similar:
        ...     print(f"Similarity: {score:.3f}")
    """
    similarities = []
    
    target_nodes = len(target.nodes)
    target_edges = len(target.edges)
    target_params = target.estimate_parameters()
    target_depth = target.depth()
    
    for candidate in candidates:
        if metric == "structure":
            # Structure similarity based on nodes and edges
            node_diff = abs(len(candidate.nodes) - target_nodes) / max(target_nodes, 1)
            edge_diff = abs(len(candidate.edges) - target_edges) / max(target_edges, 1)
            similarity = 1.0 / (1.0 + node_diff + edge_diff)
            
        elif metric == "parameters":
            # Parameter similarity
            param_diff = abs(candidate.estimate_parameters() - target_params) / max(target_params, 1)
            similarity = 1.0 / (1.0 + param_diff)
            
        elif metric == "depth":
            # Depth similarity
            depth_diff = abs(candidate.depth() - target_depth) / max(target_depth, 1)
            similarity = 1.0 / (1.0 + depth_diff)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        similarities.append((candidate, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def diff_architectures(arch1: ModelGraph, arch2: ModelGraph) -> Dict[str, Any]:
    """
    Compute differences between two architectures.
    
    Args:
        arch1: First architecture
        arch2: Second architecture
        
    Returns:
        Dictionary of differences
        
    Example:
        >>> diff = diff_architectures(arch1, arch2)
        >>> print(f"Node difference: {diff['nodes_diff']}")
    """
    return {
        "nodes_diff": len(arch2.nodes) - len(arch1.nodes),
        "edges_diff": len(arch2.edges) - len(arch1.edges),
        "parameters_diff": arch2.estimate_parameters() - arch1.estimate_parameters(),
        "depth_diff": arch2.depth() - arch1.depth(),
        "width_diff": arch2.width() - arch1.width(),
        "nodes_1": len(arch1.nodes),
        "nodes_2": len(arch2.nodes),
        "parameters_1": arch1.estimate_parameters(),
        "parameters_2": arch2.estimate_parameters(),
    }
