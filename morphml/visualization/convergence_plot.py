"""Convergence visualization utilities.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Dict, List, Optional

import numpy as np

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def plot_convergence(
    history: List[Dict],
    metric: str = 'best_fitness',
    save_path: Optional[str] = None,
    title: str = "Convergence Plot"
) -> None:
    """
    Plot optimization convergence over generations/iterations.
    
    Args:
        history: List of history dictionaries with metric values
        metric: Metric to plot (default: 'best_fitness')
        save_path: Path to save plot (displays if None)
        title: Plot title
        
    Example:
        >>> from morphml.visualization.convergence_plot import plot_convergence
        >>> plot_convergence(optimizer.get_history())
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    # Extract metric values
    generations = []
    values = []
    
    for i, entry in enumerate(history):
        if metric in entry:
            generations.append(i)
            values.append(entry[metric])
    
    if not values:
        logger.error(f"Metric '{metric}' not found in history")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, values, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Generation/Iteration', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_comparison(
    histories: Dict[str, List[Dict]],
    metric: str = 'best_fitness',
    save_path: Optional[str] = None,
    title: str = "Convergence Comparison"
) -> None:
    """
    Compare convergence of multiple optimizers.
    
    Args:
        histories: Dict mapping optimizer names to history lists
        metric: Metric to plot
        save_path: Path to save plot (displays if None)
        title: Plot title
        
    Example:
        >>> from morphml.visualization.convergence_plot import plot_convergence_comparison
        >>> histories = {
        ...     'GA': ga_optimizer.get_history(),
        ...     'TPE': tpe_optimizer.get_history()
        ... }
        >>> plot_convergence_comparison(histories)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting")
        return
    
    plt.figure(figsize=(12, 6))
    
    for optimizer_name, history in histories.items():
        generations = []
        values = []
        
        for i, entry in enumerate(history):
            if metric in entry:
                generations.append(i)
                values.append(entry[metric])
        
        if values:
            plt.plot(generations, values, linewidth=2, marker='o', 
                    markersize=4, label=optimizer_name, alpha=0.8)
    
    plt.xlabel('Generation/Iteration', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fitness_distribution(
    population_history: List[List[float]],
    save_path: Optional[str] = None,
    title: str = "Fitness Distribution Over Time"
) -> None:
    """
    Plot fitness distribution evolution using box plots.
    
    Args:
        population_history: List of fitness lists per generation
        save_path: Path to save plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting")
        return
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(population_history, positions=range(len(population_history)))
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Fitness distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Re-export from benchmarks module for convenience
try:
    from morphml.benchmarks.comparator import ConvergenceAnalyzer
except ImportError:
    pass
