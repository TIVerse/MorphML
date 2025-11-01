"""Progress visualization for optimization."""

from typing import Dict, List, Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ProgressPlotter:
    """
    Plot optimization progress.
    
    Example:
        >>> plotter = ProgressPlotter()
        >>> plotter.plot_fitness_evolution(history)
        >>> plotter.plot_diversity(history)
    """

    def __init__(self):
        """Initialize plotter."""
        pass
    
    def plot_fitness_evolution(
        self,
        history: List[Dict],
        output_path: Optional[str] = None,
        show_bands: bool = True
    ) -> None:
        """
        Plot fitness evolution over generations.
        
        Args:
            history: List of generation statistics
            output_path: Path to save plot
            show_bands: Show std deviation bands
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not history:
            logger.warning("No history to plot")
            return
        
        generations = [h.get('generation', i) for i, h in enumerate(history)]
        best_fitness = [h.get('best_fitness', 0) for h in history]
        mean_fitness = [h.get('mean_fitness', 0) for h in history]
        
        plt.figure(figsize=(12, 6))
        
        # Plot best and mean
        plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best')
        plt.plot(generations, mean_fitness, 'g--', linewidth=2, label='Mean')
        
        # Add worst if available
        if 'worst_fitness' in history[0]:
            worst_fitness = [h['worst_fitness'] for h in history]
            plt.plot(generations, worst_fitness, 'r:', linewidth=1, label='Worst')
        
        # Add std deviation bands
        if show_bands and 'mean_fitness' in history[0]:
            stds = []
            for h in history:
                # Estimate std from mean and best
                mean = h.get('mean_fitness', 0)
                best = h.get('best_fitness', 0)
                std_estimate = abs(best - mean) / 2
                stds.append(std_estimate)
            
            means = np.array(mean_fitness)
            stds = np.array(stds)
            
            plt.fill_between(
                generations,
                means - stds,
                means + stds,
                alpha=0.2,
                color='green',
                label='Mean ± Std'
            )
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('Fitness Evolution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fitness evolution plot saved to {output_path}")
        else:
            plt.show()
    
    def plot_diversity(
        self,
        history: List[Dict],
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot population diversity over time.
        
        Args:
            history: List of generation statistics
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not history:
            return
        
        # Extract diversity if available
        if 'diversity' not in history[0]:
            logger.warning("No diversity data in history")
            return
        
        generations = [h.get('generation', i) for i, h in enumerate(history)]
        diversity = [h['diversity'] for h in history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(generations, diversity, 'purple', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Diversity', fontsize=12)
        plt.title('Population Diversity', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diversity plot saved to {output_path}")
        else:
            plt.show()
    
    def plot_combined(
        self,
        history: List[Dict],
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot fitness and diversity in subplots.
        
        Args:
            history: List of generation statistics
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        generations = [h.get('generation', i) for i, h in enumerate(history)]
        
        # Fitness subplot
        best_fitness = [h.get('best_fitness', 0) for h in history]
        mean_fitness = [h.get('mean_fitness', 0) for h in history]
        
        ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best')
        ax1.plot(generations, mean_fitness, 'g--', linewidth=2, label='Mean')
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Diversity subplot
        if 'diversity' in history[0]:
            diversity = [h['diversity'] for h in history]
            ax2.plot(generations, diversity, 'purple', linewidth=2)
            ax2.set_xlabel('Generation', fontsize=12)
            ax2.set_ylabel('Diversity', fontsize=12)
            ax2.set_title('Population Diversity', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Combined plot saved to {output_path}")
        else:
            plt.show()
    
    def plot_comparison(
        self,
        histories: Dict[str, List[Dict]],
        output_path: Optional[str] = None
    ) -> None:
        """
        Compare multiple optimization runs.
        
        Args:
            histories: Dict mapping run names to history lists
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        plt.figure(figsize=(14, 7))
        
        for name, history in histories.items():
            if not history:
                continue
            
            generations = [h.get('generation', i) for i, h in enumerate(history)]
            best_fitness = [h.get('best_fitness', 0) for h in history]
            
            plt.plot(generations, best_fitness, linewidth=2, label=name, marker='o', markersize=3)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.title('Optimization Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")
        else:
            plt.show()
    
    def plot_pareto_front(
        self,
        individuals: list,
        objectives: List[str],
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            individuals: List of Individual instances
            objectives: List of objective names (must be exactly 2)
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if len(objectives) != 2:
            logger.warning("Pareto front plotting requires exactly 2 objectives")
            return
        
        obj1_vals = [ind.get_metric(objectives[0], 0) for ind in individuals]
        obj2_vals = [ind.get_metric(objectives[1], 0) for ind in individuals]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(obj1_vals, obj2_vals, c='blue', s=100, alpha=0.6, edgecolors='black')
        plt.xlabel(objectives[0], fontsize=12)
        plt.ylabel(objectives[1], fontsize=12)
        plt.title('Pareto Front', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pareto front plot saved to {output_path}")
        else:
            plt.show()
