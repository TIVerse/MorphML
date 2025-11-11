"""Progress tracking utilities for optimization.

Provides rich progress bars and status displays for long-running NAS operations.

Example:
    >>> from morphml.utils.progress import OptimizationProgress
    >>>
    >>> progress = OptimizationProgress(total_generations=100)
    >>> for gen in range(100):
    ...     progress.update(gen, best_fitness=0.95, diversity=0.7)
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class OptimizationProgress:
    """
    Track and display optimization progress with rich formatting.

    Attributes:
        total_generations: Total number of generations
        show_stats: Whether to show detailed statistics

    Example:
        >>> progress = OptimizationProgress(total_generations=50)
        >>> progress.start()
        >>> for gen in range(50):
        ...     progress.update(gen, best_fitness=0.9, diversity=0.5)
        >>> progress.finish()
    """

    def __init__(
        self,
        total_generations: int,
        show_stats: bool = True,
        refresh_rate: float = 0.5,
    ):
        """
        Initialize progress tracker.

        Args:
            total_generations: Total number of generations
            show_stats: Whether to show detailed statistics
            refresh_rate: Update frequency in seconds
        """
        self.total_generations = total_generations
        self.show_stats = show_stats
        self.refresh_rate = refresh_rate

        self.start_time = None
        self.current_gen = 0
        self.best_fitness = 0.0
        self.history = []

        if RICH_AVAILABLE:
            self.console = Console()
            self.progress = None
            self.task_id = None
        else:
            logger.warning("Rich not available. Install with: pip install rich")
            self.console = None

    def start(self):
        """Start progress tracking."""
        self.start_time = datetime.now()

        if RICH_AVAILABLE and self.console:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            self.progress.start()
            self.task_id = self.progress.add_task(
                "[cyan]Optimizing...", total=self.total_generations
            )
        else:
            print(f"Starting optimization: {self.total_generations} generations")

    def update(
        self,
        generation: int,
        best_fitness: Optional[float] = None,
        diversity: Optional[float] = None,
        **kwargs,
    ):
        """
        Update progress.

        Args:
            generation: Current generation number
            best_fitness: Best fitness so far
            diversity: Population diversity
            **kwargs: Additional statistics
        """
        self.current_gen = generation

        if best_fitness is not None:
            self.best_fitness = best_fitness

        # Track history
        self.history.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "diversity": diversity,
                **kwargs,
            }
        )

        if RICH_AVAILABLE and self.progress:
            # Update progress bar
            self.progress.update(
                self.task_id,
                completed=generation + 1,
                description=f"[cyan]Gen {generation+1}/{self.total_generations} | Best: {self.best_fitness:.4f}",
            )
        else:
            # Simple text output
            if generation % 10 == 0 or generation == self.total_generations - 1:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                print(
                    f"Gen {generation+1}/{self.total_generations} | "
                    f"Best: {self.best_fitness:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

    def finish(self):
        """Finish progress tracking and show summary."""
        if RICH_AVAILABLE and self.progress:
            self.progress.stop()
            self._show_summary()
        else:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            print("\nOptimization complete!")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Best fitness: {self.best_fitness:.4f}")

    def _show_summary(self):
        """Show optimization summary."""
        if not RICH_AVAILABLE:
            return

        elapsed = datetime.now() - self.start_time

        # Create summary table
        table = Table(title="Optimization Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Generations", str(self.total_generations))
        table.add_row("Best Fitness", f"{self.best_fitness:.6f}")
        table.add_row("Total Time", str(elapsed).split(".")[0])
        table.add_row(
            "Time per Generation", f"{elapsed.total_seconds() / self.total_generations:.2f}s"
        )

        if self.history:
            improvements = sum(
                1
                for i in range(1, len(self.history))
                if self.history[i].get("best_fitness", 0)
                > self.history[i - 1].get("best_fitness", 0)
            )
            table.add_row("Improvements", str(improvements))

        self.console.print(table)


class SimpleProgressBar:
    """
    Simple progress bar without rich dependency.

    Example:
        >>> bar = SimpleProgressBar(total=100, desc="Processing")
        >>> for i in range(100):
        ...     bar.update(i)
        >>> bar.finish()
    """

    def __init__(self, total: int, desc: str = "Progress", width: int = 50):
        """
        Initialize simple progress bar.

        Args:
            total: Total number of steps
            desc: Description
            width: Bar width in characters
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = datetime.now()

    def update(self, current: int):
        """Update progress bar."""
        self.current = current

        # Calculate progress
        progress = current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)

        # Calculate time
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if current > 0:
            eta = elapsed * (self.total - current) / current
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "??:??:??"

        # Print bar
        print(
            f"\r{self.desc}: |{bar}| {current}/{self.total} [{progress*100:.1f}%] ETA: {eta_str}",
            end="",
            flush=True,
        )

    def finish(self):
        """Finish progress bar."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{self.desc} complete! Time: {elapsed:.2f}s")


def create_progress_tracker(total_generations: int, use_rich: bool = True, **kwargs) -> Any:
    """
    Create appropriate progress tracker.

    Args:
        total_generations: Total number of generations
        use_rich: Whether to use rich progress (if available)
        **kwargs: Additional arguments

    Returns:
        Progress tracker instance

    Example:
        >>> tracker = create_progress_tracker(100)
        >>> tracker.start()
        >>> for i in range(100):
        ...     tracker.update(i, best_fitness=0.9)
        >>> tracker.finish()
    """
    if use_rich and RICH_AVAILABLE:
        return OptimizationProgress(total_generations, **kwargs)
    else:
        return SimpleProgressBar(total_generations, desc="Optimization", **kwargs)


class LiveDashboard:
    """
    Live dashboard for real-time optimization monitoring.

    Shows multiple metrics in a live-updating display.

    Example:
        >>> dashboard = LiveDashboard()
        >>> dashboard.start()
        >>> for gen in range(100):
        ...     dashboard.update({
        ...         'generation': gen,
        ...         'best_fitness': 0.95,
        ...         'diversity': 0.7,
        ...         'crossover_rate': 0.8
        ...     })
        >>> dashboard.stop()
    """

    def __init__(self):
        """Initialize live dashboard."""
        if not RICH_AVAILABLE:
            logger.warning("Live dashboard requires rich. Install with: pip install rich")
            self.enabled = False
            return

        self.enabled = True
        self.console = Console()
        self.live = None
        self.layout = Layout()
        self.stats = {}

    def start(self):
        """Start live dashboard."""
        if not self.enabled:
            return

        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
        self.live.start()

    def update(self, stats: Dict[str, Any]):
        """
        Update dashboard with new statistics.

        Args:
            stats: Dictionary of statistics to display
        """
        if not self.enabled or not self.live:
            return

        self.stats.update(stats)

        # Update header
        self.layout["header"].update(
            Panel("[bold cyan]MorphML Optimization Dashboard[/bold cyan]", style="cyan")
        )

        # Update body with stats table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in self.stats.items():
            if isinstance(value, float):
                table.add_row(key.replace("_", " ").title(), f"{value:.4f}")
            else:
                table.add_row(key.replace("_", " ").title(), str(value))

        self.layout["body"].update(table)

        # Update footer
        self.layout["footer"].update(Panel("[dim]Press Ctrl+C to stop[/dim]", style="dim"))

    def stop(self):
        """Stop live dashboard."""
        if self.enabled and self.live:
            self.live.stop()


# Convenience function
def with_progress(func):
    """
    Decorator to add progress tracking to optimization functions.

    Example:
        >>> @with_progress
        ... def my_optimization(num_generations=100):
        ...     for gen in range(num_generations):
        ...         # optimization logic
        ...         yield gen, {"best_fitness": 0.9}
    """

    def wrapper(*args, **kwargs):
        total = kwargs.get("num_generations", 100)
        progress = create_progress_tracker(total)
        progress.start()

        try:
            result = func(*args, **kwargs)
            if hasattr(result, "__iter__"):
                for gen, stats in result:
                    progress.update(gen, **stats)
                return result
            else:
                return result
        finally:
            progress.finish()

    return wrapper
