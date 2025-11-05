"""Main CLI entry point for MorphML."""

import importlib.util
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from morphml import __version__
from morphml.evaluation import HeuristicEvaluator
from morphml.logging_config import get_logger, setup_logging
from morphml.optimizers import (
    DARTS,
    ENAS,
    DifferentialEvolution,
    GaussianProcessOptimizer,
    GeneticAlgorithm,
    HillClimbing,
    NSGA2Optimizer,
    RandomSearch,
    SimulatedAnnealing,
    SMACOptimizer,
    TPEOptimizer,
)
from morphml.utils import ArchitectureExporter, Checkpoint

console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    MorphML - Evolutionary AutoML Construction Kit

    Use 'morphml COMMAND --help' for more information on a command.
    """
    pass


@cli.command()
@click.argument("experiment_file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="./results", help="Output directory")
@click.option("--checkpoint-dir", "-c", default=None, help="Checkpoint directory")
@click.option("--resume", "-r", type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.option(
    "--export-format",
    "-e",
    type=click.Choice(["pytorch", "keras", "both"]),
    default="both",
    help="Export format",
)
@click.option(
    "--optimizer",
    type=click.Choice(
        ["ga", "rs", "hc", "sa", "de", "gp", "tpe", "smac", "nsga2", "darts", "enas"]
    ),
    default="ga",
    help="Optimizer to use (ga=Genetic Algorithm, rs=Random Search, hc=Hill Climbing, "
    "sa=Simulated Annealing, de=Differential Evolution, gp=Gaussian Process, "
    "tpe=Tree-structured Parzen Estimator, smac=SMAC, nsga2=NSGA-II, "
    "darts=DARTS, enas=ENAS)",
)
def run(
    experiment_file: str,
    output_dir: str,
    checkpoint_dir: str,
    resume: str,
    verbose: bool,
    export_format: str,
    optimizer: str,
):
    """
    Run an experiment from a Python file.

    Example:
        morphml run experiment.py --output-dir ./results
    """
    # Setup logging
    setup_logging(level="DEBUG" if verbose else "INFO")

    console.print(f"[bold cyan]MorphML v{__version__}[/bold cyan]")
    console.print("[cyan]Author:[/cyan] Eshan Roy <eshanized@proton.me>")
    console.print("[cyan]Organization:[/cyan] TONMOY INFRASTRUCTURE & VISION\n")
    console.print(f"Running experiment: [yellow]{experiment_file}[/yellow]\n")

    # Load experiment definition
    try:
        spec = importlib.util.spec_from_file_location("experiment", experiment_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        console.print(f"[bold red]Error loading experiment:[/bold red] {e}")
        sys.exit(1)

    # Extract components
    if not hasattr(module, "search_space"):
        console.print("[bold red]Error:[/bold red] experiment file must define 'search_space'")
        sys.exit(1)

    search_space = module.search_space
    optimizer_config = getattr(
        module,
        "optimizer_config",
        {
            "population_size": 20,
            "num_generations": 50,
            "elite_size": 2,
            "mutation_rate": 0.15,
            "crossover_rate": 0.7,
        },
    )
    getattr(module, "max_evaluations", None)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    optimizer_instance = _create_optimizer(optimizer, search_space, optimizer_config)
    evaluator = HeuristicEvaluator()

    # Run experiment with progress bar
    try:
        console.print("[bold green]Starting optimization...[/bold green]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Evolving architectures...", total=optimizer_config.get("num_generations", 50)
            )

            # Optimize with callback
            def callback(generation, population):
                progress.update(task, advance=1)
                if hasattr(population, "get_statistics"):
                    stats = population.get_statistics()
                    progress.console.print(
                        f"  Gen {generation}: best={stats['best_fitness']:.4f}, "
                        f"mean={stats['mean_fitness']:.4f}, "
                        f"diversity={population.get_diversity():.3f}"
                    )
                else:
                    progress.console.print(f"  Iteration {generation}")

                # Checkpoint
                if checkpoint_dir and generation % 10 == 0:
                    cp_path = Path(checkpoint_dir) / f"checkpoint_gen_{generation}.json"
                    cp_path.parent.mkdir(parents=True, exist_ok=True)
                    Checkpoint.save(optimizer_instance, str(cp_path))

            best = optimizer_instance.optimize(evaluator, callback=callback)

        console.print("\n[bold green]✓ Optimization complete![/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)

    # Save results
    _save_results(best, optimizer_instance, output_path, export_format)

    # Display summary
    _display_results_summary(best, optimizer_instance, output_path)


def _create_optimizer(optimizer_name: str, search_space, config: dict):
    """Factory function to create optimizer instances.

    Args:
        optimizer_name: Short name of optimizer (ga, rs, hc, etc.)
        search_space: SearchSpace to optimize
        config: Configuration dict from experiment file

    Returns:
        Optimizer instance
    """
    optimizer_map = {
        "ga": GeneticAlgorithm,
        "rs": RandomSearch,
        "hc": HillClimbing,
        "sa": SimulatedAnnealing,
        "de": DifferentialEvolution,
        "gp": GaussianProcessOptimizer,
        "tpe": TPEOptimizer,
        "smac": SMACOptimizer,
        "nsga2": NSGA2Optimizer,
    }

    # GPU-dependent optimizers
    if optimizer_name == "darts":
        if DARTS is None:
            console.print(
                "[red]Error:[/red] DARTS requires PyTorch. Install with: pip install 'morphml[gpu]'"
            )
            sys.exit(1)
        console.print("[yellow]Warning:[/yellow] DARTS requires GPU for optimal performance")
        return DARTS(search_space, config)

    if optimizer_name == "enas":
        if ENAS is None:
            console.print(
                "[red]Error:[/red] ENAS requires PyTorch. Install with: pip install 'morphml[gpu]'"
            )
            sys.exit(1)
        console.print("[yellow]Warning:[/yellow] ENAS requires GPU for optimal performance")
        return ENAS(search_space, config)

    # Standard optimizers
    optimizer_class = optimizer_map.get(optimizer_name)
    if optimizer_class is None:
        console.print(f"[red]Error:[/red] Unknown optimizer: {optimizer_name}")
        sys.exit(1)

    # Bayesian optimizers use different config structure
    if optimizer_name in ["gp", "tpe", "smac"]:
        return optimizer_class(search_space, config=config)

    # NSGA2 uses different config structure
    if optimizer_name == "nsga2":
        return optimizer_class(search_space, config=config)

    # Phase 1 optimizers use **kwargs
    return optimizer_class(search_space, **config)


def _save_results(best, optimizer, output_path: Path, export_format: str):
    """Save experiment results."""
    console.print("[cyan]Saving results...[/cyan]")

    # Save best model graph
    best_model_path = output_path / "best_model.json"
    with open(best_model_path, "w") as f:
        json.dump(best.graph.to_dict(), f, indent=2)
    console.print(f"  ✓ Best model: [yellow]{best_model_path}[/yellow]")

    # Export architecture
    exporter = ArchitectureExporter()

    if export_format in ("pytorch", "both"):
        pytorch_path = output_path / "best_model_pytorch.py"
        pytorch_code = exporter.to_pytorch(best.graph, "BestModel")
        with open(pytorch_path, "w") as f:
            f.write(pytorch_code)
        console.print(f"  ✓ PyTorch export: [yellow]{pytorch_path}[/yellow]")

    if export_format in ("keras", "both"):
        keras_path = output_path / "best_model_keras.py"
        keras_code = exporter.to_keras(best.graph, "best_model")
        with open(keras_path, "w") as f:
            f.write(keras_code)
        console.print(f"  ✓ Keras export: [yellow]{keras_path}[/yellow]")

    # Save history if available
    if hasattr(optimizer, "get_history"):
        history = optimizer.get_history()
        history_path = output_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        console.print(f"  ✓ History: [yellow]{history_path}[/yellow]")

    # Save summary
    summary = {
        "best_fitness": best.fitness if hasattr(best, "fitness") else str(best),
    }

    # Add population stats if available
    if hasattr(optimizer, "population"):
        summary["final_generation"] = optimizer.population.generation
        summary["population_size"] = optimizer.population.size()
        summary["statistics"] = optimizer.population.get_statistics()

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"  ✓ Summary: [yellow]{summary_path}[/yellow]\n")


def _display_results_summary(best, optimizer, output_path: Path):
    """Display results in terminal."""
    table = Table(title="Experiment Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=20)

    # Add fitness
    table.add_row("Best Fitness", f"{best.fitness:.6f}" if hasattr(best, "fitness") else str(best))

    # Add population stats if available
    if hasattr(optimizer, "population"):
        stats = optimizer.population.get_statistics()
        table.add_row("Mean Fitness", f"{stats['mean_fitness']:.6f}")
        table.add_row("Final Generation", str(optimizer.population.generation))
        table.add_row("Population Size", str(optimizer.population.size()))
    table.add_row("Best Model Nodes", str(len(best.graph.nodes)))
    table.add_row("Best Model Depth", str(best.graph.get_depth()))
    table.add_row("Est. Parameters", f"{best.graph.estimate_parameters():,}")
    table.add_row("Output Directory", str(output_path))

    console.print(table)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
def status(results_dir: str):
    """Show status of an experiment."""
    results_path = Path(results_dir)
    summary_path = results_path / "summary.json"

    if not summary_path.exists():
        console.print(f"[red]No results found in {results_dir}[/red]")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    table = Table(
        title=f"Experiment Status: {results_dir}", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=30)

    for key, value in summary.items():
        if isinstance(value, dict):
            table.add_row(key, json.dumps(value, indent=2))
        elif isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
        else:
            table.add_row(key, str(value))

    console.print(table)

    # Show best model info if available
    best_model_path = results_path / "best_model.json"
    if best_model_path.exists():
        console.print(f"\n[cyan]Best model saved at:[/cyan] [yellow]{best_model_path}[/yellow]")


@cli.command()
@click.option("--key", "-k", default=None, help="Show specific config key")
def config(key: str):
    """Show current configuration."""
    from morphml.config import get_config

    cfg = get_config()

    if key:
        value = cfg.get(key)
        console.print(f"[cyan]{key}:[/cyan] [green]{value}[/green]")
    else:
        table = Table(title="MorphML Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan", width=40)
        table.add_column("Value", style="green", width=40)

        # Show some default configurations
        default_config = {
            "version": __version__,
            "logging.level": "INFO",
            "execution.max_workers": 4,
            "search.population_size": 50,
            "search.max_generations": 100,
        }

        for k, v in default_config.items():
            table.add_row(k, str(v))

        console.print(table)


@cli.command()
@click.argument("architecture_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["pytorch", "keras", "both"]),
    default="pytorch",
    help="Export format",
)
@click.option("--output", "-o", default=None, help="Output file path")
def export(architecture_file: str, format: str, output: str):
    """Export architecture to framework-specific code."""
    console.print(f"[cyan]Exporting architecture from:[/cyan] {architecture_file}\n")

    # Load architecture
    with open(architecture_file, "r") as f:
        arch_data = json.load(f)

    from morphml.core.graph import ModelGraph

    graph = ModelGraph.from_dict(arch_data)

    exporter = ArchitectureExporter()

    if format in ("pytorch", "both"):
        code = exporter.to_pytorch(graph, "ExportedModel")
        if output:
            with open(output, "w") as f:
                f.write(code)
            console.print(f"[green]✓ PyTorch code exported to:[/green] [yellow]{output}[/yellow]")
        else:
            console.print("[bold]PyTorch Code:[/bold]\n")
            console.print(code)

    if format in ("keras", "both") and format != "pytorch":
        code = exporter.to_keras(graph, "exported_model")
        output_keras = output.replace(".py", "_keras.py") if output else None
        if output_keras:
            with open(output_keras, "w") as f:
                f.write(code)
            console.print(
                f"[green]✓ Keras code exported to:[/green] [yellow]{output_keras}[/yellow]"
            )
        else:
            console.print("\n[bold]Keras Code:[/bold]\n")
            console.print(code)


@cli.command()
def version():
    """Show MorphML version and system info."""
    import platform
    import sys

    table = Table(title="MorphML System Information", show_header=True, header_style="bold magenta")
    table.add_column("Item", style="cyan", width=30)
    table.add_column("Value", style="green", width=50)

    table.add_row("MorphML Version", __version__)
    table.add_row("Author", "Eshan Roy <eshanized@proton.me>")
    table.add_row("Organization", "TONMOY INFRASTRUCTURE & VISION")
    table.add_row("Repository", "https://github.com/TIVerse/MorphML")
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Platform", platform.platform())

    console.print(table)


if __name__ == "__main__":
    cli()
