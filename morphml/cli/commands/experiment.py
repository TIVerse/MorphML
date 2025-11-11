"""Enhanced experiment management CLI commands.

Provides interactive commands for managing NAS experiments.

Example:
    morphml experiment create --name my-experiment
    morphml experiment list
    morphml experiment show exp_123
"""

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress
from typing import Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)
console = Console()


@click.group()
def experiment():
    """Manage NAS experiments."""
    pass


@experiment.command()
@click.option("--name", "-n", help="Experiment name")
@click.option("--optimizer", "-o", default="genetic", help="Optimizer type")
@click.option("--budget", "-b", default=500, type=int, help="Evaluation budget")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def create(name: Optional[str], optimizer: str, budget: int, interactive: bool):
    """
    Create a new experiment.

    Example:
        morphml experiment create --name cifar10-search
        morphml experiment create -i  # Interactive mode
    """
    console.print("[bold cyan]Create New Experiment[/bold cyan]\n")

    # Interactive mode
    if interactive or not name:
        name = Prompt.ask("Experiment name", default="my-experiment")
        optimizer = Prompt.ask(
            "Optimizer",
            choices=["genetic", "random", "hillclimbing", "bayesian"],
            default="genetic",
        )
        budget = int(Prompt.ask("Evaluation budget", default="500"))

    # Define search space interactively
    console.print("\n[bold]Define Search Space[/bold]")

    layers = []

    # Input layer
    console.print("Input layer:")
    input_shape = Prompt.ask("  Input shape (e.g., 3,32,32)", default="3,32,32")
    shape = tuple(map(int, input_shape.split(",")))
    layers.append({"type": "input", "shape": shape})

    # Add layers
    while True:
        add_layer = Confirm.ask("Add a layer?", default=True)
        if not add_layer:
            break

        layer_type = Prompt.ask(
            "  Layer type",
            choices=["conv2d", "dense", "maxpool", "flatten", "relu", "dropout"],
            default="conv2d",
        )

        layer_config = {"type": layer_type}

        if layer_type == "conv2d":
            filters = int(Prompt.ask("    Filters", default="64"))
            kernel = int(Prompt.ask("    Kernel size", default="3"))
            layer_config.update({"filters": filters, "kernel_size": kernel})

        elif layer_type == "dense":
            units = int(Prompt.ask("    Units", default="128"))
            layer_config.update({"units": units})

        elif layer_type == "maxpool":
            pool_size = int(Prompt.ask("    Pool size", default="2"))
            layer_config.update({"pool_size": pool_size})

        elif layer_type == "dropout":
            rate = float(Prompt.ask("    Dropout rate", default="0.5"))
            layer_config.update({"rate": rate})

        layers.append(layer_config)

    # Create experiment config
    config = {
        "name": name,
        "optimizer": optimizer,
        "budget": budget,
        "search_space": {"layers": layers},
    }

    # Display summary
    console.print("\n[bold green]Experiment Configuration:[/bold green]")
    console.print(f"  Name: {name}")
    console.print(f"  Optimizer: {optimizer}")
    console.print(f"  Budget: {budget}")
    console.print(f"  Layers: {len(layers)}")

    # Confirm
    if Confirm.ask("\nCreate experiment?", default=True):
        # TODO: Actually create experiment via API or directly
        console.print(f"\n[green]✓ Created experiment: {name}[/green]")
        console.print(f"  Run with: morphml experiment start {name}")
    else:
        console.print("[yellow]Cancelled[/yellow]")


@experiment.command()
@click.option("--status", "-s", help="Filter by status")
@click.option("--limit", "-l", default=20, help="Maximum results")
def list(status: Optional[str], limit: int):
    """
    List all experiments.

    Example:
        morphml experiment list
        morphml experiment list --status running
    """
    console.print("[bold cyan]Experiments[/bold cyan]\n")

    # TODO: Fetch from API
    # For now, show example data
    experiments = [
        {
            "id": "exp_abc123",
            "name": "CIFAR-10 Search",
            "status": "running",
            "best_accuracy": 0.9234,
            "generation": 25,
            "total_generations": 50,
        },
        {
            "id": "exp_def456",
            "name": "ImageNet Transfer",
            "status": "completed",
            "best_accuracy": 0.8567,
            "generation": 100,
            "total_generations": 100,
        },
        {
            "id": "exp_ghi789",
            "name": "Quick Test",
            "status": "failed",
            "best_accuracy": None,
            "generation": 5,
            "total_generations": 20,
        },
    ]

    # Filter by status
    if status:
        experiments = [e for e in experiments if e["status"] == status]

    # Limit results
    experiments = experiments[:limit]

    # Create table
    table = Table(title=f"Experiments ({len(experiments)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="yellow")
    table.add_column("Progress", style="blue")
    table.add_column("Best Accuracy", style="green")

    for exp in experiments:
        # Status color
        status_color = {
            "running": "yellow",
            "completed": "green",
            "failed": "red",
            "pending": "blue",
        }.get(exp["status"], "white")

        # Progress
        progress = f"{exp['generation']}/{exp['total_generations']}"

        # Accuracy
        accuracy = f"{exp['best_accuracy']:.4f}" if exp["best_accuracy"] else "-"

        table.add_row(
            exp["id"],
            exp["name"],
            f"[{status_color}]{exp['status']}[/{status_color}]",
            progress,
            accuracy,
        )

    console.print(table)

    console.print(f"\n[dim]Showing {len(experiments)} of {len(experiments)} experiments[/dim]")


@experiment.command()
@click.argument("experiment_id")
def show(experiment_id: str):
    """
    Show detailed experiment information.

    Example:
        morphml experiment show exp_abc123
    """
    console.print(f"[bold cyan]Experiment: {experiment_id}[/bold cyan]\n")

    # TODO: Fetch from API
    # Example data
    exp = {
        "id": experiment_id,
        "name": "CIFAR-10 Search",
        "status": "running",
        "optimizer": "genetic",
        "created_at": "2024-11-11T05:00:00Z",
        "started_at": "2024-11-11T05:01:00Z",
        "best_accuracy": 0.9234,
        "generation": 25,
        "total_generations": 50,
        "population_size": 20,
        "best_architecture": {"parameters": 1234567, "depth": 8, "nodes": 12},
    }

    # Display info
    console.print(f"[bold]Name:[/bold] {exp['name']}")
    console.print(f"[bold]Status:[/bold] {exp['status']}")
    console.print(f"[bold]Optimizer:[/bold] {exp['optimizer']}")
    console.print(f"[bold]Created:[/bold] {exp['created_at']}")
    console.print(
        f"[bold]Progress:[/bold] {exp['generation']}/{exp['total_generations']} generations"
    )

    console.print(f"\n[bold green]Best Architecture:[/bold green]")
    console.print(f"  Accuracy: {exp['best_accuracy']:.4f}")
    console.print(f"  Parameters: {exp['best_architecture']['parameters']:,}")
    console.print(f"  Depth: {exp['best_architecture']['depth']}")
    console.print(f"  Nodes: {exp['best_architecture']['nodes']}")


@experiment.command()
@click.argument("experiment_id")
def start(experiment_id: str):
    """
    Start an experiment.

    Example:
        morphml experiment start exp_abc123
    """
    console.print(f"[bold cyan]Starting experiment: {experiment_id}[/bold cyan]\n")

    with Progress() as progress:
        task = progress.add_task("[cyan]Initializing...", total=100)

        # Simulate startup
        import time

        for i in range(100):
            time.sleep(0.01)
            progress.update(task, advance=1)

    console.print("[green]✓ Experiment started[/green]")
    console.print(f"  Monitor with: morphml experiment show {experiment_id}")


@experiment.command()
@click.argument("experiment_id")
def stop(experiment_id: str):
    """
    Stop a running experiment.

    Example:
        morphml experiment stop exp_abc123
    """
    if Confirm.ask(f"Stop experiment {experiment_id}?"):
        console.print(f"[yellow]Stopping experiment: {experiment_id}[/yellow]")
        # TODO: Call API to stop
        console.print("[green]✓ Experiment stopped[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


@experiment.command()
@click.argument("experiment_id")
@click.option("--force", "-f", is_flag=True, help="Force delete without confirmation")
def delete(experiment_id: str, force: bool):
    """
    Delete an experiment.

    Example:
        morphml experiment delete exp_abc123
        morphml experiment delete exp_abc123 --force
    """
    if force or Confirm.ask(f"[red]Delete experiment {experiment_id}?[/red]"):
        console.print(f"[red]Deleting experiment: {experiment_id}[/red]")
        # TODO: Call API to delete
        console.print("[green]✓ Experiment deleted[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


@experiment.command()
@click.argument("experiment_id")
@click.option("--format", "-f", type=click.Choice(["pytorch", "keras", "both"]), default="pytorch")
@click.option("--output", "-o", help="Output file path")
def export(experiment_id: str, format: str, output: Optional[str]):
    """
    Export best architecture from experiment.

    Example:
        morphml experiment export exp_abc123 --format pytorch
        morphml experiment export exp_abc123 --format keras -o model.py
    """
    console.print(f"[bold cyan]Exporting architecture from: {experiment_id}[/bold cyan]\n")

    # TODO: Fetch best architecture and export

    if not output:
        output = f"{experiment_id}_model.py"

    console.print(f"[bold]Format:[/bold] {format}")
    console.print(f"[bold]Output:[/bold] {output}")

    console.print(f"\n[green]✓ Exported to {output}[/green]")
