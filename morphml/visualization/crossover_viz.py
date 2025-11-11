"""Visualization utilities for crossover operations.

This module provides tools to visualize genetic crossover operations,
showing how parent architectures are combined to create offspring.

Example:
    >>> from morphml.visualization.crossover_viz import visualize_crossover
    >>> from morphml.core.graph.mutations import crossover
    >>>
    >>> offspring1, offspring2 = crossover(parent1, parent2)
    >>> visualize_crossover(parent1, parent2, offspring1, offspring2,
    ...                     output_file="crossover.png")
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


def visualize_crossover(
    parent1: ModelGraph,
    parent2: ModelGraph,
    offspring1: ModelGraph,
    offspring2: ModelGraph,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
) -> None:
    """
    Visualize crossover operation showing parents and offspring.

    Creates a 2x2 grid showing:
    - Top left: Parent 1
    - Top right: Parent 2
    - Bottom left: Offspring 1
    - Bottom right: Offspring 2

    Args:
        parent1: First parent graph
        parent2: Second parent graph
        offspring1: First offspring graph
        offspring2: Second offspring graph
        output_file: Optional path to save figure
        figsize: Figure size (width, height)
        show_labels: Whether to show node labels

    Example:
        >>> visualize_crossover(p1, p2, o1, o2, "crossover_result.png")
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Crossover Operation", fontsize=16, fontweight="bold")

    graphs = [
        (parent1, "Parent 1", "lightblue"),
        (parent2, "Parent 2", "lightgreen"),
        (offspring1, "Offspring 1", "lightyellow"),
        (offspring2, "Offspring 2", "lightcoral"),
    ]

    for idx, (graph, title, color) in enumerate(graphs):
        ax = axes[idx // 2, idx % 2]
        _plot_graph(graph, ax, title, color, show_labels)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved crossover visualization to {output_file}")
    else:
        plt.show()


def visualize_crossover_comparison(
    parent1: ModelGraph,
    parent2: ModelGraph,
    offspring: ModelGraph,
    crossover_point: Optional[int] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Visualize single offspring with parent comparison.

    Shows parents side-by-side with offspring, highlighting the
    crossover point if provided.

    Args:
        parent1: First parent graph
        parent2: Second parent graph
        offspring: Offspring graph
        crossover_point: Optional crossover point to highlight
        output_file: Optional path to save figure
        figsize: Figure size (width, height)

    Example:
        >>> visualize_crossover_comparison(p1, p2, child, crossover_point=3)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Crossover Comparison", fontsize=14, fontweight="bold")

    _plot_graph(parent1, axes[0], "Parent 1", "lightblue", True)
    _plot_graph(parent2, axes[1], "Parent 2", "lightgreen", True)
    _plot_graph(offspring, axes[2], "Offspring", "lightyellow", True)

    # Add statistics
    stats_text = (
        f"Parent 1: {len(parent1.nodes)} nodes\n"
        f"Parent 2: {len(parent2.nodes)} nodes\n"
        f"Offspring: {len(offspring.nodes)} nodes"
    )
    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison visualization to {output_file}")
    else:
        plt.show()


def visualize_crossover_animation(
    parent1: ModelGraph,
    parent2: ModelGraph,
    offspring_sequence: list,
    output_file: str = "crossover_animation.gif",
    fps: int = 2,
) -> None:
    """
    Create animated visualization of crossover process.

    Shows the step-by-step process of creating offspring from parents.
    Requires imageio for GIF creation.

    Args:
        parent1: First parent graph
        parent2: Second parent graph
        offspring_sequence: List of intermediate offspring graphs
        output_file: Path to save GIF
        fps: Frames per second

    Example:
        >>> # Create sequence of intermediate offspring
        >>> sequence = [step1, step2, step3, final]
        >>> visualize_crossover_animation(p1, p2, sequence)
    """
    try:
        import imageio
    except ImportError:
        logger.error("imageio required for animation. Install with: pip install imageio")
        return

    import os
    import tempfile

    frames = []
    temp_dir = tempfile.mkdtemp()

    try:
        # Create frame for each step
        for i, offspring in enumerate(offspring_sequence):
            temp_file = os.path.join(temp_dir, f"frame_{i:03d}.png")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"Crossover Step {i+1}/{len(offspring_sequence)}", fontsize=14, fontweight="bold"
            )

            _plot_graph(parent1, axes[0], "Parent 1", "lightblue", True)
            _plot_graph(parent2, axes[1], "Parent 2", "lightgreen", True)
            _plot_graph(offspring, axes[2], f"Offspring (Step {i+1})", "lightyellow", True)

            plt.tight_layout()
            plt.savefig(temp_file, dpi=150, bbox_inches="tight")
            plt.close()

            frames.append(imageio.imread(temp_file))

        # Save as GIF
        imageio.mimsave(output_file, frames, fps=fps)
        logger.info(f"Saved crossover animation to {output_file}")

    finally:
        # Cleanup temp files
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def _plot_graph(
    graph: ModelGraph,
    ax: plt.Axes,
    title: str,
    node_color: str,
    show_labels: bool,
) -> None:
    """
    Plot a single graph on given axes.

    Args:
        graph: Graph to plot
        ax: Matplotlib axes
        title: Plot title
        node_color: Color for nodes
        show_labels: Whether to show node labels
    """
    # Convert to NetworkX for visualization
    try:
        nx_graph = graph.to_networkx()
    except Exception as e:
        logger.warning(f"Failed to convert graph: {e}")
        ax.text(
            0.5, 0.5, "Graph visualization failed", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(title)
        ax.axis("off")
        return

    # Use hierarchical layout for better visualization
    try:
        pos = nx.spring_layout(nx_graph, k=1, iterations=50)
    except:
        pos = nx.random_layout(nx_graph)

    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        ax=ax,
        node_color=node_color,
        node_size=500,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
    )

    # Draw edges
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=2,
    )

    # Draw labels
    if show_labels:
        labels = {}
        for node_id in nx_graph.nodes():
            node = graph.nodes.get(node_id)
            if node:
                # Shorten operation name for display
                op = node.operation
                if len(op) > 8:
                    op = op[:8] + "..."
                labels[node_id] = op
            else:
                labels[node_id] = node_id[:8]

        nx.draw_networkx_labels(
            nx_graph,
            pos,
            labels,
            ax=ax,
            font_size=8,
            font_weight="bold",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    # Add statistics
    stats = f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}"
    ax.text(0.5, -0.05, stats, ha="center", transform=ax.transAxes, fontsize=9)


def compare_crossover_diversity(
    parents: list,
    offspring_list: list,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Compare diversity of offspring from multiple crossover operations.

    Creates visualization showing distribution of offspring characteristics
    compared to parents.

    Args:
        parents: List of parent graphs
        offspring_list: List of offspring graphs
        output_file: Optional path to save figure
        figsize: Figure size (width, height)

    Example:
        >>> parents = [p1, p2, p3, p4]
        >>> offspring = [o1, o2, o3, o4, o5, o6]
        >>> compare_crossover_diversity(parents, offspring)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Crossover Diversity Analysis", fontsize=14, fontweight="bold")

    # Extract metrics
    parent_nodes = [len(g.nodes) for g in parents]
    parent_edges = [len(g.edges) for g in parents]
    parent_depth = [g.depth() for g in parents]

    offspring_nodes = [len(g.nodes) for g in offspring_list]
    offspring_edges = [len(g.edges) for g in offspring_list]
    offspring_depth = [g.depth() for g in offspring_list]

    # Plot node count distribution
    axes[0].hist(parent_nodes, alpha=0.5, label="Parents", bins=10, color="blue")
    axes[0].hist(offspring_nodes, alpha=0.5, label="Offspring", bins=10, color="orange")
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Node Count Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot edge count distribution
    axes[1].hist(parent_edges, alpha=0.5, label="Parents", bins=10, color="blue")
    axes[1].hist(offspring_edges, alpha=0.5, label="Offspring", bins=10, color="orange")
    axes[1].set_xlabel("Number of Edges")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Edge Count Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot depth distribution
    axes[2].hist(parent_depth, alpha=0.5, label="Parents", bins=10, color="blue")
    axes[2].hist(offspring_depth, alpha=0.5, label="Offspring", bins=10, color="orange")
    axes[2].set_xlabel("Depth")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Depth Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved diversity analysis to {output_file}")
    else:
        plt.show()


# Convenience function
def quick_crossover_viz(
    parent1: ModelGraph, parent2: ModelGraph, output_file: str = "crossover.png"
) -> None:
    """
    Quick visualization of crossover operation.

    Performs crossover and visualizes the result in one call.

    Args:
        parent1: First parent graph
        parent2: Second parent graph
        output_file: Path to save visualization

    Example:
        >>> quick_crossover_viz(parent1, parent2, "my_crossover.png")
    """
    from morphml.core.graph.mutations import crossover

    offspring1, offspring2 = crossover(parent1, parent2)
    visualize_crossover(parent1, parent2, offspring1, offspring2, output_file)

    logger.info(f"Crossover visualization saved to {output_file}")
