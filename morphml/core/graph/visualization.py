"""Visualization utilities for model graphs.

Provides functions to visualize ModelGraph structures using matplotlib
and networkx for graph plotting.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from morphml.core.graph.graph import ModelGraph
from morphml.exceptions import GraphError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


def plot_graph(
    graph: ModelGraph,
    output_path: Optional[Union[str, Path]] = None,
    layout: str = "hierarchical",
    figsize: Tuple[int, int] = (12, 8),
    node_size: int = 3000,
    font_size: int = 10,
    with_labels: bool = True,
    show_params: bool = False,
    dpi: int = 150,
) -> None:
    """
    Visualize model graph structure.

    Args:
        graph: ModelGraph to visualize
        output_path: Path to save figure (if None, displays interactively)
        layout: Layout algorithm ('hierarchical', 'spring', 'circular', 'kamada_kawai')
        figsize: Figure size (width, height)
        node_size: Size of nodes
        font_size: Font size for labels
        with_labels: Whether to show node labels
        show_params: Whether to show parameter counts
        dpi: Resolution for saved figure

    Example:
        >>> plot_graph(graph, 'model.png')
        >>> plot_graph(graph, layout='spring', show_params=True)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise GraphError(
            "Visualization requires matplotlib and networkx. "
            "Install with: pip install matplotlib networkx"
        )

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node_id, node in graph.nodes.items():
        label = node.operation
        if show_params:
            params = graph.estimate_parameters()
            label = f"{node.operation}\n({params:,})"
        G.add_node(node_id, label=label, operation=node.operation)

    # Add edges
    for edge_id, edge in graph.edges.items():
        if edge.source and edge.target:
            G.add_edge(edge.source.id, edge.target.id)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Choose layout
    if layout == "hierarchical":
        pos = _hierarchical_layout(graph, G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        logger.warning(f"Unknown layout '{layout}', using spring")
        pos = nx.spring_layout(G)

    # Color nodes by operation type
    node_colors = _get_node_colors(graph)

    # Draw graph
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_size, alpha=0.9, ax=ax
    )

    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.6, ax=ax
    )

    if with_labels:
        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

    # Add title
    ax.set_title(
        f"Model Graph: {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} edges, "
        f"depth {graph.get_depth()}",
        fontsize=14,
        fontweight="bold",
    )

    ax.axis("off")
    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved graph visualization to {output_path}")
        plt.close()
    else:
        plt.show()


def _hierarchical_layout(graph: ModelGraph, G) -> Dict:
    """
    Create hierarchical layout based on topological order.

    Args:
        graph: ModelGraph
        G: NetworkX graph

    Returns:
        Position dictionary
    """
    try:
        topo_order = graph.topological_sort()
    except Exception:
        # Fallback to spring layout if topological sort fails
        import networkx as nx
        return nx.spring_layout(G)

    # Group nodes by depth level
    levels: Dict[int, list] = {}
    for i, node in enumerate(topo_order):
        level = _compute_node_depth(graph, node.id)
        if level not in levels:
            levels[level] = []
        levels[level].append(node.id)

    # Assign positions
    pos = {}
    max_width = max(len(nodes) for nodes in levels.values())

    for level, node_ids in sorted(levels.items()):
        y = -level  # Negative so it goes downward
        width = len(node_ids)
        x_start = -(width - 1) / 2

        for i, node_id in enumerate(node_ids):
            x = x_start + i
            pos[node_id] = (x * 2, y * 2)  # Scale for spacing

    return pos


def _compute_node_depth(graph: ModelGraph, node_id: str) -> int:
    """Compute depth of node from input."""
    visited = set()
    queue = [(graph.get_input_nodes()[0].id if graph.get_input_nodes() else node_id, 0)]
    depths = {node_id: 0 for node_id in graph.nodes}

    while queue:
        current_id, depth = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        depths[current_id] = depth

        node = graph.nodes.get(current_id)
        if node:
            for succ in node.successors:
                if succ.id not in visited:
                    queue.append((succ.id, depth + 1))

    return depths.get(node_id, 0)


def _get_node_colors(graph: ModelGraph) -> list:
    """
    Assign colors to nodes based on operation type.

    Args:
        graph: ModelGraph

    Returns:
        List of colors for each node
    """
    color_map = {
        "input": "#90EE90",  # Light green
        "output": "#FFB6C1",  # Light pink
        "conv2d": "#87CEEB",  # Sky blue
        "conv3d": "#87CEEB",
        "dense": "#DDA0DD",  # Plum
        "linear": "#DDA0DD",
        "maxpool": "#F0E68C",  # Khaki
        "avgpool": "#F0E68C",
        "max_pool": "#F0E68C",
        "avg_pool": "#F0E68C",
        "dropout": "#FFE4B5",  # Moccasin
        "batch_norm": "#B0E0E6",  # Powder blue
        "layer_norm": "#B0E0E6",
        "batchnorm": "#B0E0E6",
        "relu": "#FFA07A",  # Light salmon
        "elu": "#FFA07A",
        "gelu": "#FFA07A",
        "sigmoid": "#FFA07A",
        "tanh": "#FFA07A",
        "flatten": "#D3D3D3",  # Light gray
    }

    colors = []
    for node_id in graph.nodes:
        node = graph.nodes[node_id]
        color = color_map.get(node.operation, "#CCCCCC")  # Default gray
        colors.append(color)

    return colors


def plot_training_history(
    history: Dict,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150,
) -> None:
    """
    Plot training history (fitness over generations).

    Args:
        history: Dictionary with 'generation', 'best_fitness', 'mean_fitness', etc.
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Example:
        >>> history = optimizer.get_history()
        >>> plot_training_history(history, 'training.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise GraphError("Visualization requires matplotlib. Install with: pip install matplotlib")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract data
    generations = [entry.get("generation", i) for i, entry in enumerate(history)]
    best_fitness = [entry.get("best_fitness", 0) for entry in history]
    mean_fitness = [entry.get("mean_fitness", 0) for entry in history]

    # Plot fitness evolution
    ax1.plot(generations, best_fitness, label="Best Fitness", marker="o", linewidth=2)
    ax1.plot(generations, mean_fitness, label="Mean Fitness", marker="s", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot diversity if available
    if "diversity" in history[0]:
        diversity = [entry.get("diversity", 0) for entry in history]
        ax2.plot(generations, diversity, label="Diversity", marker="^", linewidth=2, color="green")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Diversity")
        ax2.set_title("Population Diversity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Diversity data not available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved training history to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_architecture_comparison(
    graphs: Dict[str, ModelGraph],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 150,
) -> None:
    """
    Compare multiple architectures side by side.

    Args:
        graphs: Dictionary mapping names to ModelGraphs
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Example:
        >>> graphs = {'Model A': graph1, 'Model B': graph2}
        >>> plot_architecture_comparison(graphs, 'comparison.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise GraphError("Visualization requires matplotlib. Install with: pip install matplotlib")

    n_models = len(graphs)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (name, graph) in zip(axes, graphs.items()):
        # Plot individual graph
        _plot_graph_on_axis(graph, ax, title=name)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved comparison to {output_path}")
        plt.close()
    else:
        plt.show()


def _plot_graph_on_axis(graph: ModelGraph, ax, title: str) -> None:
    """Plot graph on specific matplotlib axis."""
    try:
        import networkx as nx
    except ImportError:
        return

    G = nx.DiGraph()

    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=node.operation)

    for edge_id, edge in graph.edges.items():
        if edge.source and edge.target:
            G.add_edge(edge.source.id, edge.target.id)

    pos = nx.spring_layout(G, k=1, iterations=30)
    node_colors = _get_node_colors(graph)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=15, alpha=0.6, ax=ax)

    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(f"{title}\n{len(graph.nodes)} nodes, depth {graph.get_depth()}", fontsize=10)
    ax.axis("off")


def export_graphviz(graph: ModelGraph, output_path: Union[str, Path]) -> None:
    """
    Export graph in Graphviz DOT format.

    Args:
        graph: ModelGraph to export
        output_path: Output .dot file path

    Example:
        >>> export_graphviz(graph, 'model.dot')
        >>> # Then: dot -Tpng model.dot -o model.png
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["digraph ModelGraph {", "    rankdir=TB;", "    node [shape=box, style=filled];", ""]

    # Add nodes
    for node_id, node in graph.nodes.items():
        short_id = node_id[:8]
        color = _get_graphviz_color(node.operation)
        lines.append(f'    "{short_id}" [label="{node.operation}", fillcolor="{color}"];')

    lines.append("")

    # Add edges
    for edge_id, edge in graph.edges.items():
        if edge.source and edge.target:
            src = edge.source.id[:8]
            tgt = edge.target.id[:8]
            lines.append(f'    "{src}" -> "{tgt}";')

    lines.append("}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Exported Graphviz DOT to {output_path}")


def _get_graphviz_color(operation: str) -> str:
    """Get Graphviz color for operation."""
    color_map = {
        "input": "lightgreen",
        "output": "lightpink",
        "conv2d": "skyblue",
        "dense": "plum",
        "maxpool": "khaki",
        "dropout": "moccasin",
        "batch_norm": "powderblue",
        "relu": "lightsalmon",
    }
    return color_map.get(operation, "lightgray")
