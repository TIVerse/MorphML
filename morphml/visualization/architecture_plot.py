"""Architecture visualization utilities.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Optional

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


def plot_architecture(
    graph: ModelGraph,
    save_path: Optional[str] = None,
    title: str = "Neural Architecture"
) -> None:
    """
    Visualize neural architecture as a graph.
    
    Args:
        graph: ModelGraph to visualize
        save_path: Path to save plot (displays if None)
        title: Plot title
        
    Example:
        >>> from morphml.visualization.architecture_plot import plot_architecture
        >>> plot_architecture(best_individual.graph)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("matplotlib and networkx required for architecture plotting")
        return
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=f"{node.operation}\n{node_id[:8]}")
    
    # Add edges
    for edge in graph.edges:
        G.add_edge(edge.source.id, edge.target.id)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', 
                          alpha=0.9, edgecolors='black', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', width=2)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Architecture plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_architecture_hierarchy(
    graph: ModelGraph,
    save_path: Optional[str] = None,
    title: str = "Architecture Hierarchy"
) -> None:
    """
    Visualize architecture with hierarchical layout.
    
    Args:
        graph: ModelGraph to visualize
        save_path: Path to save plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("matplotlib and networkx required")
        return
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, label=node.operation)
    
    for edge in graph.edges:
        G.add_edge(edge.source.id, edge.target.id)
    
    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout if graphviz not available
        pos = nx.spring_layout(G)
    
    plt.figure(figsize=(12, 10))
    
    # Color nodes by operation type
    node_colors = []
    for node_id in G.nodes():
        node = graph.nodes[node_id]
        if node.operation in ['input', 'output']:
            node_colors.append('lightgreen')
        elif 'conv' in node.operation:
            node_colors.append('lightblue')
        elif 'dense' in node.operation:
            node_colors.append('lightyellow')
        else:
            node_colors.append('lightgray')
    
    nx.draw(G, pos, node_color=node_colors, with_labels=False,
            node_size=2000, alpha=0.9, edgecolors='black', linewidths=2,
            edge_color='gray', arrows=True, arrowsize=15)
    
    # Add labels
    labels = {node_id: graph.nodes[node_id].operation for node_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Architecture hierarchy plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_architecture_stats(
    graphs: list,
    save_path: Optional[str] = None,
    title: str = "Architecture Statistics"
) -> None:
    """
    Plot statistics of multiple architectures.
    
    Args:
        graphs: List of ModelGraph objects
        save_path: Path to save plot
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required")
        return
    
    # Collect statistics
    node_counts = [len(g.nodes) for g in graphs]
    depths = [g.get_depth() for g in graphs]
    params = [g.estimate_parameters() / 1e6 for g in graphs]  # In millions
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Node count distribution
    axes[0].hist(node_counts, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Nodes', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Node Count Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Depth distribution
    axes[1].hist(depths, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Depth', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Depth Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Parameter count distribution
    axes[2].hist(params, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[2].set_xlabel('Parameters (Millions)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Parameter Count Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Architecture statistics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Re-export from existing visualization module for convenience
try:
    from morphml.visualization.graph_viz import visualize_graph
except ImportError:
    pass
