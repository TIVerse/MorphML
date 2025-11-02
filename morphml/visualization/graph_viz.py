"""Graph visualization for neural architectures."""

from typing import Optional, Tuple

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class GraphVisualizer:
    """
    Visualize neural architecture graphs.

    Supports multiple output formats:
    - Graphviz DOT
    - NetworkX plots
    - ASCII art
    - JSON

    Example:
        >>> viz = GraphVisualizer()
        >>> viz.to_graphviz(graph, 'architecture.dot')
        >>> viz.plot_graph(graph, 'architecture.png')
    """

    def __init__(self, style: str = "default"):
        """
        Initialize visualizer.

        Args:
            style: Visualization style ('default', 'modern', 'simple')
        """
        self.style = style
        self.node_colors = {
            "input": "#4CAF50",
            "conv2d": "#2196F3",
            "dense": "#FF9800",
            "maxpool": "#9C27B0",
            "avgpool": "#9C27B0",
            "dropout": "#F44336",
            "batchnorm": "#00BCD4",
            "relu": "#FFEB3B",
            "sigmoid": "#FFEB3B",
            "tanh": "#FFEB3B",
            "softmax": "#FFEB3B",
            "output": "#4CAF50",
        }

    def to_graphviz(self, graph: ModelGraph, output_path: str) -> None:
        """
        Export graph to Graphviz DOT format.

        Args:
            graph: Graph to visualize
            output_path: Output file path
        """
        try:
            import graphviz
        except ImportError:
            logger.warning("graphviz not installed")
            self._write_dot_file(graph, output_path)
            return

        dot = graphviz.Digraph(comment="Neural Architecture")
        dot.attr(rankdir="TB")

        # Add nodes
        for node_id, node in graph.nodes.items():
            color = self.node_colors.get(node.operation, "#9E9E9E")
            label = self._format_node_label(node)

            dot.node(node_id, label, style="filled", fillcolor=color, shape="box", fontname="Arial")

        # Add edges
        for edge in graph.edges:
            dot.edge(edge.source.id, edge.target.id)

        # Render
        dot.render(output_path, format="png", cleanup=True)
        logger.info(f"Graph visualization saved to {output_path}.png")

    def _write_dot_file(self, graph: ModelGraph, output_path: str) -> None:
        """Write DOT file without graphviz library."""
        with open(output_path, "w") as f:
            f.write("digraph G {\n")
            f.write("  rankdir=TB;\n")
            f.write("  node [shape=box, style=filled];\n\n")

            # Nodes
            for node_id, node in graph.nodes.items():
                color = self.node_colors.get(node.operation, "#9E9E9E")
                label = self._format_node_label(node)
                f.write(f'  "{node_id}" [label="{label}", fillcolor="{color}"];\n')

            f.write("\n")

            # Edges
            for edge in graph.edges:
                f.write(f'  "{edge.source.id}" -> "{edge.target.id}";\n')

            f.write("}\n")

        logger.info(f"DOT file saved to {output_path}")

    def _format_node_label(self, node) -> str:
        """Format node label with operation and parameters."""
        label = node.operation

        if node.params:
            params_str = []
            for key, value in list(node.params.items())[:3]:  # Limit to 3 params
                params_str.append(f"{key}={value}")

            if params_str:
                label += "\\n" + "\\n".join(params_str)

        return label

    def to_ascii(self, graph: ModelGraph) -> str:
        """
        Generate ASCII art representation.

        Args:
            graph: Graph to visualize

        Returns:
            ASCII art string
        """
        try:
            sorted_nodes = graph.topological_sort()
        except Exception:
            sorted_nodes = list(graph.nodes.values())

        lines = []
        lines.append("=" * 60)
        lines.append("NEURAL ARCHITECTURE")
        lines.append("=" * 60)
        lines.append("")

        for i, node in enumerate(sorted_nodes):
            # Node info
            op = node.operation
            params = ", ".join(f"{k}={v}" for k, v in list(node.params.items())[:2])

            lines.append(f"[{i+1}] {op}")
            if params:
                lines.append(f"    {params}")

            # Show connections
            if i < len(sorted_nodes) - 1:
                lines.append("    |")
                lines.append("    v")

            lines.append("")

        lines.append("=" * 60)
        lines.append(f"Total Nodes: {len(graph.nodes)}")
        lines.append(f"Total Edges: {len(graph.edges)}")
        lines.append(f"Depth: {graph.get_depth()}")
        lines.append(f"Est. Parameters: {graph.estimate_parameters():,}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def plot_networkx(
        self,
        graph: ModelGraph,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot graph using NetworkX and matplotlib.

        Args:
            graph: Graph to visualize
            output_path: Path to save plot (displays if None)
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.warning("matplotlib or networkx not installed")
            return

        # Convert to NetworkX
        G = graph.to_networkx()

        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Plot
        plt.figure(figsize=figsize)

        # Draw nodes with colors
        node_colors_list = []
        for node_id in G.nodes():
            node = graph.nodes[node_id]
            color = self.node_colors.get(node.operation, "#9E9E9E")
            node_colors_list.append(color)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=2000, alpha=0.9)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, width=2)

        # Draw labels
        labels = {}
        for node_id in G.nodes():
            node = graph.nodes[node_id]
            labels[node_id] = node.operation

        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

        plt.title(f"Neural Architecture (Nodes: {len(graph.nodes)}, Depth: {graph.get_depth()})")
        plt.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"NetworkX plot saved to {output_path}")
        else:
            plt.show()

    def to_html(self, graph: ModelGraph, output_path: str) -> None:
        """
        Generate interactive HTML visualization.

        Args:
            graph: Graph to visualize
            output_path: Output HTML file path
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed, falling back to simple HTML")
            self._simple_html(graph, output_path)
            return

        net = Network(height="750px", width="100%", directed=True)
        net.barnes_hut()

        # Add nodes
        for node_id, node in graph.nodes.items():
            color = self.node_colors.get(node.operation, "#9E9E9E")
            title = f"{node.operation}\n" + "\n".join(f"{k}: {v}" for k, v in node.params.items())

            net.add_node(node_id, label=node.operation, title=title, color=color)

        # Add edges
        for edge in graph.edges:
            net.add_edge(edge.source.id, edge.target.id)

        net.save_graph(output_path)
        logger.info(f"Interactive HTML saved to {output_path}")

    def _simple_html(self, graph: ModelGraph, output_path: str) -> None:
        """Generate simple HTML table."""
        html = ["<!DOCTYPE html>"]
        html.append("<html><head><title>Neural Architecture</title>")
        html.append("<style>")
        html.append("body { font-family: Arial; margin: 20px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append("</style></head><body>")
        html.append("<h1>Neural Architecture</h1>")
        html.append(f"<p><strong>Nodes:</strong> {len(graph.nodes)}</p>")
        html.append(f"<p><strong>Depth:</strong> {graph.get_depth()}</p>")
        html.append(f"<p><strong>Parameters:</strong> {graph.estimate_parameters():,}</p>")
        html.append("<h2>Layers</h2>")
        html.append("<table><tr><th>ID</th><th>Operation</th><th>Parameters</th></tr>")

        for node_id, node in graph.nodes.items():
            params = ", ".join(f"{k}={v}" for k, v in node.params.items())
            html.append(
                f"<tr><td>{node_id[:8]}</td><td>{node.operation}</td><td>{params}</td></tr>"
            )

        html.append("</table></body></html>")

        with open(output_path, "w") as f:
            f.write("\n".join(html))

        logger.info(f"Simple HTML saved to {output_path}")

    def compare_graphs(self, graphs: list, labels: list, output_path: Optional[str] = None) -> None:
        """
        Compare multiple graphs side by side.

        Args:
            graphs: List of ModelGraph instances
            labels: List of labels for each graph
            output_path: Output path for comparison plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed")
            return

        n = len(graphs)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

        if n == 1:
            axes = [axes]

        for ax, graph, label in zip(axes, graphs, labels):
            # Plot each graph
            try:
                import networkx as nx

                G = graph.to_networkx()
                pos = nx.spring_layout(G)

                nx.draw(
                    G,
                    pos,
                    ax=ax,
                    with_labels=False,
                    node_color="lightblue",
                    node_size=500,
                    arrows=True,
                )

                ax.set_title(f"{label}\n(Nodes: {len(graph.nodes)}, Depth: {graph.get_depth()})")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")
                ax.set_title(label)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {output_path}")
        else:
            plt.show()
