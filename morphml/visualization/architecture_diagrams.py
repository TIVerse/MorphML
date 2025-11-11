"""Professional architecture diagram generation for MorphML.

Generate publication-quality diagrams of neural architectures.

Example:
    >>> from morphml.visualization.architecture_diagrams import ArchitectureDiagramGenerator
    >>> generator = ArchitectureDiagramGenerator()
    >>> generator.generate_svg(graph, "architecture.svg")
"""

from typing import Dict, Optional

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    graphviz = None

from morphml.core.graph import GraphNode, ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureDiagramGenerator:
    """
    Generate professional architecture diagrams.

    Creates publication-quality visualizations of neural architectures
    using Graphviz with custom styling.

    Example:
        >>> generator = ArchitectureDiagramGenerator()
        >>> generator.generate_svg(graph, "output.svg")
        >>> generator.generate_png(graph, "output.png", dpi=300)
    """

    # Color scheme for different operations
    OPERATION_COLORS = {
        "input": "#90EE90",  # Light green
        "output": "#FFB6C1",  # Light pink
        "conv2d": "#FF6B6B",  # Red
        "dense": "#4ECDC4",  # Teal
        "maxpool": "#45B7D1",  # Blue
        "avgpool": "#5DADE2",  # Light blue
        "relu": "#96CEB4",  # Green
        "sigmoid": "#FFEAA7",  # Yellow
        "tanh": "#DFE6E9",  # Gray
        "softmax": "#FD79A8",  # Pink
        "batchnorm": "#A29BFE",  # Purple
        "dropout": "#FDCB6E",  # Orange
        "flatten": "#E17055",  # Dark orange
    }

    def __init__(self, style: str = "modern"):
        """
        Initialize diagram generator.

        Args:
            style: Visual style ('modern', 'classic', 'minimal')
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError(
                "Graphviz is required for architecture diagrams. "
                "Install with: pip install graphviz"
            )

        self.style = style
        logger.info(f"Initialized ArchitectureDiagramGenerator with style: {style}")

    def generate_svg(self, graph: ModelGraph, output_path: str, title: Optional[str] = None):
        """
        Generate SVG diagram.

        Args:
            graph: ModelGraph to visualize
            output_path: Output file path (without extension)
            title: Optional diagram title

        Example:
            >>> generator.generate_svg(graph, "architecture")
            # Creates architecture.svg
        """
        dot = self._create_graphviz_graph(graph, title)
        dot.render(output_path, format="svg", cleanup=True)
        logger.info(f"Generated SVG diagram: {output_path}.svg")

    def generate_png(
        self, graph: ModelGraph, output_path: str, dpi: int = 300, title: Optional[str] = None
    ):
        """
        Generate PNG diagram.

        Args:
            graph: ModelGraph to visualize
            output_path: Output file path (without extension)
            dpi: Resolution in DPI
            title: Optional diagram title

        Example:
            >>> generator.generate_png(graph, "architecture", dpi=300)
            # Creates architecture.png
        """
        dot = self._create_graphviz_graph(graph, title)
        dot.graph_attr["dpi"] = str(dpi)
        dot.render(output_path, format="png", cleanup=True)
        logger.info(f"Generated PNG diagram: {output_path}.png")

    def generate_pdf(self, graph: ModelGraph, output_path: str, title: Optional[str] = None):
        """
        Generate PDF diagram.

        Args:
            graph: ModelGraph to visualize
            output_path: Output file path (without extension)
            title: Optional diagram title

        Example:
            >>> generator.generate_pdf(graph, "architecture")
            # Creates architecture.pdf
        """
        dot = self._create_graphviz_graph(graph, title)
        dot.render(output_path, format="pdf", cleanup=True)
        logger.info(f"Generated PDF diagram: {output_path}.pdf")

    def _create_graphviz_graph(
        self, graph: ModelGraph, title: Optional[str] = None
    ) -> "graphviz.Digraph":
        """
        Create Graphviz graph from ModelGraph.

        Args:
            graph: ModelGraph to convert
            title: Optional title

        Returns:
            Graphviz Digraph
        """
        dot = graphviz.Digraph(comment="Neural Architecture")

        # Set graph attributes based on style
        if self.style == "modern":
            dot.attr(
                rankdir="TB",
                bgcolor="white",
                fontname="Arial",
                fontsize="12",
                splines="ortho",
                nodesep="0.5",
                ranksep="0.8",
            )
        elif self.style == "classic":
            dot.attr(rankdir="TB", bgcolor="white", fontname="Times", fontsize="11")
        else:  # minimal
            dot.attr(
                rankdir="TB", bgcolor="white", fontname="Helvetica", fontsize="10", splines="line"
            )

        # Add title if provided
        if title:
            dot.attr(label=title, labelloc="t", fontsize="16")

        # Add nodes
        for _node_id, node in graph.nodes.items():
            self._add_node(dot, node)

        # Add edges
        for _edge_id, edge in graph.edges.items():
            self._add_edge(dot, edge)

        return dot

    def _add_node(self, dot: "graphviz.Digraph", node: GraphNode):
        """
        Add node to Graphviz graph.

        Args:
            dot: Graphviz graph
            node: GraphNode to add
        """
        operation = node.operation

        # Get color
        color = self.OPERATION_COLORS.get(operation, "#CCCCCC")

        # Create label
        label = self._create_node_label(node)

        # Node attributes
        node_attrs = {
            "label": label,
            "shape": "box",
            "style": "filled,rounded",
            "fillcolor": color,
            "fontname": "Arial",
            "fontsize": "10",
            "margin": "0.2,0.1",
        }

        # Special styling for input/output
        if operation == "input":
            node_attrs["shape"] = "ellipse"
            node_attrs["style"] = "filled"
        elif operation == "output":
            node_attrs["shape"] = "ellipse"
            node_attrs["style"] = "filled"

        dot.node(str(node.id), **node_attrs)

    def _create_node_label(self, node: GraphNode) -> str:
        """
        Create formatted label for node.

        Args:
            node: GraphNode

        Returns:
            Formatted label string
        """
        operation = node.operation
        params = node.params

        # Base label
        label = f"{operation}"

        # Add key parameters
        if operation == "conv2d":
            filters = params.get("filters", "?")
            kernel = params.get("kernel_size", "?")
            label += f"\n{filters} filters\n{kernel}x{kernel} kernel"

        elif operation == "dense":
            units = params.get("units", "?")
            label += f"\n{units} units"

        elif operation in ["maxpool", "avgpool"]:
            pool_size = params.get("pool_size", "?")
            label += f"\n{pool_size}x{pool_size} pool"

        elif operation == "dropout":
            rate = params.get("rate", "?")
            label += f"\nrate={rate}"

        elif operation == "input":
            shape = params.get("shape", "?")
            label += f"\nshape={shape}"

        return label

    def _add_edge(self, dot: "graphviz.Digraph", edge):
        """
        Add edge to Graphviz graph.

        Args:
            dot: Graphviz graph
            edge: GraphEdge to add
        """
        dot.edge(
            str(edge.source.id),
            str(edge.target.id),
            color="#2C3E50",
            penwidth="1.5",
            arrowsize="0.8",
        )

    def generate_comparison_diagram(
        self, graphs: Dict[str, ModelGraph], output_path: str, format: str = "svg"
    ):
        """
        Generate side-by-side comparison of multiple architectures.

        Args:
            graphs: Dict mapping names to ModelGraphs
            output_path: Output file path (without extension)
            format: Output format ('svg', 'png', 'pdf')

        Example:
            >>> generator.generate_comparison_diagram({
            ...     'Architecture A': graph_a,
            ...     'Architecture B': graph_b
            ... }, "comparison")
        """
        # Create compound graph
        dot = graphviz.Digraph(comment="Architecture Comparison")
        dot.attr(rankdir="LR", compound="true")

        # Add subgraphs for each architecture
        for i, (name, graph) in enumerate(graphs.items()):
            with dot.subgraph(name=f"cluster_{i}") as sub:
                sub.attr(label=name, style="rounded", color="gray")

                # Add nodes
                for _node_id, node in graph.nodes.items():
                    self._add_node(sub, node)

                # Add edges
                for _edge_id, edge in graph.edges.items():
                    self._add_edge(sub, edge)

        # Render
        dot.render(output_path, format=format, cleanup=True)
        logger.info(f"Generated comparison diagram: {output_path}.{format}")

    def generate_layer_statistics_diagram(self, graph: ModelGraph, output_path: str):
        """
        Generate diagram with layer statistics.

        Args:
            graph: ModelGraph to visualize
            output_path: Output file path (without extension)

        Example:
            >>> generator.generate_layer_statistics_diagram(graph, "stats")
        """
        dot = graphviz.Digraph(comment="Architecture with Statistics")
        dot.attr(rankdir="TB", bgcolor="white")

        # Calculate statistics
        total_params = graph.estimate_parameters()
        depth = graph.estimate_depth()
        width = graph.estimate_width()

        # Add info box
        info_label = (
            f"Total Parameters: {total_params:,}\\n"
            f"Depth: {depth}\\n"
            f"Width: {width}\\n"
            f"Nodes: {len(graph.nodes)}"
        )

        dot.node(
            "info",
            label=info_label,
            shape="box",
            style="filled",
            fillcolor="#E8F4F8",
            fontname="Courier",
            fontsize="10",
        )

        # Add architecture nodes
        for _node_id, node in graph.nodes.items():
            self._add_node(dot, node)

        # Add edges
        for _edge_id, edge in graph.edges.items():
            self._add_edge(dot, edge)

        # Render
        dot.render(output_path, format="svg", cleanup=True)
        logger.info(f"Generated statistics diagram: {output_path}.svg")
