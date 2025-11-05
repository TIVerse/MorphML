"""Graph-based model representation."""

from morphml.core.graph.edge import GraphEdge
from morphml.core.graph.graph import ModelGraph
from morphml.core.graph.mutations import GraphMutator, crossover
from morphml.core.graph.node import GraphNode
from morphml.core.graph.serialization import (
    save_graph,
    load_graph,
    graph_to_json_string,
    graph_from_json_string,
    export_graph_summary,
    batch_save_graphs,
    batch_load_graphs,
)
from morphml.core.graph.visualization import (
    plot_graph,
    plot_training_history,
    plot_architecture_comparison,
    export_graphviz,
)

__all__ = [
    "GraphNode",
    "GraphEdge",
    "ModelGraph",
    "GraphMutator",
    "crossover",
    "save_graph",
    "load_graph",
    "graph_to_json_string",
    "graph_from_json_string",
    "export_graph_summary",
    "batch_save_graphs",
    "batch_load_graphs",
    "plot_graph",
    "plot_training_history",
    "plot_architecture_comparison",
    "export_graphviz",
]
