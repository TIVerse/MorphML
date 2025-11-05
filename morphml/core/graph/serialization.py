"""Serialization utilities for model graphs.

Provides functions to save and load ModelGraph objects in various formats:
- JSON: Human-readable, good for inspection
- Pickle: Python-native, preserves exact state
- YAML: Configuration-friendly format

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

from morphml.core.graph.graph import ModelGraph
from morphml.exceptions import GraphError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


def save_graph(
    graph: ModelGraph,
    path: Union[str, Path],
    format: str = "json",
    indent: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model graph to file.

    Args:
        graph: ModelGraph to save
        path: Output file path
        format: Format ('json', 'pickle', 'yaml')
        indent: JSON indentation (for readability)
        metadata: Additional metadata to save with graph

    Raises:
        GraphError: If save fails

    Example:
        >>> graph = ModelGraph()
        >>> save_graph(graph, 'model.json')
        >>> save_graph(graph, 'model.pkl', format='pickle')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "json":
            _save_json(graph, path, indent, metadata)
        elif format == "pickle":
            _save_pickle(graph, path, metadata)
        elif format == "yaml":
            _save_yaml(graph, path, metadata)
        else:
            raise GraphError(f"Unknown format: {format}. Use 'json', 'pickle', or 'yaml'")

        logger.info(f"Saved graph to {path} (format: {format})")

    except Exception as e:
        logger.error(f"Failed to save graph: {e}")
        raise GraphError(f"Failed to save graph to {path}: {e}") from e


def load_graph(path: Union[str, Path], format: Optional[str] = None) -> ModelGraph:
    """
    Load model graph from file.

    Args:
        path: Input file path
        format: Format ('json', 'pickle', 'yaml'). If None, infers from extension

    Returns:
        Loaded ModelGraph

    Raises:
        GraphError: If load fails

    Example:
        >>> graph = load_graph('model.json')
        >>> graph = load_graph('model.pkl', format='pickle')
    """
    path = Path(path)

    if not path.exists():
        raise GraphError(f"File not found: {path}")

    # Infer format from extension if not specified
    if format is None:
        format = _infer_format(path)

    try:
        if format == "json":
            graph = _load_json(path)
        elif format == "pickle":
            graph = _load_pickle(path)
        elif format == "yaml":
            graph = _load_yaml(path)
        else:
            raise GraphError(f"Unknown format: {format}")

        logger.info(f"Loaded graph from {path} (format: {format})")
        return graph

    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        raise GraphError(f"Failed to load graph from {path}: {e}") from e


def _save_json(
    graph: ModelGraph, path: Path, indent: int, metadata: Optional[Dict[str, Any]]
) -> None:
    """Save graph as JSON."""
    data = graph.to_dict()

    # Add metadata if provided
    if metadata:
        data["_metadata"] = metadata

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=False)


def _load_json(path: Path) -> ModelGraph:
    """Load graph from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    # Remove metadata if present
    data.pop("_metadata", None)

    return ModelGraph.from_dict(data)


def _save_pickle(graph: ModelGraph, path: Path, metadata: Optional[Dict[str, Any]]) -> None:
    """Save graph as pickle."""
    data = {"graph": graph, "metadata": metadata}

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> ModelGraph:
    """Load graph from pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new formats
    if isinstance(data, ModelGraph):
        return data
    elif isinstance(data, dict) and "graph" in data:
        return data["graph"]
    else:
        raise GraphError("Invalid pickle format")


def _save_yaml(graph: ModelGraph, path: Path, metadata: Optional[Dict[str, Any]]) -> None:
    """Save graph as YAML."""
    try:
        import yaml
    except ImportError:
        raise GraphError("PyYAML not installed. Install with: pip install pyyaml")

    data = graph.to_dict()

    # Add metadata if provided
    if metadata:
        data["_metadata"] = metadata

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _load_yaml(path: Path) -> ModelGraph:
    """Load graph from YAML."""
    try:
        import yaml
    except ImportError:
        raise GraphError("PyYAML not installed. Install with: pip install pyyaml")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Remove metadata if present
    data.pop("_metadata", None)

    return ModelGraph.from_dict(data)


def _infer_format(path: Path) -> str:
    """Infer format from file extension."""
    suffix = path.suffix.lower()

    format_map = {
        ".json": "json",
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".yaml": "yaml",
        ".yml": "yaml",
    }

    format = format_map.get(suffix)
    if format is None:
        logger.warning(f"Unknown extension {suffix}, assuming JSON")
        return "json"

    return format


def graph_to_json_string(graph: ModelGraph, indent: int = 2) -> str:
    """
    Convert graph to JSON string.

    Args:
        graph: ModelGraph to convert
        indent: Indentation level

    Returns:
        JSON string representation

    Example:
        >>> json_str = graph_to_json_string(graph)
        >>> print(json_str)
    """
    return json.dumps(graph.to_dict(), indent=indent, sort_keys=False)


def graph_from_json_string(json_str: str) -> ModelGraph:
    """
    Create graph from JSON string.

    Args:
        json_str: JSON string representation

    Returns:
        ModelGraph instance

    Example:
        >>> graph = graph_from_json_string(json_str)
    """
    data = json.loads(json_str)
    return ModelGraph.from_dict(data)


def export_graph_summary(graph: ModelGraph, path: Union[str, Path]) -> None:
    """
    Export human-readable graph summary.

    Creates a text file with graph statistics and structure.

    Args:
        graph: ModelGraph to summarize
        path: Output file path

    Example:
        >>> export_graph_summary(graph, 'model_summary.txt')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("MODEL GRAPH SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Basic info
    lines.append(f"Nodes:  {len(graph.nodes)}")
    lines.append(f"Edges:  {len(graph.edges)}")
    lines.append(f"Depth:  {graph.get_depth()}")
    lines.append(f"Width:  {graph.get_max_width()}")
    lines.append(f"Params: {graph.estimate_parameters():,}")
    lines.append("")

    # Nodes
    lines.append("NODES:")
    lines.append("-" * 60)
    for node_id, node in graph.nodes.items():
        lines.append(f"  {node_id[:8]}... | {node.operation:15s} | {node.params}")

    lines.append("")

    # Edges
    lines.append("EDGES:")
    lines.append("-" * 60)
    for _edge_id, edge in graph.edges.items():
        src = edge.source.id[:8] if edge.source else "None"
        tgt = edge.target.id[:8] if edge.target else "None"
        lines.append(f"  {src}... -> {tgt}...")

    lines.append("")
    lines.append("=" * 60)

    with open(path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Exported graph summary to {path}")


def batch_save_graphs(
    graphs: Dict[str, ModelGraph], output_dir: Union[str, Path], format: str = "json"
) -> None:
    """
    Save multiple graphs to directory.

    Args:
        graphs: Dictionary mapping names to graphs
        output_dir: Output directory
        format: Save format

    Example:
        >>> graphs = {'model1': graph1, 'model2': graph2}
        >>> batch_save_graphs(graphs, 'models/', format='json')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, graph in graphs.items():
        ext = {"json": ".json", "pickle": ".pkl", "yaml": ".yaml"}[format]
        path = output_dir / f"{name}{ext}"
        save_graph(graph, path, format=format)

    logger.info(f"Saved {len(graphs)} graphs to {output_dir}")


def batch_load_graphs(
    input_dir: Union[str, Path], format: Optional[str] = None
) -> Dict[str, ModelGraph]:
    """
    Load all graphs from directory.

    Args:
        input_dir: Input directory
        format: Load format (None to infer from extensions)

    Returns:
        Dictionary mapping filenames to graphs

    Example:
        >>> graphs = batch_load_graphs('models/')
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise GraphError(f"Directory not found: {input_dir}")

    graphs = {}

    # Find all graph files
    patterns = ["*.json", "*.pkl", "*.pickle", "*.yaml", "*.yml"]
    for pattern in patterns:
        for path in input_dir.glob(pattern):
            name = path.stem
            graphs[name] = load_graph(path, format=format)

    logger.info(f"Loaded {len(graphs)} graphs from {input_dir}")
    return graphs
