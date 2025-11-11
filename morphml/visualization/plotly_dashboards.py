"""Interactive Plotly dashboards for MorphML.

Create publication-quality interactive visualizations for NAS experiments.

Example:
    >>> from morphml.visualization.plotly_dashboards import InteractiveDashboard
    >>> dashboard = InteractiveDashboard()
    >>> fig = dashboard.create_experiment_dashboard(history)
    >>> fig.show()
"""

from typing import Any, Dict, List

import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None
    px = None

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class InteractiveDashboard:
    """
    Create interactive Plotly dashboards for NAS experiments.

    Provides comprehensive visualizations including:
    - Convergence curves
    - Pareto fronts (multi-objective)
    - Architecture distributions
    - Performance heatmaps

    Example:
        >>> dashboard = InteractiveDashboard()
        >>> fig = dashboard.create_experiment_dashboard(history)
        >>> fig.write_html("dashboard.html")
    """

    def __init__(self):
        """Initialize dashboard generator."""
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for interactive dashboards. " "Install with: pip install plotly"
            )
        logger.info("Initialized InteractiveDashboard")

    def create_experiment_dashboard(
        self, history: Dict[str, Any], title: str = "Experiment Dashboard"
    ) -> "go.Figure":
        """
        Create comprehensive experiment dashboard.

        Args:
            history: Optimization history with keys:
                - best_fitness: List of best fitness per generation
                - mean_fitness: List of mean fitness per generation
                - diversity: List of diversity scores
                - architectures: List of architecture dicts
            title: Dashboard title

        Returns:
            Plotly Figure with subplots

        Example:
            >>> fig = dashboard.create_experiment_dashboard(optimizer.history)
            >>> fig.show()
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Convergence Curve",
                "Fitness Distribution",
                "Diversity Over Time",
                "Performance vs Complexity",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # 1. Convergence curve
        generations = list(range(len(history.get("best_fitness", []))))
        best_fitness = history.get("best_fitness", [])
        mean_fitness = history.get("mean_fitness", [])

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=best_fitness,
                mode="lines+markers",
                name="Best Fitness",
                line={"color": "#FF6B6B", "width": 2},
                marker={"size": 6},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=mean_fitness,
                mode="lines",
                name="Mean Fitness",
                line={"color": "#4ECDC4", "width": 2, "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        # 2. Fitness distribution
        all_fitness = []
        for arch_list in history.get("population_fitness", []):
            all_fitness.extend(arch_list)

        if all_fitness:
            fig.add_trace(
                go.Histogram(
                    x=all_fitness,
                    nbinsx=30,
                    name="Fitness Distribution",
                    marker={"color": "#45B7D1"},
                ),
                row=1,
                col=2,
            )

        # 3. Diversity over time
        diversity = history.get("diversity", [])
        if diversity:
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=diversity,
                    mode="lines+markers",
                    name="Diversity",
                    line={"color": "#96CEB4", "width": 2},
                    marker={"size": 6},
                ),
                row=2,
                col=1,
            )

        # 4. Performance vs Complexity
        architectures = history.get("architectures", [])
        if architectures:
            params = [a.get("parameters", 0) for a in architectures]
            fitness = [a.get("fitness", 0) for a in architectures]

            fig.add_trace(
                go.Scatter(
                    x=params,
                    y=fitness,
                    mode="markers",
                    name="Architectures",
                    marker={
                        "size": 8,
                        "color": fitness,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Fitness"},
                    },
                    text=[f"Params: {p:,}" for p in params],
                    hovertemplate="<b>Fitness:</b> %{y:.4f}<br><b>%{text}</b>",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_xaxes(title_text="Generation", row=1, col=1)
        fig.update_yaxes(title_text="Fitness", row=1, col=1)

        fig.update_xaxes(title_text="Fitness", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.update_xaxes(title_text="Generation", row=2, col=1)
        fig.update_yaxes(title_text="Diversity", row=2, col=1)

        fig.update_xaxes(title_text="Parameters", row=2, col=2)
        fig.update_yaxes(title_text="Fitness", row=2, col=2)

        fig.update_layout(title=title, height=800, showlegend=True, template="plotly_white")

        return fig

    def create_pareto_front_3d(
        self,
        architectures: List[Dict[str, Any]],
        objectives: List[str] = ["accuracy", "latency", "parameters"],
    ) -> "go.Figure":
        """
        Create 3D Pareto front visualization.

        Args:
            architectures: List of architecture dicts with objective values
            objectives: List of 3 objective names

        Returns:
            3D scatter plot

        Example:
            >>> fig = dashboard.create_pareto_front_3d(architectures)
            >>> fig.show()
        """
        if len(objectives) != 3:
            raise ValueError("Exactly 3 objectives required for 3D plot")

        # Extract objective values
        x = [a.get(objectives[0], 0) for a in architectures]
        y = [a.get(objectives[1], 0) for a in architectures]
        z = [a.get(objectives[2], 0) for a in architectures]

        # Create 3D scatter
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={
                        "size": 6,
                        "color": x,  # Color by first objective
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": objectives[0]},
                    },
                    text=[f"ID: {a.get('id', 'N/A')}" for a in architectures],
                    hovertemplate=(
                        f"<b>{objectives[0]}:</b> %{{x:.4f}}<br>"
                        f"<b>{objectives[1]}:</b> %{{y:.4f}}<br>"
                        f"<b>{objectives[2]}:</b> %{{z:.4f}}<br>"
                        "<b>%{text}</b>"
                    ),
                )
            ]
        )

        fig.update_layout(
            title="3D Pareto Front",
            scene={
                "xaxis_title": objectives[0],
                "yaxis_title": objectives[1],
                "zaxis_title": objectives[2],
            },
            height=700,
        )

        return fig

    def create_operation_distribution(self, architectures: List[Dict[str, Any]]) -> "go.Figure":
        """
        Create operation distribution pie chart.

        Args:
            architectures: List of architecture dicts

        Returns:
            Pie chart figure
        """
        # Count operations
        operation_counts = {}
        for arch in architectures:
            for node in arch.get("nodes", []):
                op = node.get("operation", "unknown")
                operation_counts[op] = operation_counts.get(op, 0) + 1

        # Create pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(operation_counts.keys()),
                    values=list(operation_counts.values()),
                    hole=0.3,
                    marker={"colors": px.colors.qualitative.Set3},
                )
            ]
        )

        fig.update_layout(title="Operation Distribution", height=500)

        return fig

    def create_convergence_comparison(self, experiments: Dict[str, Dict[str, Any]]) -> "go.Figure":
        """
        Compare convergence across multiple experiments.

        Args:
            experiments: Dict mapping experiment names to history dicts

        Returns:
            Line plot comparing convergence

        Example:
            >>> experiments = {
            ...     'GA': ga_history,
            ...     'Random': random_history,
            ...     'Bayesian': bayesian_history
            ... }
            >>> fig = dashboard.create_convergence_comparison(experiments)
        """
        fig = go.Figure()

        colors = px.colors.qualitative.Set1

        for i, (name, history) in enumerate(experiments.items()):
            generations = list(range(len(history.get("best_fitness", []))))
            best_fitness = history.get("best_fitness", [])

            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=best_fitness,
                    mode="lines+markers",
                    name=name,
                    line={"color": colors[i % len(colors)], "width": 2},
                    marker={"size": 6},
                )
            )

        fig.update_layout(
            title="Convergence Comparison",
            xaxis_title="Generation",
            yaxis_title="Best Fitness",
            height=500,
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def create_performance_heatmap(
        self,
        architectures: List[Dict[str, Any]],
        x_metric: str = "depth",
        y_metric: str = "width",
        z_metric: str = "fitness",
    ) -> "go.Figure":
        """
        Create performance heatmap.

        Args:
            architectures: List of architecture dicts
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            z_metric: Metric for color (performance)

        Returns:
            Heatmap figure
        """
        # Extract metrics
        x_values = [a.get(x_metric, 0) for a in architectures]
        y_values = [a.get(y_metric, 0) for a in architectures]
        z_values = [a.get(z_metric, 0) for a in architectures]

        # Create bins
        x_bins = np.linspace(min(x_values), max(x_values), 20)
        y_bins = np.linspace(min(y_values), max(y_values), 20)

        # Compute heatmap
        heatmap, xedges, yedges = np.histogram2d(
            x_values, y_values, bins=[x_bins, y_bins], weights=z_values
        )

        counts, _, _ = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins])
        heatmap = np.divide(heatmap, counts, where=counts != 0)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap.T,
                x=xedges[:-1],
                y=yedges[:-1],
                colorscale="Viridis",
                colorbar={"title": z_metric},
            )
        )

        fig.update_layout(
            title=f"{z_metric} Heatmap", xaxis_title=x_metric, yaxis_title=y_metric, height=500
        )

        return fig

    def export_dashboard(self, fig: "go.Figure", output_path: str, format: str = "html"):
        """
        Export dashboard to file.

        Args:
            fig: Plotly figure
            output_path: Output file path
            format: Export format ('html', 'png', 'svg', 'pdf')

        Example:
            >>> dashboard.export_dashboard(fig, "dashboard.html")
        """
        if format == "html":
            fig.write_html(output_path)
        elif format == "png":
            fig.write_image(output_path, format="png")
        elif format == "svg":
            fig.write_image(output_path, format="svg")
        elif format == "pdf":
            fig.write_image(output_path, format="pdf")
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported dashboard to {output_path}")
