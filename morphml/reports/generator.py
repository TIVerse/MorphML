"""Report generation for NAS experiments.

This module provides tools to generate comprehensive HTML reports
summarizing optimization results, comparisons, and architectures.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate HTML reports for NAS experiments.

    Creates comprehensive reports including:
    - Experiment summary
    - Optimizer comparisons
    - Performance metrics
    - Best architectures
    - Convergence plots

    Example:
        >>> from morphml.reports import ReportGenerator
        >>>
        >>> generator = ReportGenerator("My NAS Experiment")
        >>> generator.add_section("Overview", "Experiment description...")
        >>> generator.add_optimizer_results("GP", results)
        >>> generator.generate("report.html")
    """

    def __init__(self, experiment_name: str):
        """
        Initialize report generator.

        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.sections: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "experiment_name": experiment_name,
        }

        logger.debug(f"Initialized ReportGenerator for '{experiment_name}'")

    def add_section(self, title: str, content: str, section_type: str = "text") -> None:
        """
        Add a section to the report.

        Args:
            title: Section title
            content: Section content (HTML or text)
            section_type: Type of section ('text', 'html', 'table', 'image')
        """
        self.sections.append({"title": title, "content": content, "type": section_type})
        logger.debug(f"Added section: {title}")

    def add_optimizer_results(self, optimizer_name: str, results: Dict[str, Any]) -> None:
        """
        Add optimizer results section.

        Args:
            optimizer_name: Name of the optimizer
            results: Results dictionary with statistics
        """
        content = self._format_optimizer_results(optimizer_name, results)
        self.add_section(f"{optimizer_name} Results", content, "html")

    def add_comparison_table(self, comparison_data: Dict[str, Dict]) -> None:
        """
        Add optimizer comparison table.

        Args:
            comparison_data: Dictionary of optimizer results
        """
        content = self._create_comparison_table(comparison_data)
        self.add_section("Optimizer Comparison", content, "html")

    def generate(self, output_path: str) -> None:
        """
        Generate HTML report.

        Args:
            output_path: Path to save the report
        """
        html = self._generate_html()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Report generated: {output_path}")

    def _format_optimizer_results(self, name: str, results: Dict[str, Any]) -> str:
        """Format optimizer results as HTML."""
        html = "<div class='optimizer-results'>\n"
        html += f"<h3>{name}</h3>\n"

        if "mean_best" in results:
            html += "<table class='results-table'>\n"
            html += "<tr><th>Metric</th><th>Value</th></tr>\n"
            html += f"<tr><td>Mean Best</td><td>{results['mean_best']:.4f}</td></tr>\n"
            html += f"<tr><td>Std Dev</td><td>{results['std_best']:.4f}</td></tr>\n"
            html += f"<tr><td>Min</td><td>{results['min_best']:.4f}</td></tr>\n"
            html += f"<tr><td>Max</td><td>{results['max_best']:.4f}</td></tr>\n"
            html += f"<tr><td>Median</td><td>{results['median_best']:.4f}</td></tr>\n"

            if "mean_time" in results:
                html += f"<tr><td>Mean Time (s)</td><td>{results['mean_time']:.2f}</td></tr>\n"

            html += "</table>\n"

        html += "</div>\n"
        return html

    def _create_comparison_table(self, comparison_data: Dict[str, Dict]) -> str:
        """Create comparison table HTML."""
        html = "<table class='comparison-table'>\n"
        html += "<tr><th>Optimizer</th><th>Mean Best</th><th>Std</th><th>Min</th><th>Max</th><th>Time (s)</th></tr>\n"

        # Sort by mean_best descending
        sorted_data = sorted(
            comparison_data.items(), key=lambda x: x[1].get("mean_best", 0), reverse=True
        )

        for i, (name, results) in enumerate(sorted_data):
            row_class = "best-row" if i == 0 else ""
            html += f"<tr class='{row_class}'>\n"
            html += f"<td><strong>{name}</strong></td>\n"
            html += f"<td>{results.get('mean_best', 0):.4f}</td>\n"
            html += f"<td>{results.get('std_best', 0):.4f}</td>\n"
            html += f"<td>{results.get('min_best', 0):.4f}</td>\n"
            html += f"<td>{results.get('max_best', 0):.4f}</td>\n"
            html += f"<td>{results.get('mean_time', 0):.2f}</td>\n"
            html += "</tr>\n"

        html += "</table>\n"
        return html

    def _generate_html(self) -> str:
        """Generate complete HTML document."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.experiment_name} - Report</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.experiment_name}</h1>
            <p class="subtitle">Neural Architecture Search Report</p>
            <p class="meta">Generated: {self.metadata['generated_at']}</p>
        </header>

        <main>
"""

        # Add all sections
        for section in self.sections:
            html += self._render_section(section)

        html += """
        </main>

        <footer>
            <p>Generated by MorphML - Morphological Machine Learning</p>
            <p>Organization: TONMOY INFRASTRUCTURE & VISION</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _render_section(self, section: Dict[str, Any]) -> str:
        """Render a single section."""
        html = "<section>\n"
        html += f"<h2>{section['title']}</h2>\n"

        if section["type"] == "text":
            html += f"<p>{section['content']}</p>\n"
        elif section["type"] == "html":
            html += section["content"]
        elif section["type"] == "table":
            html += section["content"]

        html += "</section>\n"
        return html

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 10px;
        }

        .meta {
            font-size: 0.9em;
            opacity: 0.8;
        }

        main {
            padding: 40px;
        }

        section {
            margin-bottom: 40px;
        }

        h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        table th, table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        table th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }

        table tr:hover {
            background: #f5f5f5;
        }

        .best-row {
            background: #d4edda !important;
            font-weight: 600;
        }

        .optimizer-results {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .results-table {
            background: white;
        }

        footer {
            background: #333;
            color: white;
            padding: 20px;
            text-align: center;
        }

        footer p {
            margin: 5px 0;
            font-size: 0.9em;
        }
        """


# Convenience function
def create_comparison_report(
    experiment_name: str,
    comparison_results: Dict[str, Dict],
    output_path: str,
    description: Optional[str] = None,
) -> None:
    """
    Quick function to create a comparison report.

    Args:
        experiment_name: Name of the experiment
        comparison_results: Results from OptimizerComparison
        output_path: Where to save the report
        description: Optional experiment description

    Example:
        >>> from morphml.reports import create_comparison_report
        >>>
        >>> create_comparison_report(
        ...     "NAS Comparison",
        ...     results,
        ...     "comparison_report.html",
        ...     "Comparing 4 optimizers on CIFAR-10"
        ... )
    """
    generator = ReportGenerator(experiment_name)

    if description:
        generator.add_section("Experiment Description", description, "text")

    # Add comparison table
    generator.add_comparison_table(comparison_results)

    # Add individual results
    for name, results in comparison_results.items():
        generator.add_optimizer_results(name, results)

    # Generate report
    generator.generate(output_path)

    logger.info(f"Comparison report created: {output_path}")
