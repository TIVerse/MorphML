"""Report generation tools.

This module provides tools to generate comprehensive HTML reports
for neural architecture search experiments.

Example:
    >>> from morphml.reports import ReportGenerator, create_comparison_report
    >>> 
    >>> # Quick report
    >>> create_comparison_report(
    ...     "My Experiment",
    ...     comparison_results,
    ...     "report.html"
    ... )
    
    >>> # Custom report
    >>> generator = ReportGenerator("Custom Report")
    >>> generator.add_section("Results", content)
    >>> generator.generate("custom_report.html")
"""

from morphml.reports.generator import (
    ReportGenerator,
    create_comparison_report,
)

__all__ = [
    "ReportGenerator",
    "create_comparison_report",
]
