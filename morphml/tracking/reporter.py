"""Report generation for experiments."""

import json
from datetime import datetime
from typing import Any, Dict, List

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class Reporter:
    """
    Generate reports for experiments.

    Example:
        >>> reporter = Reporter()
        >>> report = reporter.generate_markdown_report(experiment)
        >>> reporter.save_report(report, "report.md")
    """

    def __init__(self):
        """Initialize reporter."""
        pass

    def generate_markdown_report(self, experiment: Any) -> str:
        """
        Generate Markdown report.

        Args:
            experiment: Experiment instance

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# Experiment Report: {experiment.name}")
        lines.append(f"\n**ID:** `{experiment.id}`\n")
        lines.append(f"**Status:** {experiment.status}")
        lines.append(f"**Duration:** {experiment.get_duration():.2f} seconds\n")

        # Configuration
        lines.append("## Configuration\n")
        lines.append("```json")
        lines.append(json.dumps(experiment.config, indent=2))
        lines.append("```\n")

        # Best Result
        if experiment.best_result:
            lines.append("## Best Result\n")
            for key, value in experiment.best_result.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Metrics Summary
        if experiment.metrics:
            lines.append("## Metrics Summary\n")
            lines.append("| Metric | Count | Final Value |")
            lines.append("|--------|-------|-------------|")

            for metric_name, entries in experiment.metrics.items():
                count = len(entries)
                final_value = entries[-1]["value"] if entries else "N/A"
                lines.append(f"| {metric_name} | {count} | {final_value} |")

            lines.append("")

        # Artifacts
        if experiment.artifacts:
            lines.append("## Artifacts\n")
            for name, path in experiment.artifacts.items():
                lines.append(f"- **{name}:** `{path}`")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def generate_html_report(self, experiment: Any) -> str:
        """
        Generate HTML report.

        Args:
            experiment: Experiment instance

        Returns:
            HTML string
        """
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Experiment Report: {experiment.name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "pre { background: #f4f4f4; padding: 15px; border-radius: 5px; }",
            ".metric-card { background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Experiment Report: {experiment.name}</h1>",
            f"<p><strong>ID:</strong> {experiment.id}</p>",
            f"<p><strong>Status:</strong> {experiment.status}</p>",
            f"<p><strong>Duration:</strong> {experiment.get_duration():.2f} seconds</p>",
        ]

        # Configuration
        html.append("<h2>Configuration</h2>")
        html.append("<pre>")
        html.append(json.dumps(experiment.config, indent=2))
        html.append("</pre>")

        # Best Result
        if experiment.best_result:
            html.append("<h2>Best Result</h2>")
            html.append("<table>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            for key, value in experiment.best_result.items():
                html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            html.append("</table>")

        # Metrics
        if experiment.metrics:
            html.append("<h2>Metrics</h2>")
            for metric_name, entries in experiment.metrics.items():
                html.append("<div class='metric-card'>")
                html.append(f"<h3>{metric_name}</h3>")
                html.append(f"<p>Total entries: {len(entries)}</p>")
                if entries:
                    html.append(f"<p>Final value: {entries[-1]['value']}</p>")
                html.append("</div>")

        # Footer
        html.append("<hr>")
        html.append(f"<p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")
        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)

    def generate_comparison_report(
        self, experiments: List[Any], metric: str = "best_fitness"
    ) -> str:
        """
        Generate comparison report for multiple experiments.

        Args:
            experiments: List of experiments
            metric: Metric to compare

        Returns:
            Markdown string
        """
        lines = []

        lines.append("# Experiment Comparison\n")
        lines.append(f"**Metric:** {metric}\n")
        lines.append(f"**Experiments:** {len(experiments)}\n")

        # Table
        lines.append("| Experiment | Status | Duration (s) | Best Value |")
        lines.append("|------------|--------|--------------|------------|")

        for exp in experiments:
            best_value = "N/A"
            if exp.best_result and metric in exp.best_result:
                best_value = f"{exp.best_result[metric]:.4f}"

            lines.append(
                f"| {exp.name} | {exp.status} | " f"{exp.get_duration():.2f} | {best_value} |"
            )

        lines.append("")

        # Rankings
        ranked = sorted(
            [e for e in experiments if e.best_result and metric in e.best_result],
            key=lambda x: x.best_result[metric],
            reverse=True,
        )

        if ranked:
            lines.append("## Rankings\n")
            for i, exp in enumerate(ranked, 1):
                value = exp.best_result[metric]
                lines.append(f"{i}. **{exp.name}** - {value:.4f}")
            lines.append("")

        return "\n".join(lines)

    def save_report(self, content: str, filepath: str) -> None:
        """Save report to file."""
        with open(filepath, "w") as f:
            f.write(content)

        logger.info(f"Report saved to {filepath}")

    def generate_latex_report(self, experiment: Any) -> str:
        """
        Generate LaTeX report.

        Args:
            experiment: Experiment instance

        Returns:
            LaTeX string
        """
        lines = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
            "\\begin{document}",
            "",
            f"\\title{{Experiment Report: {experiment.name}}}",
            "\\maketitle",
            "",
            "\\section{Overview}",
            "",
            f"\\textbf{{Experiment ID:}} \\texttt{{{experiment.id}}}",
            "",
            f"\\textbf{{Status:}} {experiment.status}",
            "",
            f"\\textbf{{Duration:}} {experiment.get_duration():.2f} seconds",
            "",
        ]

        # Best Result
        if experiment.best_result:
            lines.append("\\section{Best Result}")
            lines.append("\\begin{table}[h]")
            lines.append("\\centering")
            lines.append("\\begin{tabular}{ll}")
            lines.append("\\toprule")
            lines.append("Metric & Value \\\\")
            lines.append("\\midrule")

            for key, value in experiment.best_result.items():
                lines.append(f"{key} & {value} \\\\")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")

        lines.append("\\end{document}")

        return "\n".join(lines)


class ProgressReporter:
    """Report progress during optimization."""

    def __init__(self, report_every: int = 10):
        """
        Initialize progress reporter.

        Args:
            report_every: Report every N steps
        """
        self.report_every = report_every
        self.step = 0

    def report(self, metrics: Dict[str, Any]) -> None:
        """Report progress."""
        self.step += 1

        if self.step % self.report_every == 0:
            self._print_progress(metrics)

    def _print_progress(self, metrics: Dict[str, Any]) -> None:
        """Print progress to console."""
        print(f"\n[Step {self.step}]")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


class EmailReporter:
    """Send email reports (requires SMTP configuration)."""

    def __init__(self, smtp_server: str, smtp_port: int, sender: str, password: str):
        """Initialize email reporter."""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password

    def send_report(self, recipient: str, subject: str, body: str) -> None:
        """Send email report."""
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.sender
            msg["To"] = recipient

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)

            logger.info(f"Email report sent to {recipient}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")


class SlackReporter:
    """Send reports to Slack."""

    def __init__(self, webhook_url: str):
        """Initialize Slack reporter."""
        self.webhook_url = webhook_url

    def send_message(self, text: str) -> None:
        """Send message to Slack."""
        try:
            import requests

            payload = {"text": text}
            response = requests.post(self.webhook_url, json=payload)

            if response.status_code == 200:
                logger.info("Slack message sent")
            else:
                logger.error(f"Slack message failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")

    def send_experiment_report(self, experiment: Any) -> None:
        """Send experiment report to Slack."""
        text = f"*Experiment Complete:* {experiment.name}\n"
        text += f"Status: {experiment.status}\n"
        text += f"Duration: {experiment.get_duration():.2f}s\n"

        if experiment.best_result:
            text += "\n*Best Result:*\n"
            for key, value in list(experiment.best_result.items())[:5]:
                text += f"• {key}: {value}\n"

        self.send_message(text)
