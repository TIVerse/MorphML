#!/usr/bin/env python3
"""Run all tests and generate comprehensive test report.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class TestRunner:
    """Comprehensive test runner."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_pytest(self, test_path: str, test_name: str, markers: str = "") -> Dict:
        """Run pytest on specific path."""
        console.print(f"\n[cyan]Running {test_name}...[/cyan]")
        
        cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
        if markers:
            cmd.extend(["-m", markers])
        
        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Count tests
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        skipped = output.count(" SKIPPED")
        errors = output.count(" ERROR")
        
        success = result.returncode == 0
        
        return {
            "name": test_name,
            "success": success,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "elapsed": elapsed,
            "output": output
        }
    
    def run_benchmark(self) -> Dict:
        """Run benchmark suite."""
        console.print("\n[cyan]Running benchmark suite...[/cyan]")
        
        benchmark_script = self.project_root / "benchmarks" / "run_benchmarks.py"
        
        if not benchmark_script.exists():
            return {
                "name": "Benchmarks",
                "success": False,
                "error": "Benchmark script not found"
            }
        
        start = time.time()
        result = subprocess.run(
            ["python", str(benchmark_script)],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        elapsed = time.time() - start
        
        return {
            "name": "Benchmarks",
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "output": result.stdout
        }
    
    def check_code_quality(self) -> Dict:
        """Run code quality checks."""
        console.print("\n[cyan]Checking code quality...[/cyan]")
        
        checks = {}
        
        # Ruff check
        console.print("  - Running ruff...")
        result = subprocess.run(
            ["python", "-m", "ruff", "check", "morphml/"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        checks["ruff"] = {
            "passed": result.returncode == 0,
            "output": result.stdout
        }
        
        # MyPy check (optional)
        console.print("  - Running mypy...")
        result = subprocess.run(
            ["python", "-m", "mypy", "morphml/", "--ignore-missing-imports"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        checks["mypy"] = {
            "passed": result.returncode == 0,
            "output": result.stdout
        }
        
        return {
            "name": "Code Quality",
            "success": all(c["passed"] for c in checks.values()),
            "checks": checks
        }
    
    def run_all(self) -> Dict:
        """Run all tests and checks."""
        self.start_time = time.time()
        
        console.print(Panel.fit(
            "[bold cyan]MorphML Test Suite[/bold cyan]\n"
            "Running comprehensive tests and benchmarks",
            border_style="cyan"
        ))
        
        # Unit tests
        self.results["unit_tests"] = self.run_pytest(
            "tests/",
            "Unit Tests",
            "not slow and not integration"
        )
        
        # Integration tests
        self.results["integration_tests"] = self.run_pytest(
            "tests/test_distributed/",
            "Integration Tests",
            "integration"
        )
        
        # Performance tests
        self.results["performance_tests"] = self.run_pytest(
            "tests/test_performance.py",
            "Performance Tests"
        )
        
        # Helm validation
        self.results["helm_validation"] = self.run_pytest(
            "tests/test_helm_validation.py",
            "Helm Validation"
        )
        
        # Code quality
        self.results["code_quality"] = self.check_code_quality()
        
        # Benchmarks (optional, can take time)
        try:
            self.results["benchmarks"] = self.run_benchmark()
        except Exception as e:
            self.results["benchmarks"] = {
                "name": "Benchmarks",
                "success": False,
                "error": str(e)
            }
        
        self.end_time = time.time()
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive report."""
        total_elapsed = self.end_time - self.start_time if self.end_time else 0
        
        # Calculate totals
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        
        all_success = all(r.get("success", False) for r in self.results.values())
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed": total_elapsed,
            "summary": {
                "all_success": all_success,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_skipped": total_skipped,
                "total_errors": total_errors,
                "total_tests": total_passed + total_failed + total_skipped
            },
            "results": self.results
        }
        
        return report
    
    def print_summary(self, report: Dict):
        """Print test summary."""
        console.print("\n")
        console.print("="*70)
        console.print("[bold cyan]Test Summary[/bold cyan]")
        console.print("="*70)
        
        # Summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test Suite")
        table.add_column("Status", justify="center")
        table.add_column("Passed", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Skipped", justify="right")
        table.add_column("Time (s)", justify="right")
        
        for name, result in self.results.items():
            status = "✅" if result.get("success", False) else "❌"
            passed = result.get("passed", 0)
            failed = result.get("failed", 0)
            skipped = result.get("skipped", 0)
            elapsed = result.get("elapsed", 0)
            
            table.add_row(
                result.get("name", name),
                status,
                str(passed) if passed > 0 else "-",
                str(failed) if failed > 0 else "-",
                str(skipped) if skipped > 0 else "-",
                f"{elapsed:.2f}"
            )
        
        console.print(table)
        
        # Overall summary
        summary = report["summary"]
        console.print("\n[bold]Overall Summary:[/bold]")
        console.print(f"  Total Tests: {summary['total_tests']}")
        console.print(f"  [green]Passed: {summary['total_passed']}[/green]")
        if summary['total_failed'] > 0:
            console.print(f"  [red]Failed: {summary['total_failed']}[/red]")
        if summary['total_skipped'] > 0:
            console.print(f"  [yellow]Skipped: {summary['total_skipped']}[/yellow]")
        if summary['total_errors'] > 0:
            console.print(f"  [red]Errors: {summary['total_errors']}[/red]")
        console.print(f"  Total Time: {report['total_elapsed']:.2f}s")
        
        # Final verdict
        if summary['all_success']:
            console.print("\n[bold green]✅ All tests passed![/bold green]\n")
        else:
            console.print("\n[bold red]❌ Some tests failed[/bold red]\n")
    
    def save_report(self, report: Dict, filename: str = "test_report.json"):
        """Save report to file."""
        report_path = self.project_root / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"[green]Report saved to: {report_path}[/green]")


def main():
    """Main test runner."""
    project_root = Path(__file__).parent.parent
    
    runner = TestRunner(project_root)
    
    try:
        results = runner.run_all()
        report = runner.generate_report()
        runner.print_summary(report)
        runner.save_report(report)
        
        # Exit with error code if any tests failed
        if not report["summary"]["all_success"]:
            sys.exit(1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error running tests: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
