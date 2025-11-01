"""Tests for CLI interface."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from morphml.cli.main import cli


class TestCLICommands:
    """Test CLI command interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version_command(self) -> None:
        """Test version command."""
        result = self.runner.invoke(cli, ['version'])
        
        assert result.exit_code == 0
        assert 'MorphML' in result.output
        assert 'Eshan Roy' in result.output
        assert 'TONMOY INFRASTRUCTURE' in result.output

    def test_config_command(self) -> None:
        """Test config command."""
        result = self.runner.invoke(cli, ['config'])
        
        assert result.exit_code == 0
        assert 'Configuration' in result.output

    def test_config_with_key(self) -> None:
        """Test config command with specific key."""
        result = self.runner.invoke(cli, ['config', '--key', 'version'])
        
        assert result.exit_code == 0

    def test_run_command_with_valid_experiment(self) -> None:
        """Test run command with valid experiment file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment file
            experiment_file = Path(tmpdir) / "experiment.py"
            experiment_file.write_text("""
from morphml.core.dsl import Layer, SearchSpace

search_space = SearchSpace("test")
search_space.add_layers(
    Layer.input(shape=(784,)),
    Layer.dense(units=[64, 128]),
    Layer.relu(),
    Layer.output(units=10)
)

optimizer_config = {
    'population_size': 5,
    'num_generations': 2
}

max_evaluations = 10
""")
            
            output_dir = Path(tmpdir) / "results"
            
            # Run CLI
            result = self.runner.invoke(cli, [
                'run',
                str(experiment_file),
                '--output-dir', str(output_dir)
            ])
            
            # Should complete successfully
            assert result.exit_code == 0
            
            # Check output files were created
            assert output_dir.exists()
            assert (output_dir / "best_model.json").exists()
            assert (output_dir / "summary.json").exists()

    def test_run_command_missing_search_space(self) -> None:
        """Test run command with invalid experiment file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid experiment file
            experiment_file = Path(tmpdir) / "bad_experiment.py"
            experiment_file.write_text("""
# Missing search_space definition
optimizer_config = {'population_size': 10}
""")
            
            # Run CLI
            result = self.runner.invoke(cli, [
                'run',
                str(experiment_file)
            ])
            
            # Should fail
            assert result.exit_code != 0
            assert 'search_space' in result.output

    def test_status_command(self) -> None:
        """Test status command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock results
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()
            
            summary = {
                'best_fitness': 0.85,
                'final_generation': 10,
                'population_size': 20
            }
            
            summary_file = results_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f)
            
            # Run status command
            result = self.runner.invoke(cli, ['status', str(results_dir)])
            
            assert result.exit_code == 0
            assert '0.85' in result.output or '0.850000' in result.output

    def test_status_command_missing_results(self) -> None:
        """Test status command with missing results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run status on empty directory
            result = self.runner.invoke(cli, ['status', str(tmpdir)])
            
            assert result.exit_code != 0
            assert 'No results found' in result.output

    def test_export_command(self) -> None:
        """Test export command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock architecture
            from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
            
            graph = ModelGraph()
            input_node = GraphNode.create("input", {"shape": (784,)})
            dense_node = GraphNode.create("dense", {"units": 128})
            output_node = GraphNode.create("dense", {"units": 10})
            
            graph.add_node(input_node)
            graph.add_node(dense_node)
            graph.add_node(output_node)
            graph.add_edge(GraphEdge(input_node, dense_node))
            graph.add_edge(GraphEdge(dense_node, output_node))
            
            # Save architecture
            arch_file = Path(tmpdir) / "architecture.json"
            with open(arch_file, 'w') as f:
                json.dump(graph.to_dict(), f)
            
            output_file = Path(tmpdir) / "exported_model.py"
            
            # Run export command
            result = self.runner.invoke(cli, [
                'export',
                str(arch_file),
                '--format', 'pytorch',
                '--output', str(output_file)
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
            
            # Check exported code
            with open(output_file, 'r') as f:
                code = f.read()
            
            assert 'import torch' in code
            assert 'class ExportedModel' in code

    def test_export_command_keras(self) -> None:
        """Test export command with Keras format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from morphml.core.graph import GraphEdge, GraphNode, ModelGraph
            
            graph = ModelGraph()
            input_node = GraphNode.create("input", {"shape": (784,)})
            output_node = GraphNode.create("dense", {"units": 10})
            
            graph.add_node(input_node)
            graph.add_node(output_node)
            graph.add_edge(GraphEdge(input_node, output_node))
            
            arch_file = Path(tmpdir) / "architecture.json"
            with open(arch_file, 'w') as f:
                json.dump(graph.to_dict(), f)
            
            output_file = Path(tmpdir) / "exported_keras.py"
            
            result = self.runner.invoke(cli, [
                'export',
                str(arch_file),
                '--format', 'keras',
                '--output', str(output_file)
            ])
            
            assert result.exit_code == 0

    def test_run_with_verbose_flag(self) -> None:
        """Test run command with verbose flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_file = Path(tmpdir) / "experiment.py"
            experiment_file.write_text("""
from morphml.core.dsl import Layer, SearchSpace

search_space = SearchSpace("verbose_test")
search_space.add_layers(
    Layer.input(shape=(784,)),
    Layer.dense(units=64),
    Layer.output(units=10)
)

optimizer_config = {'population_size': 3, 'num_generations': 2}
""")
            
            output_dir = Path(tmpdir) / "results"
            
            result = self.runner.invoke(cli, [
                'run',
                str(experiment_file),
                '--output-dir', str(output_dir),
                '--verbose'
            ])
            
            # Should include more detailed output with verbose
            assert result.exit_code == 0

    def test_run_with_export_format(self) -> None:
        """Test run command with different export formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_file = Path(tmpdir) / "experiment.py"
            experiment_file.write_text("""
from morphml.core.dsl import Layer, SearchSpace

search_space = SearchSpace("export_test")
search_space.add_layers(
    Layer.input(shape=(784,)),
    Layer.dense(units=64),
    Layer.output(units=10)
)

optimizer_config = {'population_size': 3, 'num_generations': 2}
""")
            
            output_dir = Path(tmpdir) / "results"
            
            # Test with pytorch only
            result = self.runner.invoke(cli, [
                'run',
                str(experiment_file),
                '--output-dir', str(output_dir),
                '--export-format', 'pytorch'
            ])
            
            assert result.exit_code == 0
            assert (output_dir / "best_model_pytorch.py").exists()


class TestCLIEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_run_nonexistent_file(self) -> None:
        """Test run with nonexistent experiment file."""
        result = self.runner.invoke(cli, [
            'run',
            '/nonexistent/path/experiment.py'
        ])
        
        assert result.exit_code != 0

    def test_status_nonexistent_directory(self) -> None:
        """Test status with nonexistent directory."""
        result = self.runner.invoke(cli, [
            'status',
            '/nonexistent/path'
        ])
        
        assert result.exit_code != 0

    def test_export_nonexistent_file(self) -> None:
        """Test export with nonexistent architecture file."""
        result = self.runner.invoke(cli, [
            'export',
            '/nonexistent/architecture.json'
        ])
        
        assert result.exit_code != 0

    def test_help_messages(self) -> None:
        """Test help messages are displayed."""
        # Main help
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'MorphML' in result.output
        
        # Run command help
        result = self.runner.invoke(cli, ['run', '--help'])
        assert result.exit_code == 0
        assert 'experiment' in result.output.lower()
        
        # Status command help
        result = self.runner.invoke(cli, ['status', '--help'])
        assert result.exit_code == 0
        
        # Config command help
        result = self.runner.invoke(cli, ['config', '--help'])
        assert result.exit_code == 0
        
        # Export command help
        result = self.runner.invoke(cli, ['export', '--help'])
        assert result.exit_code == 0

    def test_version_option(self) -> None:
        """Test --version option."""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        # Version should be displayed


class TestCLIWorkflows:
    """Test complete CLI workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_complete_workflow(self) -> None:
        """Test complete workflow: run -> status -> export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Step 1: Create and run experiment
            experiment_file = tmpdir / "experiment.py"
            experiment_file.write_text("""
from morphml.core.dsl import Layer, SearchSpace

search_space = SearchSpace("workflow_test")
search_space.add_layers(
    Layer.input(shape=(3, 32, 32)),
    Layer.conv2d(filters=[32, 64], kernel_size=3),
    Layer.relu(),
    Layer.maxpool(pool_size=2),
    Layer.flatten(),
    Layer.dense(units=[128, 256]),
    Layer.output(units=10)
)

optimizer_config = {
    'population_size': 5,
    'num_generations': 3
}
""")
            
            output_dir = tmpdir / "results"
            
            # Run experiment
            result = self.runner.invoke(cli, [
                'run',
                str(experiment_file),
                '--output-dir', str(output_dir)
            ])
            
            assert result.exit_code == 0
            
            # Step 2: Check status
            result = self.runner.invoke(cli, ['status', str(output_dir)])
            assert result.exit_code == 0
            
            # Step 3: Export architecture
            arch_file = output_dir / "best_model.json"
            export_file = tmpdir / "exported.py"
            
            result = self.runner.invoke(cli, [
                'export',
                str(arch_file),
                '--format', 'pytorch',
                '--output', str(export_file)
            ])
            
            assert result.exit_code == 0
            assert export_file.exists()

    def test_checkpoint_workflow(self) -> None:
        """Test workflow with checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            experiment_file = tmpdir / "experiment.py"
            experiment_file.write_text("""
from morphml.core.dsl import Layer, SearchSpace

search_space = SearchSpace("checkpoint_test")
search_space.add_layers(
    Layer.input(shape=(784,)),
    Layer.dense(units=[64, 128]),
    Layer.relu(),
    Layer.output(units=10)
)

optimizer_config = {
    'population_size': 5,
    'num_generations': 10
}
""")
            
            output_dir = tmpdir / "results"
            checkpoint_dir = tmpdir / "checkpoints"
            
            # Run with checkpointing
            result = self.runner.invoke(cli, [
                'run',
                str(experiment_file),
                '--output-dir', str(output_dir),
                '--checkpoint-dir', str(checkpoint_dir)
            ])
            
            assert result.exit_code == 0


def test_cli_integration() -> None:
    """Integration test for CLI."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create comprehensive experiment
        experiment_file = tmpdir / "integration.py"
        experiment_file.write_text("""
from morphml.core.dsl import Layer, SearchSpace
from morphml.constraints import MaxParametersConstraint

search_space = SearchSpace("cli_integration")

search_space.add_layers(
    Layer.input(shape=(3, 32, 32)),
    Layer.conv2d(filters=[32, 64], kernel_size=[3, 5]),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2),
    Layer.conv2d(filters=[64, 128], kernel_size=3),
    Layer.relu(),
    Layer.flatten(),
    Layer.dense(units=[256, 512]),
    Layer.dropout(rate=[0.3, 0.5]),
    Layer.output(units=10)
)

search_space.add_constraint(MaxParametersConstraint(max_params=1000000))

optimizer_config = {
    'population_size': 10,
    'num_generations': 5,
    'elite_size': 2,
    'mutation_rate': 0.15,
    'crossover_rate': 0.7
}
""")
        
        output_dir = tmpdir / "results"
        
        # Run experiment
        result = runner.invoke(cli, [
            'run',
            str(experiment_file),
            '--output-dir', str(output_dir),
            '--export-format', 'both',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        
        # Verify all outputs
        assert (output_dir / "best_model.json").exists()
        assert (output_dir / "best_model_pytorch.py").exists()
        assert (output_dir / "best_model_keras.py").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "history.json").exists()
        
        print("\n✓ CLI integration test passed!")
