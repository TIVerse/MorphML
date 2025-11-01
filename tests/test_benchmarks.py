"""Tests for benchmark system."""

import pytest
import tempfile
import os

from morphml.benchmarks import (
    BenchmarkSuite,
    OptimizerComparator,
    SimpleProblem,
    ComplexProblem,
    MultiModalProblem,
    ConstrainedProblem,
)
from morphml.benchmarks.problems import (
    NoisyProblem,
    RastriginProblem,
    RosenbrockProblem,
    get_all_problems,
)
from morphml.optimizers import RandomSearch, GeneticAlgorithm, HillClimbing


class TestBenchmarkProblems:
    """Test benchmark problems."""

    def test_simple_problem(self) -> None:
        """Test SimpleProblem."""
        problem = SimpleProblem()
        
        assert problem.name == "SimpleProblem"
        assert problem.search_space is not None
        
        # Sample and evaluate
        arch = problem.search_space.sample()
        fitness = problem.evaluate(arch)
        
        assert 0.0 <= fitness <= 1.0

    def test_complex_problem(self) -> None:
        """Test ComplexProblem."""
        problem = ComplexProblem()
        
        assert problem.name == "ComplexProblem"
        
        arch = problem.search_space.sample()
        fitness = problem.evaluate(arch)
        
        assert 0.0 <= fitness <= 1.0

    def test_multimodal_problem(self) -> None:
        """Test MultiModalProblem."""
        problem = MultiModalProblem()
        
        assert problem.name == "MultiModalProblem"
        
        # Evaluate multiple architectures
        fitnesses = []
        for _ in range(10):
            arch = problem.search_space.sample()
            fitness = problem.evaluate(arch)
            fitnesses.append(fitness)
            assert 0.0 <= fitness <= 1.0
        
        # Should have some variation
        assert len(set(fitnesses)) > 1

    def test_constrained_problem(self) -> None:
        """Test ConstrainedProblem."""
        problem = ConstrainedProblem()
        
        assert problem.name == "ConstrainedProblem"
        
        arch = problem.search_space.sample()
        fitness = problem.evaluate(arch)
        
        assert 0.0 <= fitness <= 1.0

    def test_noisy_problem(self) -> None:
        """Test NoisyProblem."""
        problem = NoisyProblem(noise_level=0.1)
        
        arch = problem.search_space.sample()
        
        # Multiple evaluations should give different results
        fitness1 = problem.evaluate(arch)
        fitness2 = problem.evaluate(arch)
        
        # May be different due to noise
        assert 0.0 <= fitness1 <= 1.0
        assert 0.0 <= fitness2 <= 1.0

    def test_rastrigin_problem(self) -> None:
        """Test RastriginProblem."""
        problem = RastriginProblem()
        
        arch = problem.search_space.sample()
        fitness = problem.evaluate(arch)
        
        assert 0.0 <= fitness <= 1.0

    def test_rosenbrock_problem(self) -> None:
        """Test RosenbrockProblem."""
        problem = RosenbrockProblem()
        
        arch = problem.search_space.sample()
        fitness = problem.evaluate(arch)
        
        assert 0.0 <= fitness <= 1.0

    def test_get_all_problems(self) -> None:
        """Test getting all problems."""
        problems = get_all_problems()
        
        assert len(problems) > 0
        assert all(hasattr(p, 'evaluate') for p in problems)
        assert all(hasattr(p, 'search_space') for p in problems)


class TestBenchmarkSuite:
    """Test BenchmarkSuite."""

    def test_suite_creation(self) -> None:
        """Test creating benchmark suite."""
        suite = BenchmarkSuite()
        
        assert suite is not None
        assert len(suite.optimizers) == 0

    def test_add_optimizer(self) -> None:
        """Test adding optimizers."""
        suite = BenchmarkSuite()
        
        suite.add_optimizer(
            "RS",
            RandomSearch,
            {'num_samples': 10}
        )
        
        assert "RS" in suite.optimizers

    def test_run_single_benchmark(self) -> None:
        """Test running single benchmark."""
        suite = BenchmarkSuite()
        suite.add_optimizer("RS", RandomSearch, {'num_samples': 5})
        
        problem = SimpleProblem()
        
        result = suite.run_single(
            problem,
            "RS",
            RandomSearch,
            {'num_samples': 5}
        )
        
        assert result is not None
        assert result.optimizer_name == "RS"
        assert result.problem_name == "SimpleProblem"
        assert result.best_fitness >= 0.0

    def test_run_suite(self) -> None:
        """Test running full suite."""
        suite = BenchmarkSuite()
        suite.add_optimizer("RS", RandomSearch, {'num_samples': 5})
        suite.add_optimizer("HC", HillClimbing, {'max_iterations': 5})
        
        problems = [SimpleProblem()]
        
        results = suite.run(problems, num_runs=2)
        
        # Should have 2 optimizers * 1 problem * 2 runs = 4 results
        assert len(results) == 4

    def test_get_summary(self) -> None:
        """Test getting summary."""
        suite = BenchmarkSuite()
        suite.add_optimizer("RS", RandomSearch, {'num_samples': 5})
        
        problems = [SimpleProblem()]
        results = suite.run(problems, num_runs=2)
        
        summary = suite.get_summary(results)
        
        assert len(summary) > 0
        assert any('mean_best_fitness' in data for data in summary.values())

    def test_get_winner(self) -> None:
        """Test getting winner."""
        suite = BenchmarkSuite()
        suite.add_optimizer("RS", RandomSearch, {'num_samples': 10})
        suite.add_optimizer("HC", HillClimbing, {'max_iterations': 10})
        
        problems = [SimpleProblem()]
        results = suite.run(problems, num_runs=2)
        
        winner = suite.get_winner("SimpleProblem", results)
        
        assert winner in ["RS", "HC"]

    def test_export_results(self) -> None:
        """Test exporting results."""
        suite = BenchmarkSuite()
        suite.add_optimizer("RS", RandomSearch, {'num_samples': 5})
        
        problems = [SimpleProblem()]
        suite.run(problems, num_runs=2)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            suite.export_results(output_path)
            
            assert os.path.exists(output_path)
            
            # Verify valid JSON
            import json
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'results' in data
            assert 'summary' in data
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestOptimizerComparator:
    """Test OptimizerComparator."""

    def test_comparator_creation(self) -> None:
        """Test creating comparator."""
        comparator = OptimizerComparator()
        
        assert comparator is not None
        assert len(comparator.results) == 0

    def test_add_result(self) -> None:
        """Test adding results."""
        comparator = OptimizerComparator()
        
        comparator.add_result("GA", "Problem1", [0.9, 0.92, 0.91])
        
        assert ("GA", "Problem1") in comparator.results

    def test_get_statistics(self) -> None:
        """Test getting statistics."""
        comparator = OptimizerComparator()
        comparator.add_result("GA", "Problem1", [0.9, 0.92, 0.91, 0.89, 0.93])
        
        stats = comparator.get_statistics("GA", "Problem1")
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert stats['count'] == 5

    def test_rank_optimizers(self) -> None:
        """Test ranking optimizers."""
        comparator = OptimizerComparator()
        comparator.add_result("GA", "Problem1", [0.9, 0.92, 0.91])
        comparator.add_result("RS", "Problem1", [0.85, 0.87, 0.86])
        comparator.add_result("HC", "Problem1", [0.88, 0.89, 0.87])
        
        rankings = comparator.rank_optimizers("Problem1")
        
        assert len(rankings) == 3
        # GA should be first (highest mean)
        assert rankings[0][0] == "GA"

    def test_compare_pair(self) -> None:
        """Test pairwise comparison."""
        comparator = OptimizerComparator()
        comparator.add_result("GA", "Problem1", [0.9, 0.92, 0.91])
        comparator.add_result("RS", "Problem1", [0.85, 0.87, 0.86])
        
        comparison = comparator.compare_pair("GA", "RS", "Problem1")
        
        assert comparison is not None
        assert 'winner' in comparison
        assert comparison['winner'] == "GA"

    def test_get_dominance_matrix(self) -> None:
        """Test dominance matrix."""
        comparator = OptimizerComparator()
        comparator.add_result("GA", "Problem1", [0.9, 0.92])
        comparator.add_result("RS", "Problem1", [0.85, 0.87])
        comparator.add_result("GA", "Problem2", [0.88, 0.89])
        comparator.add_result("RS", "Problem2", [0.86, 0.87])
        
        dominance = comparator.get_dominance_matrix()
        
        assert len(dominance) > 0
        # GA should dominate RS on both problems
        assert dominance.get(("GA", "RS"), 0) == 2

    def test_get_best_optimizer(self) -> None:
        """Test getting best overall optimizer."""
        comparator = OptimizerComparator()
        comparator.add_result("GA", "Problem1", [0.9, 0.92])
        comparator.add_result("RS", "Problem1", [0.85, 0.87])
        comparator.add_result("GA", "Problem2", [0.88, 0.89])
        comparator.add_result("RS", "Problem2", [0.84, 0.86])
        
        best = comparator.get_best_optimizer()
        
        assert best == "GA"


class TestConvergenceAnalyzer:
    """Test ConvergenceAnalyzer."""

    def test_analyzer_creation(self) -> None:
        """Test creating analyzer."""
        from morphml.benchmarks.comparator import ConvergenceAnalyzer
        
        analyzer = ConvergenceAnalyzer()
        
        assert analyzer is not None

    def test_add_history(self) -> None:
        """Test adding history."""
        from morphml.benchmarks.comparator import ConvergenceAnalyzer
        
        analyzer = ConvergenceAnalyzer()
        history = [0.5, 0.6, 0.7, 0.75, 0.8]
        
        analyzer.add_history("GA", history)
        
        assert "GA" in analyzer.histories

    def test_get_mean_convergence(self) -> None:
        """Test getting mean convergence."""
        from morphml.benchmarks.comparator import ConvergenceAnalyzer
        
        analyzer = ConvergenceAnalyzer()
        analyzer.add_history("GA", [0.5, 0.6, 0.7])
        analyzer.add_history("GA", [0.6, 0.7, 0.8])
        
        mean_curve = analyzer.get_mean_convergence("GA")
        
        assert len(mean_curve) == 3
        assert mean_curve[0] == 0.55  # (0.5 + 0.6) / 2

    def test_calculate_auc(self) -> None:
        """Test AUC calculation."""
        from morphml.benchmarks.comparator import ConvergenceAnalyzer
        
        analyzer = ConvergenceAnalyzer()
        analyzer.add_history("GA", [0.5, 0.6, 0.7, 0.8])
        
        auc = analyzer.calculate_auc("GA")
        
        assert auc > 0.0


def test_benchmark_integration() -> None:
    """Integration test for benchmark system."""
    # Create suite
    suite = BenchmarkSuite()
    
    # Add optimizers
    suite.add_optimizer("RS", RandomSearch, {'num_samples': 10})
    suite.add_optimizer("HC", HillClimbing, {'max_iterations': 10})
    
    # Run on multiple problems
    problems = [SimpleProblem(), MultiModalProblem()]
    results = suite.run(problems, num_runs=2)
    
    # Verify results
    assert len(results) == 8  # 2 optimizers * 2 problems * 2 runs
    
    # Get summary
    summary = suite.get_summary(results)
    assert len(summary) > 0
    
    # Export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        suite.export_results(output_path)
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_comparator_workflow() -> None:
    """Test comparator workflow."""
    # Create comparator
    comparator = OptimizerComparator()
    
    # Add results from multiple runs
    comparator.add_result("GA", "Problem1", [0.90, 0.92, 0.91, 0.93, 0.89])
    comparator.add_result("RS", "Problem1", [0.85, 0.87, 0.86, 0.88, 0.84])
    comparator.add_result("HC", "Problem1", [0.88, 0.89, 0.87, 0.90, 0.86])
    
    comparator.add_result("GA", "Problem2", [0.85, 0.87, 0.86, 0.88, 0.84])
    comparator.add_result("RS", "Problem2", [0.80, 0.82, 0.81, 0.83, 0.79])
    comparator.add_result("HC", "Problem2", [0.83, 0.84, 0.82, 0.85, 0.81])
    
    # Rank optimizers
    rankings1 = comparator.rank_optimizers("Problem1")
    rankings2 = comparator.rank_optimizers("Problem2")
    
    assert len(rankings1) == 3
    assert len(rankings2) == 3
    
    # Get best overall
    best = comparator.get_best_optimizer()
    assert best in ["GA", "RS", "HC"]
    
    # Dominance matrix
    dominance = comparator.get_dominance_matrix()
    assert len(dominance) > 0


def test_full_benchmark_pipeline() -> None:
    """Test complete benchmark pipeline."""
    # Setup
    suite = BenchmarkSuite()
    suite.add_optimizer("RandomSearch", RandomSearch, {'num_samples': 15})
    suite.add_optimizer("HillClimbing", HillClimbing, {'max_iterations': 15, 'patience': 5})
    suite.add_optimizer("GeneticAlgorithm", GeneticAlgorithm, {'population_size': 10, 'num_generations': 3})
    
    # Run benchmarks
    problems = [SimpleProblem(), MultiModalProblem()]
    results = suite.run(problems, num_runs=3)
    
    # Verify all completed
    assert len(results) == 18  # 3 optimizers * 2 problems * 3 runs
    
    # Analyze results
    summary = suite.get_summary(results)
    assert len(summary) == 6  # 3 optimizers * 2 problems
    
    # Get winners
    winner1 = suite.get_winner("SimpleProblem", results)
    winner2 = suite.get_winner("MultiModalProblem", results)
    
    assert winner1 is not None
    assert winner2 is not None
