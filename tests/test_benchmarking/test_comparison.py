"""Tests for benchmarking and comparison tools.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.benchmarking import OptimizerComparison
from morphml.core.dsl import create_cnn_space


class TestOptimizerComparison:
    """Test optimizer comparison framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        
        def evaluator(graph):
            return np.random.rand()
        
        self.evaluator = evaluator
    
    def test_initialization(self):
        """Test comparison initialization."""
        comparison = OptimizerComparison(
            self.search_space,
            self.evaluator,
            budget=10,
            num_runs=2
        )
        
        assert comparison.budget == 10
        assert comparison.num_runs == 2
        assert len(comparison.optimizers) == 0
    
    def test_add_optimizer(self):
        """Test adding optimizers."""
        from morphml.optimizers.evolutionary import ParticleSwarmOptimizer
        
        comparison = OptimizerComparison(
            self.search_space,
            self.evaluator,
            budget=10,
            num_runs=2
        )
        
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            {'num_particles': 5, 'max_iterations': 2}
        )
        
        comparison.add_optimizer('PSO', optimizer)
        
        assert 'PSO' in comparison.optimizers
    
    def test_run_comparison(self):
        """Test running comparison."""
        from morphml.optimizers.evolutionary import ParticleSwarmOptimizer
        
        comparison = OptimizerComparison(
            self.search_space,
            self.evaluator,
            budget=10,
            num_runs=2
        )
        
        # Add a simple optimizer
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            {'num_particles': 3, 'max_iterations': 2}
        )
        comparison.add_optimizer('PSO', optimizer)
        
        # Run comparison
        results = comparison.run()
        
        assert 'PSO' in results
        assert 'mean_best' in results['PSO']
        assert 'std_best' in results['PSO']
        assert results['PSO']['num_successful_runs'] == 2
    
    def test_statistics_computation(self):
        """Test statistics computation."""
        comparison = OptimizerComparison(
            self.search_space,
            self.evaluator,
            budget=10,
            num_runs=3
        )
        
        # Mock results
        comparison.results = {
            'Optimizer1': [
                {'run_id': 0, 'best_fitness': 0.8, 'success': True, 'time_seconds': 1.0},
                {'run_id': 1, 'best_fitness': 0.9, 'success': True, 'time_seconds': 1.1},
                {'run_id': 2, 'best_fitness': 0.85, 'success': True, 'time_seconds': 1.2},
            ]
        }
        
        stats = comparison._compute_statistics()
        
        assert 'Optimizer1' in stats
        assert np.isclose(stats['Optimizer1']['mean_best'], 0.85)
        assert stats['Optimizer1']['min_best'] == 0.8
        assert stats['Optimizer1']['max_best'] == 0.9


def test_compare_optimizers_function():
    """Test compare_optimizers convenience function."""
    from morphml.benchmarking import compare_optimizers
    from morphml.optimizers.evolutionary import ParticleSwarmOptimizer
    
    space = create_cnn_space(num_classes=10)
    
    def evaluator(graph):
        return np.random.rand()
    
    optimizer1 = ParticleSwarmOptimizer(
        space,
        {'num_particles': 3, 'max_iterations': 2}
    )
    
    results = compare_optimizers(
        optimizers={'PSO': optimizer1},
        search_space=space,
        evaluator=evaluator,
        budget=10,
        num_runs=2,
        plot=False  # Disable plotting in tests
    )
    
    assert 'PSO' in results
    assert 'mean_best' in results['PSO']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
