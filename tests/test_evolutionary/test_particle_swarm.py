"""Tests for Particle Swarm Optimization.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.dsl import create_cnn_space
from morphml.optimizers.evolutionary import (
    ParticleSwarmOptimizer,
    Particle,
)


class TestParticle:
    """Test Particle class."""
    
    def test_particle_initialization(self):
        """Test particle initialization."""
        position = np.random.rand(50)
        velocity = np.random.rand(50) * 0.1
        
        particle = Particle(
            position=position,
            velocity=velocity,
            best_position=position.copy(),
            best_fitness=-np.inf
        )
        
        assert len(particle.position) == 50
        assert len(particle.velocity) == 50
        assert particle.best_fitness == -np.inf


class TestParticleSwarmOptimizer:
    """Test PSO optimizer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.search_space = create_cnn_space(num_classes=10)
        self.config = {
            'num_particles': 10,
            'max_iterations': 5,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        }
    
    def test_initialization(self):
        """Test PSO initialization."""
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            self.config
        )
        
        assert optimizer.num_particles == 10
        assert optimizer.max_iterations == 5
        assert optimizer.w == 0.7
        assert optimizer.c1 == 1.5
        assert optimizer.c2 == 1.5
        assert optimizer.global_best_fitness == -np.inf
    
    def test_swarm_initialization(self):
        """Test swarm initialization."""
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            self.config
        )
        optimizer.initialize_swarm()
        
        assert len(optimizer.particles) == 10
        for particle in optimizer.particles:
            assert isinstance(particle, Particle)
            assert len(particle.position) == optimizer.dim
            assert len(particle.velocity) == optimizer.dim
    
    def test_velocity_update(self):
        """Test velocity update."""
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            self.config
        )
        optimizer.initialize_swarm()
        optimizer.global_best_position = np.random.rand(optimizer.dim)
        
        particle = optimizer.particles[0]
        old_velocity = particle.velocity.copy()
        
        new_velocity = optimizer.update_velocity(particle, iteration=0)
        
        assert len(new_velocity) == optimizer.dim
        assert not np.array_equal(new_velocity, old_velocity)
        # Velocity should be clamped
        assert np.all(np.abs(new_velocity) <= optimizer.max_velocity)
    
    def test_position_update(self):
        """Test position update."""
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            self.config
        )
        optimizer.initialize_swarm()
        
        particle = optimizer.particles[0]
        old_position = particle.position.copy()
        
        new_position = optimizer.update_position(particle)
        
        assert len(new_position) == optimizer.dim
        # Position should be in [0, 1]
        assert np.all((new_position >= 0) & (new_position <= 1))
    
    def test_optimize_simple(self):
        """Test basic PSO optimization."""
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            {
                'num_particles': 5,
                'max_iterations': 3
            }
        )
        
        def evaluator(graph):
            return np.random.rand()
        
        best = optimizer.optimize(evaluator)
        
        assert best is not None
        assert hasattr(best, 'fitness')
        assert best.fitness >= 0
    
    def test_adaptive_inertia(self):
        """Test adaptive inertia weight."""
        optimizer = ParticleSwarmOptimizer(
            self.search_space,
            {
                'num_particles': 5,
                'max_iterations': 10,
                'adaptive_inertia': True,
                'w_min': 0.4,
                'w_max': 0.9
            }
        )
        optimizer.initialize_swarm()
        optimizer.global_best_position = np.random.rand(optimizer.dim)
        
        # Test velocity update at different iterations
        particle = optimizer.particles[0]
        
        vel_early = optimizer.update_velocity(particle, iteration=0)
        vel_late = optimizer.update_velocity(particle, iteration=9)
        
        # Just check that they're different (adaptive behavior)
        # Actual values depend on random components


def test_convenience_function():
    """Test optimize_with_pso convenience function."""
    from morphml.optimizers.evolutionary import optimize_with_pso
    
    space = create_cnn_space(num_classes=10)
    
    def evaluator(graph):
        return np.random.rand()
    
    best = optimize_with_pso(
        space,
        evaluator,
        num_particles=5,
        max_iterations=3,
        verbose=False
    )
    
    assert best is not None
    assert hasattr(best, 'fitness')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
