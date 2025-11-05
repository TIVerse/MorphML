"""Tests for acquisition functions.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.optimizers.bayesian.acquisition import (
    expected_improvement,
    lower_confidence_bound,
    probability_of_improvement,
    upper_confidence_bound,
)


class TestAcquisitionFunctions:
    """Test acquisition functions."""

    def test_expected_improvement_basic(self):
        """Test EI with basic inputs."""
        mu = np.array([0.5, 0.7, 0.3, 0.9])
        sigma = np.array([0.1, 0.15, 0.05, 0.2])
        f_best = 0.6

        ei = expected_improvement(mu, sigma, f_best)

        assert len(ei) == 4
        assert np.all(ei >= 0)
        # Point with mu=0.9 should have highest EI
        assert ei[3] > ei[0]

    def test_expected_improvement_edge_cases(self):
        """Test EI edge cases."""
        # Zero sigma should give zero EI
        mu = np.array([0.5, 0.7])
        sigma = np.array([0.0, 0.0])
        f_best = 0.6

        ei = expected_improvement(mu, sigma, f_best)
        assert np.all(ei == 0)

        # All below f_best with zero sigma
        mu = np.array([0.3, 0.4])
        sigma = np.array([0.0, 0.0])
        f_best = 0.6

        ei = expected_improvement(mu, sigma, f_best)
        assert np.all(ei == 0)

    def test_expected_improvement_with_xi(self):
        """Test EI with exploration parameter."""
        mu = np.array([0.5, 0.7])
        sigma = np.array([0.1, 0.15])
        f_best = 0.6

        ei_no_xi = expected_improvement(mu, sigma, f_best, xi=0.0)
        ei_with_xi = expected_improvement(mu, sigma, f_best, xi=0.01)

        # With xi, EI should be different
        assert not np.allclose(ei_no_xi, ei_with_xi)

    def test_upper_confidence_bound(self):
        """Test UCB acquisition function."""
        mu = np.array([0.5, 0.7, 0.3])
        sigma = np.array([0.1, 0.15, 0.05])

        # Test with different kappa values
        ucb1 = upper_confidence_bound(mu, sigma, kappa=1.0)
        ucb2 = upper_confidence_bound(mu, sigma, kappa=2.0)

        assert len(ucb1) == 3
        assert len(ucb2) == 3

        # Higher kappa should give higher UCB
        assert np.all(ucb2 > ucb1)

        # UCB should be mu + kappa * sigma
        expected_ucb1 = mu + 1.0 * sigma
        assert np.allclose(ucb1, expected_ucb1)

    def test_probability_of_improvement(self):
        """Test PI acquisition function."""
        mu = np.array([0.5, 0.7, 0.3, 0.9])
        sigma = np.array([0.1, 0.15, 0.05, 0.2])
        f_best = 0.6

        pi = probability_of_improvement(mu, sigma, f_best)

        assert len(pi) == 4
        assert np.all(pi >= 0)
        assert np.all(pi <= 1)  # Probabilities should be in [0, 1]

        # Higher mu should generally give higher PI
        assert pi[3] > pi[2]  # mu=0.9 > mu=0.3

    def test_probability_of_improvement_with_xi(self):
        """Test PI with exploration parameter."""
        mu = np.array([0.5, 0.7])
        sigma = np.array([0.1, 0.15])
        f_best = 0.6

        pi_no_xi = probability_of_improvement(mu, sigma, f_best, xi=0.0)
        pi_with_xi = probability_of_improvement(mu, sigma, f_best, xi=0.01)

        # With xi, PI should be different
        assert not np.allclose(pi_no_xi, pi_with_xi)

    def test_lower_confidence_bound(self):
        """Test LCB acquisition function (for minimization)."""
        mu = np.array([0.5, 0.7, 0.3])
        sigma = np.array([0.1, 0.15, 0.05])

        # Test with different kappa values
        lcb1 = lower_confidence_bound(mu, sigma, kappa=1.0)
        lcb2 = lower_confidence_bound(mu, sigma, kappa=2.0)

        assert len(lcb1) == 3
        assert len(lcb2) == 3

        # Higher kappa should give lower LCB (more exploration)
        assert np.all(lcb2 < lcb1)

        # LCB should be mu - kappa * sigma
        expected_lcb1 = mu - 1.0 * sigma
        assert np.allclose(lcb1, expected_lcb1)

    def test_acquisition_consistency(self):
        """Test that acquisition functions return consistent shapes."""
        mu = np.array([0.5, 0.6, 0.7, 0.8])
        sigma = np.array([0.1, 0.1, 0.1, 0.1])
        f_best = 0.65

        ei = expected_improvement(mu, sigma, f_best)
        ucb = upper_confidence_bound(mu, sigma)
        pi = probability_of_improvement(mu, sigma, f_best)
        lcb = lower_confidence_bound(mu, sigma)

        # All should return same shape
        assert ei.shape == mu.shape
        assert ucb.shape == mu.shape
        assert pi.shape == mu.shape
        assert lcb.shape == mu.shape

    def test_acquisition_with_single_point(self):
        """Test acquisition functions with single point."""
        mu = np.array([0.5])
        sigma = np.array([0.1])
        f_best = 0.6

        ei = expected_improvement(mu, sigma, f_best)
        ucb = upper_confidence_bound(mu, sigma)
        pi = probability_of_improvement(mu, sigma, f_best)

        assert len(ei) == 1
        assert len(ucb) == 1
        assert len(pi) == 1
        assert not np.isnan(ei[0])
        assert not np.isnan(ucb[0])
        assert not np.isnan(pi[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
