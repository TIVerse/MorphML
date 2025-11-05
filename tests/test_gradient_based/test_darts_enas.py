"""Tests for DARTS and ENAS optimizers (GPU-guarded).

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import pytest

# Check if PyTorch and CUDA are available
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available"
)


@requires_torch
class TestDARTSOptimizer:
    """Test suite for DARTS optimizer (requires PyTorch)."""

    def test_import(self):
        """Test that DARTS can be imported."""
        from morphml.optimizers.gradient_based import DARTS

        assert DARTS is not None

    def test_initialization(self):
        """Test DARTS initialization."""
        from morphml.core.dsl import create_cnn_space
        from morphml.optimizers.gradient_based import DARTS

        space = create_cnn_space(num_classes=10)
        config = {
            'learning_rate_w': 0.025,
            'learning_rate_alpha': 3e-4,
            'num_nodes': 4,
            'num_steps': 10
        }

        optimizer = DARTS(space, config)

        assert optimizer is not None
        assert optimizer.lr_w == 0.025
        assert optimizer.lr_alpha == 3e-4
        assert optimizer.num_nodes == 4

    def test_operation_set(self):
        """Test DARTS operation set definition."""
        from morphml.core.dsl import create_cnn_space
        from morphml.optimizers.gradient_based import DARTS

        space = create_cnn_space(num_classes=10)
        optimizer = DARTS(space, {})

        ops = optimizer._get_operation_set()

        assert 'skip_connect' in ops
        assert 'sep_conv_3x3' in ops
        assert 'none' in ops
        assert len(ops) > 0

    @requires_cuda
    def test_search_requires_gpu(self):
        """Test that search method exists (GPU validation deferred)."""
        from morphml.core.dsl import create_cnn_space
        from morphml.optimizers.gradient_based import DARTS

        space = create_cnn_space(num_classes=10)
        optimizer = DARTS(space, {'num_steps': 5})

        # Just check method exists
        assert hasattr(optimizer, 'search')
        assert hasattr(optimizer, 'derive_architecture')


@requires_torch
class TestENASOptimizer:
    """Test suite for ENAS optimizer (requires PyTorch)."""

    def test_import(self):
        """Test that ENAS can be imported."""
        from morphml.optimizers.gradient_based import ENAS

        assert ENAS is not None

    def test_initialization(self):
        """Test ENAS initialization."""
        from morphml.core.dsl import create_cnn_space
        from morphml.optimizers.gradient_based import ENAS

        space = create_cnn_space(num_classes=10)
        config = {
            'controller_lr': 3e-4,
            'shared_lr': 0.05,
            'num_layers': 12
        }

        optimizer = ENAS(space, config)

        assert optimizer is not None
        assert optimizer.controller_lr == 3e-4
        assert optimizer.shared_lr == 0.05
        assert optimizer.num_layers == 12

    def test_operation_set(self):
        """Test ENAS operation set definition."""
        from morphml.core.dsl import create_cnn_space
        from morphml.optimizers.gradient_based import ENAS

        space = create_cnn_space(num_classes=10)
        optimizer = ENAS(space, {})

        ops = optimizer._get_operation_set()

        assert 'identity' in ops
        assert 'sep_conv_3x3' in ops
        assert 'none' in ops
        assert len(ops) > 0

    @requires_cuda
    def test_search_requires_gpu(self):
        """Test that search method exists (GPU validation deferred)."""
        from morphml.core.dsl import create_cnn_space
        from morphml.optimizers.gradient_based import ENAS

        space = create_cnn_space(num_classes=10)
        optimizer = ENAS(space, {'num_layers': 6})

        # Just check method exists
        assert hasattr(optimizer, 'search')
        assert hasattr(optimizer, 'train_shared_model')
        assert hasattr(optimizer, 'train_controller')


@requires_torch
class TestGradientBasedOperations:
    """Test operations module."""

    def test_import_operations(self):
        """Test that operations can be imported."""
        from morphml.optimizers.gradient_based.operations import (
            SepConv,
            DilConv,
            Identity,
            Zero,
        )

        assert SepConv is not None
        assert DilConv is not None
        assert Identity is not None
        assert Zero is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
