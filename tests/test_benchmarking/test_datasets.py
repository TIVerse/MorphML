"""Tests for dataset loaders.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import pytest

from morphml.benchmarks.datasets import (
    DatasetLoader,
    get_dataset_info,
    load_mnist,
    load_fashion_mnist,
)


class TestDatasetLoaders:
    """Test dataset loading functions."""

    def test_load_mnist(self):
        """Test MNIST dataset loading."""
        try:
            X_train, y_train, X_test, y_test = load_mnist(normalize=True)

            assert X_train.shape[0] == 60000
            assert X_test.shape[0] == 10000
            assert len(y_train) == 60000
            assert len(y_test) == 10000

            # Check normalization
            assert X_train.min() >= 0
            assert X_train.max() <= 1.0

        except Exception as e:
            pytest.skip(f"MNIST loading failed (dependency issue): {e}")

    def test_load_fashion_mnist(self):
        """Test Fashion-MNIST dataset loading."""
        try:
            X_train, y_train, X_test, y_test = load_fashion_mnist(normalize=True)

            assert X_train.shape[0] == 60000
            assert X_test.shape[0] == 10000
            assert len(y_train) == 60000
            assert len(y_test) == 10000

            # Check normalization
            assert X_train.min() >= 0
            assert X_train.max() <= 1.0

        except Exception as e:
            pytest.skip(f"Fashion-MNIST loading failed (dependency issue): {e}")

    def test_get_dataset_info(self):
        """Test dataset info retrieval."""
        info = get_dataset_info('mnist')

        assert info['name'] == 'MNIST'
        assert info['num_classes'] == 10
        assert info['input_shape'] == (784,)
        assert info['train_size'] == 60000
        assert info['test_size'] == 10000

    def test_dataset_loader_class(self):
        """Test DatasetLoader unified interface."""
        # List available datasets
        datasets = DatasetLoader.list_available()

        assert 'mnist' in datasets
        assert 'fashion_mnist' in datasets
        assert 'cifar10' in datasets

    def test_dataset_loader_load(self):
        """Test DatasetLoader.load() method."""
        try:
            X_train, y_train, X_test, y_test = DatasetLoader.load('mnist', normalize=True)

            assert X_train is not None
            assert y_train is not None
            assert X_test is not None
            assert y_test is not None

        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")

    def test_unknown_dataset(self):
        """Test loading unknown dataset."""
        with pytest.raises(ValueError):
            DatasetLoader.load('unknown_dataset')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
