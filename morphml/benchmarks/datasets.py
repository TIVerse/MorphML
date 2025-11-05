"""Dataset loaders and utilities for benchmarking.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Dict, Optional, Tuple

import numpy as np

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def load_cifar10(
    data_dir: Optional[str] = None, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset.

    Tries multiple backends in order:
    1. torchvision (if available)
    2. tensorflow/keras (if available)
    3. OpenML (fallback, may be slow)

    Args:
        data_dir: Directory to download/cache data
        normalize: Whether to normalize to [0, 1]

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Try torchvision first
    try:
        import torchvision
        import torchvision.transforms as transforms

        logger.info("Loading CIFAR-10 via torchvision...")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=data_dir or "./data", train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_dir or "./data", train=False, download=True, transform=transform
        )

        # Convert to numpy
        X_train = np.array([trainset[i][0].numpy() for i in range(len(trainset))])
        y_train = np.array([trainset[i][1] for i in range(len(trainset))])
        X_test = np.array([testset[i][0].numpy() for i in range(len(testset))])
        y_test = np.array([testset[i][1] for i in range(len(testset))])

        if not normalize:
            X_train *= 255.0
            X_test *= 255.0

        logger.info(f"Loaded CIFAR-10 via torchvision: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test

    except ImportError:
        pass

    # Try Keras/TensorFlow
    try:
        from tensorflow import keras

        logger.info("Loading CIFAR-10 via Keras...")

        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

        # Convert to channel-first format (3, 32, 32)
        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_test = np.transpose(X_test, (0, 3, 1, 2))

        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # Normalize
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

        if normalize:
            X_train /= 255.0
            X_test /= 255.0

        logger.info(f"Loaded CIFAR-10 via Keras: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test

    except ImportError:
        pass

    # Fallback to OpenML (slower)
    try:
        from sklearn.datasets import fetch_openml

        logger.info("Loading CIFAR-10 via OpenML (may be slow)...")
        logger.warning("For faster loading, install torchvision or tensorflow")

        cifar = fetch_openml("CIFAR_10", version=1, cache=True, data_home=data_dir)
        X = cifar.data.astype("float32")
        y = cifar.target.astype("int64")

        # Reshape to (N, 3, 32, 32)
        X = X.reshape(-1, 3, 32, 32)

        if normalize:
            X /= 255.0

        # Split train/test (first 50000 train, rest test)
        X_train, X_test = X[:50000], X[50000:]
        y_train, y_test = y[:50000], y[50000:]

        logger.info(f"Loaded CIFAR-10 via OpenML: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error(f"Failed to load CIFAR-10: {e}")
        logger.error("Please install torchvision or tensorflow for CIFAR-10 support")
        raise RuntimeError(
            "CIFAR-10 dataset loading failed. Install one of:\n"
            "  pip install torchvision\n"
            "  pip install tensorflow\n"
            "  pip install scikit-learn (slower)"
        )


def load_mnist(
    data_dir: Optional[str] = None, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset.

    Args:
        data_dir: Directory to download/cache data
        normalize: Whether to normalize to [0, 1]

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        logger.error("scikit-learn required")
        raise

    logger.info("Loading MNIST dataset from OpenML...")

    mnist = fetch_openml("mnist_784", version=1, cache=True, data_home=data_dir)
    X = mnist.data.astype("float32")
    y = mnist.target.astype("int64")

    if normalize:
        X /= 255.0

    # Split train/test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    logger.info(f"Loaded MNIST: train={X_train.shape}, test={X_test.shape}")

    return X_train, y_train, X_test, y_test


def load_fashion_mnist(
    data_dir: Optional[str] = None, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Fashion-MNIST dataset.

    Args:
        data_dir: Directory to download/cache data
        normalize: Whether to normalize to [0, 1]

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        logger.error("scikit-learn required")
        raise

    logger.info("Loading Fashion-MNIST dataset from OpenML...")

    fashion = fetch_openml("Fashion-MNIST", version=1, cache=True, data_home=data_dir)
    X = fashion.data.astype("float32")
    y = fashion.target.astype("int64")

    if normalize:
        X /= 255.0

    # Split train/test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    logger.info(f"Loaded Fashion-MNIST: train={X_train.shape}, test={X_test.shape}")

    return X_train, y_train, X_test, y_test


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary with dataset metadata
    """
    dataset_info = {
        "cifar10": {
            "name": "CIFAR-10",
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "train_size": 50000,
            "test_size": 10000,
            "type": "image_classification",
        },
        "mnist": {
            "name": "MNIST",
            "num_classes": 10,
            "input_shape": (784,),
            "train_size": 60000,
            "test_size": 10000,
            "type": "image_classification",
        },
        "fashion_mnist": {
            "name": "Fashion-MNIST",
            "num_classes": 10,
            "input_shape": (784,),
            "train_size": 60000,
            "test_size": 10000,
            "type": "image_classification",
        },
    }

    return dataset_info.get(dataset_name, {})


class DatasetLoader:
    """Unified dataset loader interface."""

    LOADERS = {
        "cifar10": load_cifar10,
        "mnist": load_mnist,
        "fashion_mnist": load_fashion_mnist,
    }

    @classmethod
    def load(cls, dataset_name: str, **kwargs):
        """
        Load a dataset by name.

        Args:
            dataset_name: Name of dataset
            **kwargs: Additional arguments for loader

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        loader = cls.LOADERS.get(dataset_name)
        if loader is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return loader(**kwargs)

    @classmethod
    def list_available(cls):
        """List available datasets."""
        return list(cls.LOADERS.keys())
