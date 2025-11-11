"""Framework integrations for MorphML.

Provides adapters to convert ModelGraph architectures to various ML frameworks:
- PyTorch (with training support)
- TensorFlow/Keras
- JAX/Flax
- Scikit-learn

Example:
    >>> from morphml.integrations import PyTorchAdapter
    >>> adapter = PyTorchAdapter()
    >>> model = adapter.build_model(graph)
    >>> trainer = adapter.get_trainer(model, config)
    >>> results = trainer.train(train_loader, val_loader)
"""

from morphml.integrations.jax_adapter import JAXAdapter
from morphml.integrations.pytorch_adapter import PyTorchAdapter, PyTorchTrainer
from morphml.integrations.sklearn_adapter import SklearnAdapter
from morphml.integrations.tensorflow_adapter import TensorFlowAdapter

__all__ = [
    "PyTorchAdapter",
    "PyTorchTrainer",
    "TensorFlowAdapter",
    "JAXAdapter",
    "SklearnAdapter",
]
