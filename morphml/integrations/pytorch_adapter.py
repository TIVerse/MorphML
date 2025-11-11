"""PyTorch adapter for MorphML.

Converts ModelGraph to PyTorch nn.Module with full training support.

Example:
    >>> from morphml.integrations import PyTorchAdapter
    >>> adapter = PyTorchAdapter()
    >>> model = adapter.build_model(graph)
    >>> trainer = adapter.get_trainer(model, config={'learning_rate': 1e-3})
    >>> results = trainer.train(train_loader, val_loader, num_epochs=50)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from morphml.core.graph import GraphNode, ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class PyTorchAdapter:
    """
    Convert ModelGraph to PyTorch nn.Module.

    Provides full integration with PyTorch including:
    - Model building from graph
    - Automatic shape inference
    - Training support
    - GPU acceleration

    Example:
        >>> adapter = PyTorchAdapter()
        >>> model = adapter.build_model(graph)
        >>> model.train()
        >>> output = model(torch.randn(1, 3, 32, 32))
    """

    def __init__(self):
        """Initialize PyTorch adapter."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PyTorchAdapter. " "Install with: pip install torch"
            )
        logger.info("Initialized PyTorchAdapter")

    def build_model(
        self, graph: ModelGraph, input_shape: Optional[Tuple[int, ...]] = None
    ) -> nn.Module:
        """
        Build PyTorch model from graph.

        Args:
            graph: ModelGraph to convert
            input_shape: Optional input shape (C, H, W)

        Returns:
            nn.Module instance

        Example:
            >>> model = adapter.build_model(graph, input_shape=(3, 32, 32))
        """
        return GraphToModule(graph, input_shape)

    def get_trainer(
        self, model: nn.Module, config: Optional[Dict[str, Any]] = None
    ) -> "PyTorchTrainer":
        """
        Get trainer for model.

        Args:
            model: PyTorch model
            config: Training configuration

        Returns:
            PyTorchTrainer instance

        Example:
            >>> trainer = adapter.get_trainer(model, {
            ...     'learning_rate': 1e-3,
            ...     'weight_decay': 1e-4
            ... })
        """
        return PyTorchTrainer(model, config or {})


class GraphToModule(nn.Module):
    """
    PyTorch module generated from ModelGraph.

    Dynamically creates layers based on graph structure and handles
    forward pass following graph topology.

    Attributes:
        graph: Source ModelGraph
        layers: ModuleDict of created layers
        input_shape: Expected input shape
    """

    def __init__(self, graph: ModelGraph, input_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize module from graph.

        Args:
            graph: ModelGraph to convert
            input_shape: Optional input shape for inference
        """
        super().__init__()

        self.graph = graph
        self.input_shape = input_shape or (3, 32, 32)
        self.layers = nn.ModuleDict()

        # Infer shapes
        self.shapes = self._infer_shapes()

        # Build layers
        for node_id, node in graph.nodes.items():
            layer = self._create_layer(node)
            if layer is not None:
                self.layers[str(node_id)] = layer

        logger.info(f"Created PyTorch model with {len(self.layers)} layers")

    def _infer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Infer shapes for all nodes."""
        shapes = {}

        for node in self.graph.topological_sort():
            if node.operation == "input":
                shapes[node.id] = self.input_shape
            else:
                shapes[node.id] = self._infer_node_shape(node, shapes)

        return shapes

    def _infer_node_shape(
        self, node: GraphNode, shapes: Dict[str, Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        """Infer shape for a single node."""
        if not node.predecessors:
            return self.input_shape

        pred_shape = shapes[list(node.predecessors)[0].id]
        op = node.operation
        params = node.params

        if op == "conv2d":
            C, H, W = pred_shape
            filters = params.get("filters", 64)
            kernel = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", 1)

            H_out = (H + 2 * padding - kernel) // stride + 1
            W_out = (W + 2 * padding - kernel) // stride + 1
            return (filters, H_out, W_out)

        elif op in ["maxpool", "avgpool"]:
            C, H, W = pred_shape
            pool_size = params.get("pool_size", 2)
            stride = params.get("stride", pool_size)
            return (C, H // stride, W // stride)

        elif op == "flatten":
            return (int(np.prod(pred_shape)),)

        elif op == "dense":
            return (params.get("units", 10),)

        else:
            return pred_shape

    def _create_layer(self, node: GraphNode) -> Optional[nn.Module]:
        """
        Create PyTorch layer from node.

        Args:
            node: GraphNode to convert

        Returns:
            nn.Module or None for functional operations
        """
        op = node.operation
        params = node.params
        self.shapes.get(node.id)

        if op == "input":
            return None  # No layer needed

        elif op == "conv2d":
            # Get input channels from predecessor
            if node.predecessors:
                pred_shape = self.shapes[list(node.predecessors)[0].id]
                in_channels = pred_shape[0]
            else:
                in_channels = params.get("in_channels", 3)

            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=params.get("filters", 64),
                kernel_size=params.get("kernel_size", 3),
                stride=params.get("stride", 1),
                padding=params.get("padding", 1),
            )

        elif op == "maxpool":
            return nn.MaxPool2d(
                kernel_size=params.get("pool_size", 2),
                stride=params.get("stride", params.get("pool_size", 2)),
            )

        elif op == "avgpool":
            return nn.AvgPool2d(
                kernel_size=params.get("pool_size", 2),
                stride=params.get("stride", params.get("pool_size", 2)),
            )

        elif op == "dense":
            # Get input features from predecessor
            if node.predecessors:
                pred_shape = self.shapes[list(node.predecessors)[0].id]
                in_features = int(np.prod(pred_shape))
            else:
                in_features = params.get("in_features", 512)

            return nn.Linear(in_features=in_features, out_features=params.get("units", 10))

        elif op == "relu":
            return nn.ReLU()

        elif op == "sigmoid":
            return nn.Sigmoid()

        elif op == "tanh":
            return nn.Tanh()

        elif op == "softmax":
            return nn.Softmax(dim=1)

        elif op == "batchnorm":
            # Infer num_features from predecessor
            if node.predecessors:
                pred_shape = self.shapes[list(node.predecessors)[0].id]
                if len(pred_shape) == 3:  # (C, H, W)
                    return nn.BatchNorm2d(pred_shape[0])
                else:  # (features,)
                    return nn.BatchNorm1d(pred_shape[0])
            return nn.Identity()

        elif op == "dropout":
            return nn.Dropout(p=params.get("rate", 0.5))

        elif op == "flatten":
            return nn.Flatten()

        else:
            logger.warning(f"Unknown operation: {op}, using Identity")
            return nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass following graph topology.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Topological sort
        topo_order = self.graph.topological_sort()

        # Track outputs
        outputs = {}

        for node in topo_order:
            # Get layer
            layer = self.layers.get(str(node.id))

            # Get input
            if not node.predecessors:
                # Input node
                node_input = x
            else:
                # Combine predecessor outputs
                pred_outputs = [outputs[pred.id] for pred in node.predecessors]

                if len(pred_outputs) == 1:
                    node_input = pred_outputs[0]
                else:
                    # Concatenate along channel dimension
                    node_input = torch.cat(pred_outputs, dim=1)

            # Apply layer
            if layer is not None:
                outputs[node.id] = layer(node_input)
            else:
                outputs[node.id] = node_input

        # Return output node's output
        output_nodes = [n for n in self.graph.nodes.values() if not n.successors]
        if output_nodes:
            return outputs[output_nodes[0].id]
        else:
            # Return last node's output
            return outputs[topo_order[-1].id]


class PyTorchTrainer:
    """
    Trainer for PyTorch models.

    Handles training loop, validation, logging, and checkpointing.

    Attributes:
        model: PyTorch model to train
        config: Training configuration
        device: Device (CPU/GPU)
        optimizer: Optimizer instance
        criterion: Loss function
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            config: Training configuration with keys:
                - learning_rate: Learning rate (default: 1e-3)
                - weight_decay: Weight decay (default: 0)
                - optimizer: Optimizer name (default: 'adam')
                - loss: Loss function name (default: 'cross_entropy')
        """
        self.model = model
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        logger.info(f"Using device: {self.device}")

        # Optimizer
        optimizer_name = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 0)

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Loss function
        loss_name = config.get("loss", "cross_entropy").lower()
        if loss_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        if config.get("use_scheduler", False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )
        else:
            self.scheduler = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
    ) -> Dict[str, float]:
        """
        Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs

        Returns:
            Training results dictionary with:
                - best_val_accuracy: Best validation accuracy
                - final_train_accuracy: Final training accuracy
                - final_val_accuracy: Final validation accuracy
        """
        best_val_acc = 0.0
        train_acc = 0.0
        val_acc = 0.0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step(val_acc)

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                    )

        return {
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": train_acc,
            "final_val_accuracy": val_acc,
        }

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Single training epoch.

        Args:
            loader: Data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            # Forward
            logits = self.model(X)
            loss = self.criterion(logits, y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return total_loss / len(loader), correct / total

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Validation.

        Args:
            loader: Data loader

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return total_loss / len(loader), correct / total

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Evaluation metrics
        """
        test_loss, test_acc = self._validate(test_loader)

        return {"test_loss": test_loss, "test_accuracy": test_acc}
