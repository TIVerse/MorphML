"""GNN-based performance predictor using Graph Neural Networks.

Predicts architecture performance from graph structure without training.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


# Check for PyTorch dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch Geometric not available. GNNPredictor requires: "
        "pip install torch torch-geometric"
    )


if TORCH_AVAILABLE:

    class ArchitectureGNN(nn.Module):
        """
        Graph Neural Network for architecture performance prediction.

        Architecture:
        - Graph Attention Network (GAT) for node embeddings
        - Global pooling (mean + max)
        - MLP predictor head

        Input: ModelGraph
        Output: Predicted accuracy (0-1)
        """

        def __init__(
            self,
            node_feature_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.3,
        ):
            """
            Initialize GNN model.

            Args:
                node_feature_dim: Dimension of node features
                hidden_dim: Hidden dimension for GNN layers
                num_layers: Number of GNN layers
                num_heads: Number of attention heads (for GAT)
                dropout: Dropout rate
            """
            super().__init__()

            self.node_feature_dim = node_feature_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # Graph attention layers
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            # First layer
            self.convs.append(
                GATConv(
                    node_feature_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Predictor head (mean + max pooling = 2 * hidden_dim)
            self.predictor = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )

        def forward(
            self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Node features [num_nodes, node_feature_dim]
                edge_index: Edge indices [2, num_edges]
                batch: Batch assignment [num_nodes]

            Returns:
                Predicted accuracy [batch_size]
            """
            # Graph convolutions with residual connections
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x_new = conv(x, edge_index)
                x_new = bn(x_new)
                x_new = F.elu(x_new)

                # Residual connection (if dimensions match)
                if i > 0 and x.shape[1] == x_new.shape[1]:
                    x = x + x_new
                else:
                    x = x_new

            # Global pooling (combine mean and max)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

            # Predict
            out = self.predictor(x)

            return out.squeeze(-1)

    class GNNPredictor:
        """
        Train and use GNN for architecture performance prediction.

        This predictor learns to estimate architecture performance from
        graph structure, enabling fast evaluation without training.

        Target: 75%+ prediction accuracy on held-out architectures
        Speedup: 100-1000x faster than full training

        Args:
            config: Configuration dict
                - node_feature_dim: Node feature dimension (default: 128)
                - hidden_dim: Hidden dimension (default: 256)
                - num_layers: Number of GNN layers (default: 4)
                - num_heads: Attention heads (default: 4)
                - dropout: Dropout rate (default: 0.3)
                - lr: Learning rate (default: 1e-3)
                - weight_decay: L2 regularization (default: 1e-5)

        Example:
            >>> # Collect training data from past experiments
            >>> train_data = [
            ...     (graph1, 0.92),  # (ModelGraph, accuracy)
            ...     (graph2, 0.88),
            ...     # ... more examples
            ... ]
            >>>
            >>> # Train predictor
            >>> predictor = GNNPredictor({'num_layers': 4})
            >>> predictor.train(train_data, num_epochs=100)
            >>>
            >>> # Predict on new architecture
            >>> predicted_acc = predictor.predict(new_graph)
            >>> print(f"Predicted accuracy: {predicted_acc:.2%}")
        """

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            """Initialize GNN predictor."""
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "GNNPredictor requires PyTorch and PyTorch Geometric. "
                    "Install with: pip install torch torch-geometric"
                )

            self.config = config or {}

            # Model configuration
            self.model = ArchitectureGNN(
                node_feature_dim=self.config.get("node_feature_dim", 128),
                hidden_dim=self.config.get("hidden_dim", 256),
                num_layers=self.config.get("num_layers", 4),
                num_heads=self.config.get("num_heads", 4),
                dropout=self.config.get("dropout", 0.3),
            )

            # Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)

            logger.info(f"GNNPredictor initialized on device: {self.device}")

            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("lr", 1e-3),
                weight_decay=self.config.get("weight_decay", 1e-5),
            )

            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
            )

            # Training stats
            self.training_history: List[Dict[str, float]] = []
            self.is_trained = False

        def train(
            self,
            train_data: List[Tuple[ModelGraph, float]],
            val_data: Optional[List[Tuple[ModelGraph, float]]] = None,
            num_epochs: int = 100,
            batch_size: int = 32,
            early_stopping_patience: int = 20,
        ) -> Dict[str, Any]:
            """
            Train GNN predictor on historical data.

            Args:
                train_data: List of (architecture, accuracy) pairs
                val_data: Optional validation data
                num_epochs: Maximum training epochs
                batch_size: Batch size
                early_stopping_patience: Stop if no improvement for N epochs

            Returns:
                Training statistics dict
            """
            logger.info(f"Training GNN predictor on {len(train_data)} examples")

            # Convert to PyTorch Geometric Data objects
            train_dataset = [self._graph_to_pyg_data(g, acc) for g, acc in train_data]

            if val_data:
                val_dataset = [self._graph_to_pyg_data(g, acc) for g, acc in val_data]
            else:
                # Use 20% of training data for validation
                split_idx = int(0.8 * len(train_dataset))
                val_dataset = train_dataset[split_idx:]
                train_dataset = train_dataset[:split_idx]

            logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} validation")

            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(num_epochs):
                # Train
                train_loss, train_mae = self._train_epoch(train_dataset, batch_size)

                # Validate
                val_loss, val_mae = self._validate(val_dataset, batch_size)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Track history
                self.training_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_mae": train_mae,
                        "val_loss": val_loss,
                        "val_mae": val_mae,
                    }
                )

                # Log progress
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    logger.info(
                        f"Epoch {epoch:3d}/{num_epochs}: "
                        f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, "
                        f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Restore best model
            self.model.load_state_dict(self.best_model_state)
            self.is_trained = True

            logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

            return {
                "best_val_loss": best_val_loss,
                "num_epochs": epoch + 1,
                "history": self.training_history,
            }

        def _train_epoch(self, dataset: List[Data], batch_size: int) -> Tuple[float, float]:
            """Train one epoch."""
            self.model.train()

            # Shuffle and batch
            indices = torch.randperm(len(dataset))
            total_loss = 0.0
            total_mae = 0.0
            num_batches = 0

            for i in range(0, len(dataset), batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_data = [dataset[idx] for idx in batch_indices]

                batch = Batch.from_data_list(batch_data).to(self.device)

                # Forward
                pred = self.model(batch.x, batch.edge_index, batch.batch)
                loss = F.mse_loss(pred, batch.y)
                mae = F.l1_loss(pred, batch.y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

            return total_loss / num_batches, total_mae / num_batches

        def _validate(self, dataset: List[Data], batch_size: int) -> Tuple[float, float]:
            """Validate on dataset."""
            self.model.eval()

            total_loss = 0.0
            total_mae = 0.0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(dataset), batch_size):
                    batch_data = dataset[i : i + batch_size]
                    batch = Batch.from_data_list(batch_data).to(self.device)

                    pred = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = F.mse_loss(pred, batch.y)
                    mae = F.l1_loss(pred, batch.y)

                    total_loss += loss.item()
                    total_mae += mae.item()
                    num_batches += 1

            return total_loss / num_batches, total_mae / num_batches

        def predict(self, graph: ModelGraph) -> float:
            """
            Predict accuracy for architecture.

            Args:
                graph: ModelGraph to evaluate

            Returns:
                Predicted accuracy (0-1)
            """
            if not self.is_trained:
                logger.warning("GNN predictor not trained, prediction may be inaccurate")

            self.model.eval()

            data = self._graph_to_pyg_data(graph, 0.0).to(self.device)

            with torch.no_grad():
                pred = self.model(data.x, data.edge_index, data.batch)

            return float(pred.item())

        def _graph_to_pyg_data(self, graph: ModelGraph, accuracy: float) -> Data:
            """
            Convert ModelGraph to PyTorch Geometric Data.

            Node features encode:
            - Operation type (one-hot)
            - Hyperparameters (normalized)
            - Positional encoding (layer depth)
            """
            # Extract nodes and edges
            node_list = list(graph.nodes.values())
            node_to_idx = {node.id: i for i, node in enumerate(node_list)}

            # Node features
            node_features = []
            for i, node in enumerate(node_list):
                feat = self._encode_node(node, i, len(node_list))
                node_features.append(feat)

            x = torch.tensor(node_features, dtype=torch.float)

            # Pad/truncate to fixed dimension
            if x.shape[1] < self.config.get("node_feature_dim", 128):
                padding = torch.zeros(
                    x.shape[0],
                    self.config.get("node_feature_dim", 128) - x.shape[1],
                )
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] > self.config.get("node_feature_dim", 128):
                x = x[:, : self.config.get("node_feature_dim", 128)]

            # Edge index
            edge_list = []
            for edge in graph.edges.values():
                source_idx = node_to_idx[edge.source_id]
                target_idx = node_to_idx[edge.target_id]
                edge_list.append([source_idx, target_idx])

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                # Empty graph - create self-loops
                edge_index = torch.tensor(
                    [[i, i] for i in range(len(node_list))], dtype=torch.long
                ).t()

            # Label
            y = torch.tensor([accuracy], dtype=torch.float)

            # Batch indicator (for single graph)
            batch = torch.zeros(x.shape[0], dtype=torch.long)

            return Data(x=x, edge_index=edge_index, y=y, batch=batch)

        def _encode_node(self, node, position: int, total_nodes: int) -> List[float]:
            """
            Encode node as feature vector.

            Features:
            - One-hot operation type (20 dims)
            - Hyperparameters (variable)
            - Positional encoding (2 dims)
            """
            features = []

            # Operation type (one-hot)
            operation_types = [
                "input",
                "output",
                "conv2d",
                "conv1d",
                "depthwise_conv",
                "maxpool",
                "avgpool",
                "globalavgpool",
                "dense",
                "linear",
                "relu",
                "gelu",
                "sigmoid",
                "tanh",
                "batchnorm",
                "layernorm",
                "dropout",
                "residual",
                "concat",
                "add",
            ]

            op_encoding = [0.0] * len(operation_types)
            if node.operation in operation_types:
                op_encoding[operation_types.index(node.operation)] = 1.0

            features.extend(op_encoding)

            # Hyperparameters (normalized)
            if hasattr(node, "params") and node.params:
                # Conv layers
                if "filters" in node.params:
                    features.append(min(node.params["filters"] / 512.0, 1.0))
                if "kernel_size" in node.params:
                    features.append(node.params["kernel_size"] / 7.0)
                if "stride" in node.params:
                    features.append(node.params["stride"] / 2.0)

                # Dense layers
                if "units" in node.params:
                    features.append(min(node.params["units"] / 2048.0, 1.0))

                # Dropout
                if "rate" in node.params:
                    features.append(node.params["rate"])

            # Positional encoding (normalized depth)
            features.append(position / max(total_nodes, 1))
            features.append(np.sin(position * 2 * np.pi / max(total_nodes, 1)))

            return features

        def save(self, path: str) -> None:
            """Save model to file."""
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "config": self.config,
                    "training_history": self.training_history,
                    "is_trained": self.is_trained,
                },
                path,
            )
            logger.info(f"GNN predictor saved to {path}")

        def load(self, path: str) -> None:
            """Load model from file."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.config = checkpoint["config"]
            self.training_history = checkpoint["training_history"]
            self.is_trained = checkpoint["is_trained"]
            logger.info(f"GNN predictor loaded from {path}")


# Fallback if PyTorch not available
else:

    class GNNPredictor:
        """Fallback GNN predictor (PyTorch not available)."""

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            raise ImportError(
                "GNNPredictor requires PyTorch and PyTorch Geometric. "
                "Install with: pip install torch torch-geometric"
            )
