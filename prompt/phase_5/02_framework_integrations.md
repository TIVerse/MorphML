# Component 2: Framework Integrations

**Duration:** Week 4  
**LOC Target:** ~4,000  
**Dependencies:** Phase 1-4

---

## 🎯 Objective

Integrate with major ML frameworks:
1. **PyTorch** - Full training support
2. **TensorFlow/Keras** - Model building
3. **JAX/Flax** - Functional NAS
4. **Scikit-learn** - Classical ML pipelines

---

## 📋 Files to Create

### 1. `integrations/pytorch_adapter.py` (~1,500 LOC)

```python
import torch
import torch.nn as nn
from typing import Dict, Any
from morphml.core.graph import ModelGraph, GraphNode

class PyTorchAdapter:
    """
    Convert ModelGraph to PyTorch nn.Module.
    
    Usage:
        adapter = PyTorchAdapter()
        model = adapter.build_model(graph)
        model.train()
    """
    
    def build_model(self, graph: ModelGraph) -> nn.Module:
        """
        Build PyTorch model from graph.
        
        Returns:
            nn.Module instance
        """
        return GraphToModule(graph)
    
    def get_trainer(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> 'PyTorchTrainer':
        """Get trainer for model."""
        return PyTorchTrainer(model, config)


class GraphToModule(nn.Module):
    """
    PyTorch module generated from ModelGraph.
    
    Dynamically creates layers based on graph structure.
    """
    
    def __init__(self, graph: ModelGraph):
        super().__init__()
        
        self.graph = graph
        self.layers = nn.ModuleDict()
        
        # Build layers
        for node_id, node in graph.nodes.items():
            layer = self._create_layer(node)
            self.layers[str(node_id)] = layer
    
    def _create_layer(self, node: GraphNode) -> nn.Module:
        """Create PyTorch layer from node."""
        op = node.operation
        params = node.params
        
        if op == 'conv2d':
            return nn.Conv2d(
                in_channels=params.get('in_channels', 3),
                out_channels=params.get('filters', 64),
                kernel_size=params.get('kernel_size', 3),
                padding=params.get('padding', 1)
            )
        
        elif op == 'maxpool':
            return nn.MaxPool2d(
                kernel_size=params.get('pool_size', 2),
                stride=params.get('stride', 2)
            )
        
        elif op == 'dense':
            return nn.Linear(
                in_features=params.get('in_features', 512),
                out_features=params.get('units', 10)
            )
        
        elif op == 'relu':
            return nn.ReLU()
        
        elif op == 'batchnorm':
            return nn.BatchNorm2d(params.get('num_features', 64))
        
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass following graph topology."""
        # Topological sort
        topo_order = self.graph.topological_sort()
        
        # Track outputs
        outputs = {}
        
        for node in topo_order:
            layer = self.layers[str(node.id)]
            
            # Get inputs
            if not node.predecessors:
                # Input node
                node_input = x
            else:
                # Combine predecessor outputs
                pred_outputs = [outputs[pred.id] for pred in node.predecessors]
                
                if len(pred_outputs) == 1:
                    node_input = pred_outputs[0]
                else:
                    # Concatenate or add
                    node_input = torch.cat(pred_outputs, dim=1)
            
            # Apply layer
            outputs[node.id] = layer(node_input)
        
        # Return output node's output
        output_node = self.graph.get_output_node()
        return outputs[output_node.id]


class PyTorchTrainer:
    """
    Trainer for PyTorch models.
    
    Handles training loop, logging, checkpointing.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50
    ) -> Dict[str, float]:
        """
        Train model.
        
        Returns:
            Training results (accuracy, loss, etc.)
        """
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            logger.info(
                f"Epoch {epoch}: "
                f"train_acc={train_acc:.4f}, "
                f"val_acc={val_acc:.4f}"
            )
        
        return {
            'val_accuracy': best_val_acc,
            'final_train_acc': train_acc
        }
    
    def _train_epoch(self, loader):
        """Single training epoch."""
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
    
    def _validate(self, loader):
        """Validation."""
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
```

---

### 2. `integrations/tensorflow_adapter.py` (~1,000 LOC)

```python
import tensorflow as tf
from tensorflow import keras

class TensorFlowAdapter:
    """TensorFlow/Keras adapter."""
    
    def build_model(self, graph: ModelGraph) -> keras.Model:
        """Build Keras model from graph."""
        inputs = keras.Input(shape=(32, 32, 3))
        
        # Build layers
        layer_outputs = {}
        
        for node in graph.topological_sort():
            layer = self._create_layer(node)
            
            if not node.predecessors:
                x = inputs
            else:
                # Get predecessor outputs
                pred_outputs = [layer_outputs[p.id] for p in node.predecessors]
                x = pred_outputs[0] if len(pred_outputs) == 1 else keras.layers.Concatenate()(pred_outputs)
            
            layer_outputs[node.id] = layer(x)
        
        # Output
        output_node = graph.get_output_node()
        outputs = layer_outputs[output_node.id]
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _create_layer(self, node: GraphNode):
        """Create Keras layer."""
        op = node.operation
        params = node.params
        
        if op == 'conv2d':
            return keras.layers.Conv2D(
                filters=params.get('filters', 64),
                kernel_size=params.get('kernel_size', 3),
                padding='same',
                activation=None
            )
        elif op == 'dense':
            return keras.layers.Dense(units=params.get('units', 10))
        elif op == 'relu':
            return keras.layers.ReLU()
        else:
            return keras.layers.Lambda(lambda x: x)
```

---

### 3. `integrations/jax_adapter.py` (~800 LOC)

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class JAXAdapter:
    """JAX/Flax adapter."""
    
    def build_model(self, graph: ModelGraph):
        """Build Flax module from graph."""
        return GraphModule(graph)


class GraphModule(nn.Module):
    """Flax module from ModelGraph."""
    graph: ModelGraph
    
    @nn.compact
    def __call__(self, x):
        # Build model using graph structure
        # Similar to PyTorch but functional
        pass
```

---

### 4. `integrations/sklearn_adapter.py` (~700 LOC)

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

class SklearnAdapter:
    """Scikit-learn adapter for classical ML."""
    
    def build_pipeline(self, graph: ModelGraph) -> Pipeline:
        """Build sklearn pipeline."""
        steps = []
        
        for node in graph.topological_sort():
            if node.operation == 'pca':
                from sklearn.decomposition import PCA
                steps.append(('pca', PCA(n_components=node.params.get('n_components', 50))))
            
            elif node.operation == 'random_forest':
                steps.append(('rf', RandomForestClassifier(
                    n_estimators=node.params.get('n_estimators', 100)
                )))
        
        return Pipeline(steps)
```

---

## 🧪 Tests

```python
def test_pytorch_adapter():
    """Test PyTorch model building."""
    graph = create_sample_graph()
    
    adapter = PyTorchAdapter()
    model = adapter.build_model(graph)
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (1, 10)
```

---

## ✅ Deliverables

- [ ] PyTorch adapter with full training
- [ ] TensorFlow/Keras adapter
- [ ] JAX/Flax adapter
- [ ] Scikit-learn pipeline builder
- [ ] Tests for each framework

---

**Next:** `03_rest_api.md`, `04_visualization.md`, `05_plugins_docs.md`
