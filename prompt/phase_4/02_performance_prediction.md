# Component 2: Performance Prediction

**Duration:** Weeks 3-4  
**LOC Target:** ~4,000  
**Dependencies:** Component 1

---

## 🎯 Objective

Predict architecture performance without full training:
1. **Learning Curve Extrapolation** - Predict final accuracy from early epochs
2. **Proxy Metrics** - Use cheaper metrics (e.g., #params, FLOPs)
3. **Graph Neural Networks** - Learn performance from graph structure
4. **Target:** 75%+ prediction accuracy, 10x speedup

---

## 📋 Files to Create

### 1. `meta_learning/predictors/gnn_predictor.py` (~2,000 LOC)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class ArchitectureGNN(nn.Module):
    """
    Graph Neural Network for architecture performance prediction.
    
    Input: ModelGraph
    Output: Predicted accuracy
    
    Architecture:
    - Node features: operation type, hyperparameters
    - Edge features: connection type
    - Global pooling + MLP predictor
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Predicted accuracy [batch_size, 1]
        """
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Predict
        out = self.predictor(x)
        
        return out.squeeze(-1)


class GNNPredictor:
    """
    Train and use GNN for performance prediction.
    
    Usage:
        predictor = GNNPredictor()
        predictor.train(train_data)  # List of (graph, accuracy) pairs
        predicted_acc = predictor.predict(new_graph)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.model = ArchitectureGNN(
            node_feature_dim=config.get('node_feature_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 3)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 1e-3)
        )
    
    def train(
        self,
        train_data: List[Tuple[ModelGraph, float]],
        num_epochs: int = 100
    ):
        """
        Train GNN predictor.
        
        Args:
            train_data: List of (architecture, accuracy) pairs
            num_epochs: Training epochs
        """
        from torch_geometric.data import Data, DataLoader
        
        # Convert to PyG Data objects
        dataset = []
        for graph, accuracy in train_data:
            data = self._graph_to_pyg_data(graph, accuracy)
            dataset.append(data)
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in loader:
                batch = batch.to(self.device)
                
                # Forward
                pred = self.model(batch.x, batch.edge_index, batch.batch)
                loss = nn.MSELoss()(pred, batch.y)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
    
    def predict(self, graph: ModelGraph) -> float:
        """Predict accuracy for architecture."""
        self.model.eval()
        
        data = self._graph_to_pyg_data(graph, 0.0)  # Dummy label
        data = data.to(self.device)
        
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index, data.batch)
        
        return pred.item()
    
    def _graph_to_pyg_data(self, graph: ModelGraph, accuracy: float) -> Data:
        """Convert ModelGraph to PyG Data."""
        # Node features (one-hot operation types + hyperparameters)
        node_features = []
        for node in graph.nodes.values():
            feat = self._encode_node(node)
            node_features.append(feat)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge index
        edge_list = []
        for edge in graph.edges.values():
            edge_list.append([edge.source.id, edge.target.id])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Label
        y = torch.tensor([accuracy], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def _encode_node(self, node: GraphNode) -> List[float]:
        """Encode node as feature vector."""
        # Simplified: one-hot operation type
        operations = ['conv2d', 'maxpool', 'avgpool', 'dense', 'relu', 'batchnorm']
        feat = [0.0] * len(operations)
        
        if node.operation in operations:
            feat[operations.index(node.operation)] = 1.0
        
        # Add hyperparameters
        if node.operation == 'conv2d':
            feat.append(node.get_param('filters', 32) / 512.0)
            feat.append(node.get_param('kernel_size', 3) / 7.0)
        
        return feat
```

---

### 2. `meta_learning/predictors/early_stopping.py` (~1,000 LOC)

```python
class LearningCurvePredictor:
    """
    Predict final accuracy from early training epochs.
    
    Uses power law or exponential fitting:
        acc(t) = a - b * t^(-c)
    """
    
    @staticmethod
    def predict_final_accuracy(
        observed_accuracies: List[float],
        observed_epochs: List[int],
        final_epoch: int = 200
    ) -> float:
        """
        Extrapolate learning curve.
        
        Args:
            observed_accuracies: Accuracies at early epochs
            observed_epochs: Corresponding epochs
            final_epoch: Epoch to predict
        
        Returns:
            Predicted final accuracy
        """
        from scipy.optimize import curve_fit
        
        def power_law(t, a, b, c):
            return a - b * np.power(t, -c)
        
        # Fit curve
        try:
            params, _ = curve_fit(
                power_law,
                observed_epochs,
                observed_accuracies,
                p0=[0.9, 0.1, 0.5],
                bounds=([0, 0, 0], [1.0, 1.0, 5.0])
            )
            
            # Predict
            pred = power_law(final_epoch, *params)
            return np.clip(pred, 0.0, 1.0)
        
        except Exception as e:
            logger.warning(f"Curve fitting failed: {e}")
            return observed_accuracies[-1]  # Return last observed
```

---

### 3. `meta_learning/predictors/proxy_metrics.py` (~500 LOC)

```python
class ProxyMetricPredictor:
    """
    Predict performance using cheap proxy metrics.
    
    Proxies:
    - Number of parameters
    - FLOPs
    - Network depth
    - Width
    """
    
    def __init__(self):
        # Train linear model on historical data
        self.model = None
    
    def train(self, data: List[Tuple[Dict, float]]):
        """
        Train predictor.
        
        Args:
            data: List of (proxy_metrics, accuracy) pairs
        """
        from sklearn.ensemble import RandomForestRegressor
        
        X = [list(metrics.values()) for metrics, _ in data]
        y = [acc for _, acc in data]
        
        self.model = RandomForestRegressor(n_estimators=100)
        self.model.fit(X, y)
    
    def predict(self, proxy_metrics: Dict[str, float]) -> float:
        """Predict accuracy from proxy metrics."""
        X = [list(proxy_metrics.values())]
        return self.model.predict(X)[0]
```

---

### 4. `meta_learning/predictors/ensemble.py` (~500 LOC)

```python
class EnsemblePredictor:
    """
    Ensemble of multiple predictors.
    
    Combines:
    - GNN predictor
    - Learning curve predictor
    - Proxy metric predictor
    """
    
    def __init__(self, predictors: List):
        self.predictors = predictors
        self.weights = [1.0 / len(predictors)] * len(predictors)
    
    def predict(self, graph: ModelGraph, **kwargs) -> float:
        """Weighted ensemble prediction."""
        predictions = [
            p.predict(graph, **kwargs) for p in self.predictors
        ]
        
        return sum(w * p for w, p in zip(self.weights, predictions))
```

---

## 🧪 Tests

```python
def test_gnn_predictor():
    """Test GNN performance prediction."""
    predictor = GNNPredictor({})
    
    # Generate synthetic training data
    train_data = [(random_graph(), random.uniform(0.7, 0.95)) for _ in range(100)]
    
    predictor.train(train_data, num_epochs=50)
    
    # Test prediction
    test_graph = random_graph()
    pred_acc = predictor.predict(test_graph)
    
    assert 0.0 <= pred_acc <= 1.0
```

---

## ✅ Deliverables

- [ ] GNN-based predictor
- [ ] Learning curve extrapolation
- [ ] Proxy metric predictor
- [ ] Ensemble predictor
- [ ] 75%+ prediction accuracy demonstrated
- [ ] 10x+ speedup over full training

---

**Next:** `03_knowledge_base.md`, `04_strategy_evolution.md`, `05_transfer_learning.md`

(Remaining Phase 4 components will be similar in structure - focused implementations with tests)
