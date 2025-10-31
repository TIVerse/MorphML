# Component 2: Gradient-Based NAS (DARTS & ENAS)

**Duration:** Weeks 3-4  
**LOC Target:** ~6,000  
**Dependencies:** Phase 1, PyTorch

---

## 🎯 Objective

Implement gradient-based Neural Architecture Search using:
1. **DARTS** (Differentiable Architecture Search) - Continuous relaxation of search space
2. **ENAS** (Efficient Neural Architecture Search) - Weight sharing with RL controller

These methods enable GPU-accelerated architecture search by making the search space differentiable.

---

## 📋 Files to Create

### 1. `gradient_based/darts_optimizer.py` (~2,500 LOC)

**`DARTSOptimizer` class:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class DARTSOptimizer(BaseOptimizer):
    """
    Differentiable Architecture Search (DARTS).
    
    Key Idea:
    - Represent architecture as a weighted sum of all possible operations
    - Architecture parameters (α) are continuous and differentiable
    - Jointly optimize weights (w) and architecture (α) via gradient descent
    
    Search Space:
    - Each edge between nodes has mixed operations
    - α_ij^o = weight for operation o on edge (i,j)
    - Final operation selected via argmax(α)
    
    Bi-level Optimization:
    - Inner loop: optimize weights w on training set
    - Outer loop: optimize architecture α on validation set
    
    Config:
        learning_rate_w: Learning rate for weights (default: 0.025)
        learning_rate_alpha: Learning rate for architecture (default: 3e-4)
        momentum: Momentum for SGD (default: 0.9)
        weight_decay: L2 regularization (default: 3e-4)
        grad_clip: Gradient clipping value (default: 5.0)
        num_nodes: Number of intermediate nodes (default: 4)
        num_steps: Total search steps (default: 50)
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        self.lr_w = config.get('learning_rate_w', 0.025)
        self.lr_alpha = config.get('learning_rate_alpha', 3e-4)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 3e-4)
        self.grad_clip = config.get('grad_clip', 5.0)
        
        self.num_nodes = config.get('num_nodes', 4)
        self.num_steps = config.get('num_steps', 50)
        
        # Build supernet
        self.supernet = self._build_supernet()
        
        # Architecture parameters (α)
        self.alphas = self._initialize_architecture_params()
        
        # Optimizers
        self.optimizer_w = torch.optim.SGD(
            self.supernet.parameters(),
            lr=self.lr_w,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        self.optimizer_alpha = torch.optim.Adam(
            self.alphas,
            lr=self.lr_alpha,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        self.step_count = 0
    
    def _build_supernet(self) -> nn.Module:
        """
        Build supernet containing all candidate operations.
        
        Structure:
        - Input node
        - N intermediate nodes
        - Output node
        - Each intermediate node receives from all previous nodes
        """
        return DARTSSupernet(
            num_nodes=self.num_nodes,
            operations=self._get_operation_set(),
            num_classes=self.config.get('num_classes', 10)
        )
    
    def _get_operation_set(self) -> List[str]:
        """Define set of candidate operations."""
        return [
            'none',           # Skip connection
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5'
        ]
    
    def _initialize_architecture_params(self) -> List[nn.Parameter]:
        """
        Initialize architecture parameters α.
        
        For each edge (i,j), we have |operations| α values.
        Shape: [num_edges, num_operations]
        """
        num_ops = len(self._get_operation_set())
        alphas = []
        
        # For each intermediate node
        for i in range(self.num_nodes):
            # Number of input edges = i + 2 (from input + previous nodes)
            n_inputs = i + 2
            alpha = nn.Parameter(torch.randn(n_inputs, num_ops) * 1e-3)
            alphas.append(alpha)
        
        return nn.ParameterList(alphas)
    
    def train_step(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Single DARTS training step.
        
        1. Update architecture α on validation set
        2. Update weights w on training set
        """
        # Get batches
        train_X, train_y = next(iter(train_loader))
        val_X, val_y = next(iter(val_loader))
        
        train_X, train_y = train_X.cuda(), train_y.cuda()
        val_X, val_y = val_X.cuda(), val_y.cuda()
        
        # Step 1: Update α
        self.optimizer_alpha.zero_grad()
        val_logits = self.supernet(val_X, self.alphas)
        val_loss = F.cross_entropy(val_logits, val_y)
        val_loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.alphas, self.grad_clip)
        self.optimizer_alpha.step()
        
        # Step 2: Update w
        self.optimizer_w.zero_grad()
        train_logits = self.supernet(train_X, self.alphas)
        train_loss = F.cross_entropy(train_logits, train_y)
        train_loss.backward()
        
        nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)
        self.optimizer_w.step()
        
        # Metrics
        train_acc = (train_logits.argmax(1) == train_y).float().mean()
        val_acc = (val_logits.argmax(1) == val_y).float().mean()
        
        self.step_count += 1
        
        return {
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'train_acc': train_acc.item(),
            'val_acc': val_acc.item()
        }
    
    def derive_architecture(self) -> ModelGraph:
        """
        Derive discrete architecture from continuous α.
        
        For each edge, select operation with highest α value.
        """
        graph = ModelGraph()
        
        # Input node
        input_node = GraphNode.create(operation='input')
        graph.add_node(input_node)
        
        # Intermediate nodes
        intermediate_nodes = []
        for i in range(self.num_nodes):
            node = GraphNode.create(operation='cell')
            
            # Get α for this node
            alpha = self.alphas[i].data.cpu().numpy()
            
            # For each input edge
            for j in range(i + 2):
                # Select best operation
                best_op_idx = alpha[j].argmax()
                best_op = self._get_operation_set()[best_op_idx]
                
                # Add edge with selected operation
                if best_op != 'none':
                    edge = GraphEdge(
                        source=intermediate_nodes[j] if j < len(intermediate_nodes) else input_node,
                        target=node,
                        operation=best_op
                    )
                    graph.add_edge(edge)
            
            intermediate_nodes.append(node)
            graph.add_node(node)
        
        # Output node
        output_node = GraphNode.create(operation='output')
        for node in intermediate_nodes:
            edge = GraphEdge(source=node, target=output_node)
            graph.add_edge(edge)
        graph.add_node(output_node)
        
        return graph
    
    def search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50
    ) -> ModelGraph:
        """
        Execute DARTS search.
        
        Returns:
            Best architecture found
        """
        logger.info(f"Starting DARTS search for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            metrics = self.train_step(train_loader, val_loader)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={metrics['train_loss']:.4f}, "
                    f"val_loss={metrics['val_loss']:.4f}, "
                    f"val_acc={metrics['val_acc']:.4f}"
                )
                
                # Log architecture weights
                self._log_architecture_weights()
        
        # Derive final architecture
        best_arch = self.derive_architecture()
        
        logger.info("DARTS search complete")
        return best_arch
    
    def _log_architecture_weights(self) -> None:
        """Log current α values for debugging."""
        for i, alpha in enumerate(self.alphas):
            weights = F.softmax(alpha, dim=-1).data.cpu().numpy()
            logger.debug(f"Node {i} architecture weights: {weights}")


class DARTSSupernet(nn.Module):
    """
    DARTS supernet with mixed operations.
    
    Each edge computes a weighted sum of all operations:
        output = Σ softmax(α_o) * operation_o(input)
    """
    
    def __init__(
        self,
        num_nodes: int,
        operations: List[str],
        num_classes: int,
        channels: int = 16
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.operations = operations
        
        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Mixed operations for each edge
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            node_ops = nn.ModuleList()
            for j in range(i + 2):  # Inputs from previous nodes
                mixed_op = MixedOp(channels, operations)
                node_ops.append(mixed_op)
            self.ops.append(node_ops)
        
        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
    
    def forward(self, x: torch.Tensor, alphas: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass with architecture parameters."""
        x = self.stem(x)
        
        states = [x, x]  # Two copies for initial inputs
        
        # Process each intermediate node
        for i in range(self.num_nodes):
            # Collect inputs from all previous nodes
            node_input = []
            for j, h in enumerate(states):
                # Apply mixed operation with architecture weights
                op_output = self.ops[i][j](h, alphas[i][j])
                node_input.append(op_output)
            
            # Concatenate all inputs
            s = sum(node_input)
            states.append(s)
        
        # Concatenate all intermediate nodes
        out = torch.cat(states[2:], dim=1)  # Skip initial two states
        
        # Global pooling and classification
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits


class MixedOp(nn.Module):
    """
    Mixed operation: weighted sum of all candidate operations.
    
    output = Σ softmax(α_i) * op_i(x)
    """
    
    def __init__(self, channels: int, operations: List[str]):
        super().__init__()
        
        self._ops = nn.ModuleList()
        for op_name in operations:
            op = self._create_operation(op_name, channels)
            self._ops.append(op)
    
    def _create_operation(self, op_name: str, C: int) -> nn.Module:
        """Create operation module."""
        if op_name == 'none':
            return Zero()
        elif op_name == 'skip_connect':
            return Identity()
        elif op_name == 'max_pool_3x3':
            return nn.MaxPool2d(3, stride=1, padding=1)
        elif op_name == 'avg_pool_3x3':
            return nn.AvgPool2d(3, stride=1, padding=1)
        elif op_name == 'sep_conv_3x3':
            return SepConv(C, C, 3, 1, 1)
        elif op_name == 'sep_conv_5x5':
            return SepConv(C, C, 5, 1, 2)
        elif op_name == 'dil_conv_3x3':
            return DilConv(C, C, 3, 1, 2, 2)
        elif op_name == 'dil_conv_5x5':
            return DilConv(C, C, 5, 1, 4, 2)
        else:
            raise ValueError(f"Unknown operation: {op_name}")
    
    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed operation.
        
        Args:
            x: Input tensor
            alpha: Architecture weights for this edge
        
        Returns:
            Weighted sum of operation outputs
        """
        weights = F.softmax(alpha, dim=0)
        
        return sum(w * op(x) for w, op in zip(weights, self._ops))
```

---

### 2. `gradient_based/enas_optimizer.py` (~2,000 LOC)

**`ENASOptimizer` class:**

```python
class ENASOptimizer(BaseOptimizer):
    """
    Efficient Neural Architecture Search (ENAS).
    
    Key Idea:
    - All child models share weights in a supergraph
    - RL controller samples architectures
    - Train controller to maximize validation accuracy
    
    Two-stage Training:
    1. Train shared weights on sampled architectures
    2. Train controller via REINFORCE
    
    Advantages:
    - 1000x faster than standard NAS
    - Weight sharing reduces training cost
    
    Config:
        controller_lr: Learning rate for controller (default: 3e-4)
        shared_lr: Learning rate for shared weights (default: 0.05)
        entropy_weight: Entropy regularization (default: 1e-4)
        baseline_decay: EMA decay for baseline (default: 0.99)
    """
    
    def __init__(self, search_space: SearchSpace, config: Dict[str, Any]):
        super().__init__(search_space, config)
        
        # Shared weights (supergraph)
        self.shared_model = ENASSharedModel(config)
        
        # Controller (RNN)
        self.controller = ENASController(
            num_layers=config.get('num_layers', 12),
            num_operations=len(self._get_operations()),
            hidden_size=config.get('controller_hidden', 100)
        )
        
        # Optimizers
        self.shared_optimizer = torch.optim.SGD(
            self.shared_model.parameters(),
            lr=config.get('shared_lr', 0.05),
            momentum=0.9,
            weight_decay=1e-4
        )
        
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(),
            lr=config.get('controller_lr', 3e-4)
        )
        
        self.entropy_weight = config.get('entropy_weight', 1e-4)
        self.baseline_decay = config.get('baseline_decay', 0.99)
        self.baseline = None
    
    def train_shared_model(self, data_loader: DataLoader) -> float:
        """
        Train shared weights on sampled architectures.
        
        Sample architectures from controller and train shared model.
        """
        self.shared_model.train()
        total_loss = 0.0
        
        for batch in data_loader:
            X, y = batch
            X, y = X.cuda(), y.cuda()
            
            # Sample architecture from controller
            arch, _, _ = self.controller.sample()
            
            # Forward pass with sampled architecture
            logits = self.shared_model(X, arch)
            loss = F.cross_entropy(logits, y)
            
            # Backward
            self.shared_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.shared_model.parameters(), 5.0)
            self.shared_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def train_controller(self, val_loader: DataLoader, num_samples: int = 10) -> float:
        """
        Train controller via REINFORCE.
        
        Sample architectures, evaluate on validation set,
        and update controller to maximize reward.
        """
        self.controller.train()
        self.shared_model.eval()
        
        total_loss = 0.0
        
        for _ in range(num_samples):
            # Sample architecture
            arch, log_probs, entropies = self.controller.sample()
            
            # Evaluate on validation set
            reward = self._evaluate_architecture(arch, val_loader)
            
            # Update baseline (exponential moving average)
            if self.baseline is None:
                self.baseline = reward
            else:
                self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
            
            # REINFORCE loss
            advantage = reward - self.baseline
            policy_loss = -log_probs * advantage
            entropy_loss = -entropies * self.entropy_weight
            
            loss = policy_loss + entropy_loss
            
            # Backward
            self.controller_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.controller.parameters(), 5.0)
            self.controller_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_samples
    
    def _evaluate_architecture(
        self,
        arch: List[int],
        val_loader: DataLoader
    ) -> float:
        """Evaluate architecture on validation set."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.cuda(), y.cuda()
                logits = self.shared_model(X, arch)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total
        return accuracy
    
    def search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 150
    ) -> ModelGraph:
        """Execute ENAS search."""
        logger.info(f"Starting ENAS search for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train shared model
            shared_loss = self.train_shared_model(train_loader)
            
            # Train controller
            controller_loss = self.train_controller(val_loader)
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"shared_loss={shared_loss:.4f}, "
                    f"controller_loss={controller_loss:.4f}, "
                    f"baseline={self.baseline:.4f}"
                )
        
        # Derive best architecture
        best_arch = self._derive_best_architecture(val_loader)
        
        return best_arch


class ENASController(nn.Module):
    """
    RNN controller for sampling architectures.
    
    At each layer, predict:
    1. Operation type
    2. Skip connections
    """
    
    def __init__(self, num_layers: int, num_operations: int, hidden_size: int = 100):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_operations = num_operations
        
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # Embedding for previous decisions
        self.g_emb = nn.Embedding(1, hidden_size)
        
        # Decoder for operations
        self.w_soft = nn.Linear(hidden_size, num_operations)
    
    def sample(self) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Sample architecture.
        
        Returns:
            arch: List of operation indices
            log_probs: Log probabilities of sampled operations
            entropies: Entropy of distributions
        """
        # Implementation details...
        pass
```

---

### 3. `gradient_based/operations.py` (~1,000 LOC)

**Operation primitives for DARTS/ENAS:**

```python
class SepConv(nn.Module):
    """Separable convolution."""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Dilated convolution."""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):
    """Zero operation (no connection)."""
    
    def forward(self, x):
        return x * 0.0


class Identity(nn.Module):
    """Identity operation (skip connection)."""
    
    def forward(self, x):
        return x
```

---

### 4. `gradient_based/utils.py` (~500 LOC)

**Utility functions:**

```python
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """
    Drop path (Stochastic Depth).
    
    Randomly drop entire paths during training.
    """
    if not training or drop_prob == 0.:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    
    return x.div(keep_prob) * random_tensor
```

---

## 🧪 Tests

**`test_darts.py`:**
```python
def test_darts_supernet_forward():
    """Test DARTS supernet forward pass."""
    supernet = DARTSSupernet(
        num_nodes=4,
        operations=['skip_connect', 'sep_conv_3x3'],
        num_classes=10
    )
    
    # Random architecture params
    alphas = [torch.randn(i+2, 2) for i in range(4)]
    
    x = torch.randn(2, 3, 32, 32)
    out = supernet(x, alphas)
    
    assert out.shape == (2, 10)


def test_darts_derive_architecture():
    """Test architecture derivation from α."""
    space = SearchSpace(...)
    darts = DARTSOptimizer(space, {})
    
    # Mock trained alphas
    darts.alphas = [torch.randn(i+2, 8) for i in range(4)]
    
    graph = darts.derive_architecture()
    
    assert len(graph.nodes) > 0
    assert graph.is_valid()
```

---

## ✅ Deliverables

- [ ] DARTS optimizer with bi-level optimization
- [ ] ENAS optimizer with RL controller
- [ ] Operation primitives (SepConv, DilConv, etc.)
- [ ] Architecture derivation logic
- [ ] GPU support and efficient training
- [ ] Tests showing convergence
- [ ] Example on CIFAR-10

---

**Next:** `03_multi_objective.md`
