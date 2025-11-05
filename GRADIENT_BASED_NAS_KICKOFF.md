# 🚀 Component 2: Gradient-Based NAS - Kickoff Plan

**Start Date:** November 5, 2025, 04:29 AM IST  
**Target Duration:** 2 weeks  
**Target LOC:** ~6,000  
**Dependencies:** PyTorch, Phase 1, Bayesian Optimization

---

## 📋 Overview

Implementing **gradient-based Neural Architecture Search** with two state-of-the-art methods:

### 1. **DARTS** (Differentiable Architecture Search)
- **Key Idea:** Continuous relaxation of discrete architecture space
- **Method:** Bi-level optimization (architecture α + weights w)
- **Speed:** ~1000x faster than discrete NAS
- **GPU:** Required for efficiency

### 2. **ENAS** (Efficient Neural Architecture Search)
- **Key Idea:** Weight sharing across child models
- **Method:** RL controller + shared supergraph
- **Speed:** 1000x faster than vanilla NAS with RL
- **GPU:** Required

---

## 🎯 Implementation Plan

### Phase 1: Foundation (Days 1-3)
**Files:**
1. `operations.py` (~1,000 LOC) - Operation primitives
   - SepConv, DilConv, Identity, Zero
   - Mixed operations for DARTS
   - Drop path, auxiliary losses

2. `utils.py` (~500 LOC) - Helper functions
   - Parameter counting
   - Drop path
   - Learning rate schedulers
   - GPU utilities

### Phase 2: DARTS (Days 4-7)
**Files:**
3. `darts.py` (~2,500 LOC) - DARTS implementation
   - `DARTSOptimizer` class
   - `DARTSSupernet` class
   - `MixedOp` class
   - Bi-level optimization
   - Architecture derivation

### Phase 3: ENAS (Days 8-11)
**Files:**
4. `enas.py` (~2,000 LOC) - ENAS implementation
   - `ENASOptimizer` class
   - `ENASController` (RNN)
   - `ENASSharedModel` class
   - REINFORCE training
   - Weight sharing

### Phase 4: Integration & Testing (Days 12-14)
**Files:**
5. `__init__.py` - Module exports
6. Tests - Unit and integration tests
7. Examples - CIFAR-10 demo
8. Documentation - Usage guide

---

## 🏗️ Architecture Overview

### DARTS Architecture
```
Input → Supernet (all operations) → Softmax(α) → Weighted Sum → Output
         ↑
    Architecture parameters α (learnable)
```

**Key Components:**
- **Supernet:** Contains all candidate operations
- **Architecture Parameters (α):** Continuous weights for operations
- **Mixed Operations:** Weighted sum of all ops per edge
- **Derivation:** argmax(α) to get discrete architecture

### ENAS Architecture
```
Input → Shared Supergraph → Sample(controller) → Child Model → Output
         ↑                      ↑
    Shared Weights        RL Controller (RNN)
```

**Key Components:**
- **Shared Supergraph:** All child models share weights
- **Controller:** RNN that samples architectures
- **REINFORCE:** Policy gradient to train controller
- **Baseline:** EMA of rewards for variance reduction

---

## 📦 Files to Create

| File | LOC | Priority | Description |
|------|-----|----------|-------------|
| `operations.py` | 1,000 | HIGH | Operation primitives |
| `utils.py` | 500 | HIGH | Helper functions |
| `darts.py` | 2,500 | HIGH | DARTS optimizer |
| `enas.py` | 2,000 | MEDIUM | ENAS optimizer |
| `__init__.py` | 100 | HIGH | Module exports |
| **TOTAL** | **6,100** | - | - |

---

## 🔧 Technical Challenges

### 1. **GPU Memory Management**
- **Challenge:** DARTS supernet is memory-intensive
- **Solution:** 
  - Mixed precision training
  - Gradient checkpointing
  - Batch size tuning

### 2. **Architecture Encoding**
- **Challenge:** Convert continuous α to discrete graph
- **Solution:**
  - argmax selection per edge
  - Pruning low-weight operations
  - Graph validation post-derivation

### 3. **Bi-Level Optimization (DARTS)**
- **Challenge:** Optimize α and w simultaneously
- **Solution:**
  - Alternating optimization
  - Separate optimizers (Adam for α, SGD for w)
  - Proper gradient clipping

### 4. **REINFORCE Variance (ENAS)**
- **Challenge:** High variance in policy gradients
- **Solution:**
  - EMA baseline
  - Entropy regularization
  - Multiple samples per update

### 5. **Operation Set Design**
- **Challenge:** Balance expressiveness vs efficiency
- **Solution:**
  - 8 operations (none, skip, 2 pools, 4 convs)
  - Separable convolutions for efficiency
  - Dilated convolutions for receptive field

---

## 🎨 Operation Primitives

### Implemented Operations:
1. **none** - Zero operation (no connection)
2. **skip_connect** - Identity (skip connection)
3. **max_pool_3x3** - Max pooling 3×3
4. **avg_pool_3x3** - Average pooling 3×3
5. **sep_conv_3x3** - Separable convolution 3×3
6. **sep_conv_5x5** - Separable convolution 5×5
7. **dil_conv_3x3** - Dilated convolution 3×3 (dilation=2)
8. **dil_conv_5x5** - Dilated convolution 5×5 (dilation=2)

### Operation Design Principles:
- **Efficient:** Separable convolutions reduce params
- **Diverse:** Mix of pooling, skip, and convolutions
- **Proven:** Used in DARTS/ENAS papers

---

## 📊 Expected Performance

### DARTS on CIFAR-10:
- **Search Time:** ~6 hours (single GPU)
- **Search Cost:** ~0.4 GPU-days
- **Final Architecture:** ~2.8M parameters
- **Test Accuracy:** ~97%+ (with proper training)

### ENAS on CIFAR-10:
- **Search Time:** ~12 hours (single GPU)
- **Search Cost:** ~0.5 GPU-days
- **Final Architecture:** ~4.6M parameters
- **Test Accuracy:** ~96%+

### vs Bayesian Optimization:
- **Speed:** 50-100x faster than BO
- **Quality:** Comparable or better architectures
- **Cost:** Requires GPU (BO doesn't)

---

## 🧪 Testing Strategy

### Unit Tests:
1. **Operations:**
   - Test each operation independently
   - Check shapes and gradients
   - Verify GPU compatibility

2. **Supernet:**
   - Test forward pass with random α
   - Verify gradient flow
   - Check memory usage

3. **Optimizers:**
   - Test ask/tell interface
   - Verify architecture derivation
   - Check weight updates

### Integration Tests:
1. **Mini-CIFAR:**
   - Run 5-epoch search
   - Verify architecture derivation
   - Check performance improvement

2. **Toy Dataset:**
   - Quick convergence test
   - Architecture validation
   - GPU vs CPU consistency

### Performance Tests:
1. **Memory Profiling:**
   - Track GPU memory usage
   - Identify bottlenecks
   - Optimize batch sizes

2. **Speed Benchmarking:**
   - Measure iterations/second
   - Compare DARTS vs ENAS
   - Profile operations

---

## 📝 Configuration Schemas

### DARTS Config:
```python
darts_config = {
    'learning_rate_w': 0.025,          # Weights learning rate
    'learning_rate_alpha': 3e-4,       # Architecture learning rate
    'momentum': 0.9,                   # SGD momentum
    'weight_decay': 3e-4,              # L2 regularization
    'grad_clip': 5.0,                  # Gradient clipping
    'num_nodes': 4,                    # Intermediate nodes
    'num_steps': 50,                   # Search epochs
    'channels': 16,                    # Initial channels
    'num_classes': 10,                 # Output classes
}
```

### ENAS Config:
```python
enas_config = {
    'controller_lr': 3e-4,             # Controller learning rate
    'shared_lr': 0.05,                 # Shared weights learning rate
    'entropy_weight': 1e-4,            # Entropy regularization
    'baseline_decay': 0.99,            # EMA decay for baseline
    'num_layers': 12,                  # Number of layers
    'controller_hidden': 100,          # Controller hidden size
    'num_steps': 150,                  # Search epochs
}
```

---

## 🚦 Success Criteria

### Minimum Viable:
- [ ] Operations module complete
- [ ] DARTS optimizer functional
- [ ] Architecture derivation works
- [ ] Can run on GPU
- [ ] Basic tests passing

### Complete Implementation:
- [ ] All operations implemented
- [ ] DARTS optimizer complete
- [ ] ENAS optimizer complete
- [ ] GPU-optimized
- [ ] Full test coverage >75%
- [ ] CIFAR-10 example working
- [ ] Documentation complete

---

## ⚠️ Known Limitations

### DARTS:
1. **Memory:** Requires significant GPU memory (~8GB)
2. **Stability:** Can collapse to skip connections
3. **Hyperparameters:** Sensitive to learning rates

### ENAS:
1. **Variance:** REINFORCE has high variance
2. **Convergence:** Slower than DARTS
3. **Complexity:** More components to tune

### Both:
1. **GPU Dependency:** Cannot run efficiently on CPU
2. **Search Space:** Limited to predefined operations
3. **Transferability:** Architectures may not transfer well

---

## 🎯 Integration with Phase 1 & 2

### Compatible with Bayesian Optimization:
- Can use DARTS/ENAS to initialize BO search
- Hybrid: DARTS for coarse search, BO for fine-tuning
- Compare sample efficiency

### Uses Existing Infrastructure:
- SearchSpace from Phase 1 DSL
- ModelGraph for architecture representation
- Logging and tracking systems

---

## 📚 References

### DARTS:
- Liu et al. "DARTS: Differentiable Architecture Search" (ICLR 2019)
- GitHub: quark0/darts

### ENAS:
- Pham et al. "Efficient Neural Architecture Search via Parameter Sharing" (ICML 2018)
- GitHub: melodyguan/enas

### Implementation Guides:
- PyTorch DARTS implementation
- NAS-Bench-201 codebase
- AutoDL-Projects repository

---

## 🚀 Let's Begin!

**Starting with:** `operations.py` - Operation primitives  
**Estimated Time:** 2-3 hours for operations module  
**Next:** `utils.py` then `darts.py`

**Ready to implement gradient-based NAS! 🔥**

---

**Prepared by:** Cascade (AI Assistant)  
**Date:** November 5, 2025, 04:29 AM IST  
**Status:** Ready to Start
