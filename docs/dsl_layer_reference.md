# DSL Layer Reference

Complete reference for all layer types available in the MorphML DSL.

## Table of Contents

- [Convolutional Layers](#convolutional-layers)
- [Dense Layers](#dense-layers)
- [Pooling Layers](#pooling-layers)
- [Normalization Layers](#normalization-layers)
- [Regularization Layers](#regularization-layers)
- [Activation Layers](#activation-layers)
- [Utility Layers](#utility-layers)
- [Input/Output Layers](#inputoutput-layers)

---

## Convolutional Layers

### Layer.conv2d()

2D Convolutional layer for processing spatial data (images).

**Parameters:**
- `filters` (int | List[int]): Number of output filters/channels
- `kernel_size` (int | List[int], default=3): Size of the convolutional kernel
- `strides` (int | List[int], optional): Stride of the convolution
- `padding` (str, default="same"): Padding mode ("same" or "valid")
- `activation` (str, optional): Activation function to apply
- `**kwargs`: Additional parameters

**Example:**
```python
from morphml.core.dsl import Layer, SearchSpace

space = SearchSpace("cnn_search")
space.add_layers(
    Layer.input(shape=(3, 32, 32)),
    # Single value - fixed during search
    Layer.conv2d(filters=64, kernel_size=3),
    # Multiple values - sampled during search
    Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5, 7]),
)
```

**Use Cases:**
- Image classification
- Object detection
- Semantic segmentation
- Feature extraction from spatial data

---

## Dense Layers

### Layer.dense()

Fully connected (dense) layer.

**Parameters:**
- `units` (int | List[int]): Number of output units/neurons
- `activation` (str, optional): Activation function
- `use_bias` (bool, default=True): Whether to use bias
- `**kwargs`: Additional parameters

**Example:**
```python
space.add_layers(
    Layer.flatten(),
    Layer.dense(units=[128, 256, 512]),  # Search over different sizes
    Layer.relu(),
    Layer.dropout(rate=0.5),
    Layer.dense(units=10),  # Output layer
)
```

**Use Cases:**
- Classification heads
- Regression outputs
- Feature transformation
- Bottleneck layers

---

## Pooling Layers

### Layer.maxpool()

Max pooling layer - downsamples by taking maximum value in each window.

**Parameters:**
- `pool_size` (int | List[int], default=2): Size of pooling window
- `strides` (int | List[int], optional): Stride (defaults to pool_size)
- `padding` (str, default="valid"): Padding mode
- `**kwargs`: Additional parameters

**Example:**
```python
space.add_layers(
    Layer.conv2d(filters=64, kernel_size=3),
    Layer.relu(),
    Layer.maxpool(pool_size=2),  # Reduces spatial dimensions by 2x
)
```

**Use Cases:**
- Spatial downsampling
- Translation invariance
- Reducing computation
- Feature aggregation

### Layer.avgpool()

Average pooling layer - downsamples by taking average value in each window.

**Parameters:**
- `pool_size` (int | List[int], default=2): Size of pooling window
- `strides` (int | List[int], optional): Stride
- `padding` (str, default="valid"): Padding mode
- `**kwargs`: Additional parameters

**Example:**
```python
space.add_layers(
    Layer.conv2d(filters=128, kernel_size=3),
    Layer.avgpool(pool_size=2),  # Smoother downsampling than maxpool
)
```

**Use Cases:**
- Smooth downsampling
- Global average pooling before classification
- Reducing spatial dimensions

---

## Normalization Layers

### Layer.batchnorm()

Batch normalization layer - normalizes activations across the batch.

**Parameters:**
- `**kwargs`: Additional parameters

**Example:**
```python
space.add_layers(
    Layer.conv2d(filters=64, kernel_size=3),
    Layer.batchnorm(),  # Normalize before activation
    Layer.relu(),
)
```

**Benefits:**
- Faster training convergence
- Reduces internal covariate shift
- Acts as regularization
- Allows higher learning rates

**Use Cases:**
- After convolutional layers
- Before or after activations
- Deep networks (>10 layers)

---

## Regularization Layers

### Layer.dropout()

Dropout layer - randomly drops units during training to prevent overfitting.

**Parameters:**
- `rate` (float | List[float], default=0.5): Dropout probability (0.0 to 1.0)
- `**kwargs`: Additional parameters

**Example:**
```python
space.add_layers(
    Layer.dense(units=512),
    Layer.relu(),
    Layer.dropout(rate=[0.3, 0.5, 0.7]),  # Search over dropout rates
    Layer.dense(units=10),
)
```

**Recommended Rates:**
- 0.2-0.3: Light regularization
- 0.5: Standard regularization
- 0.7-0.8: Heavy regularization (use cautiously)

**Use Cases:**
- Preventing overfitting
- Ensemble-like behavior
- After dense layers
- In deep networks

---

## Activation Layers

### Layer.relu()

Rectified Linear Unit activation: `f(x) = max(0, x)`

**Example:**
```python
space.add_layers(
    Layer.conv2d(filters=64, kernel_size=3),
    Layer.relu(),  # Most common activation
)
```

**Characteristics:**
- Fast computation
- No vanishing gradient for positive values
- Can cause "dying ReLU" problem
- Most popular activation function

### Layer.sigmoid()

Sigmoid activation: `f(x) = 1 / (1 + e^(-x))`

**Example:**
```python
space.add_layers(
    Layer.dense(units=1),
    Layer.sigmoid(),  # For binary classification
)
```

**Use Cases:**
- Binary classification output
- Probability outputs (0 to 1)
- Gate mechanisms

### Layer.tanh()

Hyperbolic tangent activation: `f(x) = tanh(x)`

**Example:**
```python
space.add_layers(
    Layer.dense(units=128),
    Layer.tanh(),  # Outputs in range [-1, 1]
)
```

**Characteristics:**
- Zero-centered (unlike sigmoid)
- Outputs in range [-1, 1]
- Can suffer from vanishing gradients

### Layer.softmax()

Softmax activation - converts logits to probability distribution.

**Example:**
```python
space.add_layers(
    Layer.dense(units=10),
    Layer.softmax(),  # For multi-class classification
)
```

**Use Cases:**
- Multi-class classification output
- Attention mechanisms
- Probability distributions

---

## Utility Layers

### Layer.flatten()

Flattens multi-dimensional input to 1D (excluding batch dimension).

**Parameters:**
- `**kwargs`: Additional parameters

**Example:**
```python
space.add_layers(
    Layer.conv2d(filters=128, kernel_size=3),
    Layer.maxpool(pool_size=2),
    Layer.flatten(),  # Convert (128, H, W) -> (128*H*W,)
    Layer.dense(units=256),
)
```

**Use Cases:**
- Transition from convolutional to dense layers
- Before classification head
- Feature vector extraction

**Note:** Essential when connecting convolutional layers to dense layers.

---

## Input/Output Layers

### Layer.input()

Input layer - defines the input shape for the network.

**Parameters:**
- `shape` (tuple): Input shape (excluding batch dimension)
- `**kwargs`: Additional parameters

**Example:**
```python
# Image input (channels, height, width)
Layer.input(shape=(3, 32, 32))

# Tabular input
Layer.input(shape=(100,))

# Sequence input
Layer.input(shape=(50, 128))  # (sequence_length, features)
```

**Shape Formats:**
- Images: `(channels, height, width)` for PyTorch
- Vectors: `(features,)`
- Sequences: `(timesteps, features)`

### Layer.output()

Output layer - defines the final output.

**Parameters:**
- `units` (int): Number of output units
- `activation` (str, optional): Output activation
- `**kwargs`: Additional parameters

**Example:**
```python
# Binary classification
Layer.output(units=1, activation='sigmoid')

# Multi-class classification
Layer.output(units=10, activation='softmax')

# Regression
Layer.output(units=1)  # No activation
```

---

## Complete Example

Here's a complete example combining multiple layer types:

```python
from morphml.core.dsl import Layer, SearchSpace

# Define search space for image classification
space = SearchSpace("image_classifier")

space.add_layers(
    # Input
    Layer.input(shape=(3, 32, 32)),
    
    # First conv block
    Layer.conv2d(filters=[32, 64], kernel_size=3),
    Layer.batchnorm(),
    Layer.relu(),
    Layer.maxpool(pool_size=2),
    
    # Second conv block
    Layer.conv2d(filters=[64, 128], kernel_size=3),
    Layer.batchnorm(),
    Layer.relu(),
    Layer.maxpool(pool_size=2),
    
    # Third conv block
    Layer.conv2d(filters=[128, 256], kernel_size=3),
    Layer.relu(),
    
    # Classification head
    Layer.flatten(),
    Layer.dense(units=[256, 512]),
    Layer.relu(),
    Layer.dropout(rate=[0.3, 0.5]),
    Layer.dense(units=10),
    Layer.softmax(),
)

# Sample an architecture
architecture = space.sample()
print(f"Sampled architecture with {len(architecture.nodes)} layers")
```

---

## Best Practices

### 1. Layer Ordering

**Recommended order for convolutional blocks:**
```python
Conv2D -> BatchNorm -> Activation -> Pooling
```

**Recommended order for dense blocks:**
```python
Dense -> Activation -> Dropout
```

### 2. Search Space Design

**Good practice:**
```python
# Provide meaningful ranges
Layer.conv2d(filters=[32, 64, 128, 256])  # Powers of 2
Layer.dense(units=[128, 256, 512, 1024])
```

**Avoid:**
```python
# Too many options slow down search
Layer.conv2d(filters=list(range(16, 512)))  # Too granular
```

### 3. Activation Functions

- **ReLU**: Default choice for hidden layers
- **Sigmoid**: Binary classification output
- **Softmax**: Multi-class classification output
- **Tanh**: When zero-centered outputs are needed

### 4. Regularization

- Use **BatchNorm** for deep networks (>10 layers)
- Use **Dropout** (0.5) for dense layers
- Use **Dropout** (0.2-0.3) for convolutional layers (less common)

### 5. Architecture Patterns

**ResNet-style:**
```python
# Identity shortcuts help gradient flow
Layer.conv2d(filters=64, kernel_size=3)
Layer.batchnorm()
Layer.relu()
Layer.conv2d(filters=64, kernel_size=3)
Layer.batchnorm()
# Add skip connection in implementation
```

**VGG-style:**
```python
# Multiple small convolutions
Layer.conv2d(filters=64, kernel_size=3)
Layer.relu()
Layer.conv2d(filters=64, kernel_size=3)
Layer.relu()
Layer.maxpool(pool_size=2)
```

---

## Tips for Neural Architecture Search

1. **Start Simple**: Begin with a small search space and expand
2. **Use Constraints**: Add parameter/depth constraints to guide search
3. **Layer Combinations**: Test different activation and normalization combinations
4. **Pooling Strategy**: Experiment with maxpool vs avgpool
5. **Dropout Rates**: Search over dropout rates for regularization

---

## See Also

- [Search Space Guide](search_space_guide.md)
- [Constraints Reference](constraints_reference.md)
- [Optimizer Guide](optimizer_guide.md)
- [Examples](../examples/)
