"""
CIFAR-10 Architecture Search Example

This example demonstrates a more advanced search space configuration
for CIFAR-10 image classification.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML

Usage:
    morphml run examples/cifar10_example.py --output-dir ./cifar10_results --verbose
"""

from morphml.core.dsl import Layer, SearchSpace
from morphml.constraints import MaxParametersConstraint, DepthConstraint

# Define search space for CIFAR-10 (32x32x3 images, 10 classes)
search_space = SearchSpace("cifar10_search")

# Input layer
search_space.add_layers(
    Layer.input(shape=(3, 32, 32)),
)

# First convolutional block
search_space.add_layers(
    Layer.conv2d(filters=[32, 64, 96], kernel_size=[3, 5], padding="same"),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.conv2d(filters=[32, 64, 96], kernel_size=3, padding="same"),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2, strides=2),
    Layer.dropout(rate=[0.1, 0.2, 0.3]),
)

# Second convolutional block
search_space.add_layers(
    Layer.conv2d(filters=[64, 128, 192], kernel_size=[3, 5], padding="same"),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.conv2d(filters=[64, 128, 192], kernel_size=3, padding="same"),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2, strides=2),
    Layer.dropout(rate=[0.2, 0.3, 0.4]),
)

# Optional third block
search_space.add_layers(
    Layer.conv2d(filters=[128, 256, 384], kernel_size=3, padding="same"),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.avgpool(pool_size=2, strides=2),
)

# Dense layers
search_space.add_layers(
    Layer.flatten(),
    Layer.dense(units=[256, 512, 768, 1024]),
    Layer.relu(),
    Layer.dropout(rate=[0.3, 0.4, 0.5]),
    Layer.dense(units=[128, 256, 512]),
    Layer.relu(),
    Layer.dropout(rate=[0.3, 0.4, 0.5]),
    Layer.output(units=10, activation="softmax"),
)

# Add constraints
search_space.add_constraint(MaxParametersConstraint(max_params=5000000))  # Max 5M parameters
search_space.add_constraint(DepthConstraint(min_depth=8, max_depth=20))  # 8-20 layers

# Optimizer configuration
optimizer_config = {
    "population_size": 30,
    "num_generations": 100,
    "elite_size": 3,
    "mutation_rate": 0.2,
    "crossover_rate": 0.8,
    "selection_strategy": "tournament",
    "tournament_size": 5,
}

# Evaluation budget
max_evaluations = 1000

# Notes:
# - This search space is designed for CIFAR-10 classification
# - Constraints ensure models are trainable on modest hardware
# - Higher population and generation count for better exploration
# - Tournament selection with size 5 for stronger selection pressure
