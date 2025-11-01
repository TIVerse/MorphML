"""
Simple MorphML Example

A minimal example showing the basics of MorphML.
Perfect for beginners!

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML

Usage:
    morphml run examples/simple_example.py
"""

from morphml.core.dsl import Layer, SearchSpace

# Create a simple search space
search_space = SearchSpace("simple")

# Define a basic neural network search space
search_space.add_layers(
    Layer.input(shape=(784,)),           # MNIST-like input (28x28 flattened)
    Layer.dense(units=[64, 128, 256]),   # Hidden layer with 64, 128, or 256 units
    Layer.relu(),                         # ReLU activation
    Layer.dropout(rate=[0.2, 0.5]),      # Dropout with rate 0.2 or 0.5
    Layer.dense(units=[32, 64, 128]),    # Another hidden layer
    Layer.relu(),
    Layer.output(units=10)                # Output layer (10 classes)
)

# Simple optimizer configuration
optimizer_config = {
    'population_size': 10,
    'num_generations': 20,
    'mutation_rate': 0.1,
    'crossover_rate': 0.6
}

# Run for 100 evaluations
max_evaluations = 100
