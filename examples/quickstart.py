"""
MorphML Quickstart Example

This example demonstrates basic usage:
1. Define search space
2. Configure optimizer
3. Run experiment

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML

Usage:
    morphml run examples/quickstart.py --output-dir ./results
"""

from morphml.core.dsl import Layer, SearchSpace

# Define what architectures to search over
search_space = SearchSpace("quickstart")

# Add layers to search space
search_space.add_layers(
    Layer.input(shape=(3, 32, 32)),
    Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2),
    Layer.conv2d(filters=[64, 128, 256], kernel_size=3),
    Layer.relu(),
    Layer.batchnorm(),
    Layer.maxpool(pool_size=2),
    Layer.dense(units=[128, 256, 512]),
    Layer.dropout(rate=[0.1, 0.3, 0.5]),
    Layer.dense(units=[64, 128, 256]),
    Layer.output(units=10)
)

# Configure genetic algorithm
optimizer_config = {
    'population_size': 20,
    'num_generations': 50,
    'elite_size': 2,
    'mutation_rate': 0.15,
    'crossover_rate': 0.7,
    'selection_strategy': 'tournament',
    'tournament_size': 3
}

# Budget (optional - limits total evaluations)
max_evaluations = 500

# Expected results:
# - Best architecture saved to ./results/best_model.json
# - PyTorch code in ./results/best_model_pytorch.py
# - Keras code in ./results/best_model_keras.py
# - Training history in ./results/history.json
