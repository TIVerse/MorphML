#!/usr/bin/env python3
"""Example: Fast performance prediction without full training.

Demonstrates using predictors to accelerate NAS.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from morphml.core.dsl import Layer, SearchSpace
from morphml.meta_learning.predictors import (
    ProxyMetricPredictor,
    LearningCurvePredictor,
    EnsemblePredictor,
)
from morphml.logging_config import get_logger

logger = get_logger(__name__)


def create_search_space():
    """Create search space."""
    space = SearchSpace("cifar10_prediction")
    space.add_layers(
        Layer.input(shape=(3, 32, 32)),
        Layer.conv2d(filters=64, kernel_size=3),
        Layer.conv2d(filters=128, kernel_size=3),
        Layer.maxpool2d(pool_size=2),
        Layer.conv2d(filters=256, kernel_size=3),
        Layer.flatten(),
        Layer.dense(units=512),
        Layer.dropout(rate=0.5),
        Layer.output(units=10),
    )
    return space


def simulate_training(graph, num_epochs=10):
    """
    Simulate training (in real scenario, this would be actual training).
    
    Returns observed accuracies for first few epochs.
    """
    # Simulate learning curve
    base_acc = 0.3 + np.random.rand() * 0.2
    
    accuracies = []
    for epoch in range(1, num_epochs + 1):
        # Power law: acc = a - b * epoch^(-c)
        acc = 0.9 - 0.6 * (epoch ** -0.5) + np.random.randn() * 0.01
        acc = np.clip(acc, base_acc, 0.95)
        accuracies.append(acc)
    
    return accuracies


def demo_proxy_metric_predictor():
    """Demonstrate proxy metric predictor."""
    print("\n" + "="*80)
    print("DEMO 1: Proxy Metric Predictor")
    print("="*80)
    
    space = create_search_space()
    
    # Generate training data
    print("\nGenerating training data...")
    training_data = []
    for i in range(50):
        graph = space.sample()
        # Simulate actual accuracy (based on complexity)
        true_acc = 0.7 + (graph.count_parameters() / 10000000) * 0.2
        true_acc += np.random.randn() * 0.05
        true_acc = np.clip(true_acc, 0.6, 0.95)
        
        training_data.append((graph, true_acc))
    
    # Train predictor
    print("Training proxy metric predictor...")
    predictor = ProxyMetricPredictor()
    
    start_time = time.time()
    metrics = predictor.train(training_data, validation_split=0.2)
    train_time = time.time() - start_time
    
    print(f"Training complete in {train_time:.2f}s")
    if 'train_score' in metrics:
        print(f"  Train R²: {metrics['train_score']:.3f}")
        print(f"  Val R²: {metrics['val_score']:.3f}")
        print(f"  Top features: {metrics['top_features']}")
    
    # Test predictions
    print("\nTesting predictions...")
    test_graphs = [space.sample() for _ in range(5)]
    
    for i, graph in enumerate(test_graphs):
        pred = predictor.predict(graph)
        print(f"  Graph {i+1}: Predicted accuracy = {pred:.4f}")


def demo_learning_curve_predictor():
    """Demonstrate learning curve predictor."""
    print("\n" + "="*80)
    print("DEMO 2: Learning Curve Predictor")
    print("="*80)
    
    space = create_search_space()
    graph = space.sample()
    
    # Simulate training for 10 epochs
    print("\nSimulating early training (10 epochs)...")
    observed_acc = simulate_training(graph, num_epochs=10)
    
    print("Observed accuracies:")
    for epoch, acc in enumerate(observed_acc, 1):
        print(f"  Epoch {epoch:2d}: {acc:.4f}")
    
    # Predict final accuracy
    print("\nExtrapolating to final accuracy (200 epochs)...")
    
    predictor_power = LearningCurvePredictor(curve_type="power_law")
    predicted_final_power = predictor_power.predict_final_accuracy(
        observed_acc, final_epoch=200
    )
    
    predictor_exp = LearningCurvePredictor(curve_type="exponential")
    predicted_final_exp = predictor_exp.predict_final_accuracy(
        observed_acc, final_epoch=200
    )
    
    print(f"  Power law prediction: {predicted_final_power:.4f}")
    print(f"  Exponential prediction: {predicted_final_exp:.4f}")
    
    # Early stopping decision
    print("\nEarly stopping decision:")
    should_stop = predictor_power.should_early_stop(
        observed_acc, threshold=0.85
    )
    
    if should_stop:
        print("  ⚠️  Recommend STOP: predicted final < 0.85")
    else:
        print("  ✅ Continue training: predicted final >= 0.85")


def demo_ensemble_predictor():
    """Demonstrate ensemble predictor."""
    print("\n" + "="*80)
    print("DEMO 3: Ensemble Predictor")
    print("="*80)
    
    space = create_search_space()
    
    # Create training data
    print("\nPreparing ensemble...")
    training_data = [
        (space.sample(), 0.7 + np.random.rand() * 0.2)
        for _ in range(30)
    ]
    
    # Train individual predictors
    predictor1 = ProxyMetricPredictor()
    predictor1.train(training_data[:25])
    
    predictor2 = ProxyMetricPredictor()
    predictor2.train(training_data[5:])
    
    # Create ensemble
    ensemble = EnsemblePredictor(
        predictors=[predictor1, predictor2],
        weights=[0.6, 0.4],
        aggregation='weighted_average'
    )
    
    print("Ensemble created with 2 predictors (weights: 0.6, 0.4)")
    
    # Test prediction
    print("\nTesting ensemble prediction...")
    test_graph = space.sample()
    
    individual_preds = ensemble.get_individual_predictions(test_graph)
    
    print("Individual predictions:")
    for name, pred in individual_preds.items():
        if pred is not None:
            print(f"  {name}: {pred:.4f}")
    
    print(f"\n  Final ensemble: {individual_preds['ensemble']:.4f}")


def demo_speedup_comparison():
    """Compare prediction speed vs full evaluation."""
    print("\n" + "="*80)
    print("DEMO 4: Speed Comparison")
    print("="*80)
    
    space = create_search_space()
    
    # Train predictor
    training_data = [
        (space.sample(), 0.8)
        for _ in range(30)
    ]
    
    predictor = ProxyMetricPredictor()
    predictor.train(training_data)
    
    # Generate test architectures
    test_graphs = [space.sample() for _ in range(100)]
    
    # Timing: Proxy prediction
    print("\nPredicting 100 architectures...")
    start_time = time.time()
    predictions = predictor.batch_predict(test_graphs)
    pred_time = time.time() - start_time
    
    print(f"  Proxy prediction: {pred_time:.3f}s ({pred_time*1000/len(test_graphs):.2f}ms per arch)")
    
    # Simulate full training time
    full_train_time = len(test_graphs) * 300  # 5 min per arch
    
    print(f"  Full training (simulated): {full_train_time}s ({full_train_time/60:.1f} minutes)")
    print(f"  Speedup: {full_train_time/pred_time:.0f}x faster!")


def main():
    """Run all demos."""
    print("\n" + "🔮"*40)
    print(" "*20 + "Performance Prediction Examples")
    print("🔮"*40)
    
    try:
        demo_proxy_metric_predictor()
        demo_learning_curve_predictor()
        demo_ensemble_predictor()
        demo_speedup_comparison()
        
        print("\n" + "="*80)
        print("✅ All demos complete!")
        print("="*80)
        print("\nKey Takeaways:")
        print("  • Proxy metrics provide instant predictions")
        print("  • Learning curves enable early stopping")
        print("  • Ensembles combine multiple methods")
        print("  • 1000x+ speedup over full training")
        print("\n")
    
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
