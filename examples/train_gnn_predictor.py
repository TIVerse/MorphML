"""
Train GNN Performance Predictor on Historical Data

This script demonstrates how to train the GNN predictor on past experiment data
to enable fast performance prediction for new architectures.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.core.dsl import SearchSpace, Layer
from morphml.meta_learning.predictors.gnn_predictor import GNNPredictor
from morphml.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def generate_synthetic_training_data(
    search_space: SearchSpace,
    num_samples: int = 1000,
) -> List[Tuple[ModelGraph, float]]:
    """
    Generate synthetic training data for demonstration.
    
    In production, this should be replaced with actual evaluated architectures
    from your experiment database.
    
    Args:
        search_space: Search space to sample from
        num_samples: Number of samples to generate
    
    Returns:
        List of (architecture, accuracy) pairs
    """
    logger.info(f"Generating {num_samples} synthetic training samples...")
    
    data = []
    
    for i in range(num_samples):
        # Sample random architecture
        graph = search_space.sample()
        
        # Simulate accuracy based on graph properties
        # (In reality, this would be actual trained accuracy)
        num_nodes = len(graph.nodes)
        
        # Heuristic: deeper networks tend to perform better (to a point)
        depth_score = 1.0 - abs(num_nodes - 15) / 20.0
        depth_score = max(0.3, min(1.0, depth_score))
        
        # Add randomness
        noise = random.gauss(0, 0.05)
        
        # Simulated accuracy
        accuracy = 0.6 + 0.3 * depth_score + noise
        accuracy = max(0.5, min(0.98, accuracy))
        
        data.append((graph, accuracy))
        
        if (i + 1) % 200 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")
    
    logger.info(f"Generated {len(data)} training samples")
    return data


def load_real_training_data(data_path: str) -> List[Tuple[ModelGraph, float]]:
    """
    Load real training data from experiment database.
    
    Expected format: JSON file with list of:
    {
        "architecture": {...},  # ModelGraph JSON
        "accuracy": 0.92
    }
    
    Args:
        data_path: Path to training data JSON
    
    Returns:
        List of (architecture, accuracy) pairs
    """
    logger.info(f"Loading training data from {data_path}")
    
    with open(data_path) as f:
        records = json.load(f)
    
    data = []
    for record in records:
        graph = ModelGraph.from_dict(record["architecture"])
        accuracy = record["accuracy"]
        data.append((graph, accuracy))
    
    logger.info(f"Loaded {len(data)} training samples")
    return data


def train_predictor(
    train_data: List[Tuple[ModelGraph, float]],
    config: dict,
    output_dir: Path,
):
    """
    Train GNN predictor.
    
    Args:
        train_data: Training data
        config: Predictor configuration
        output_dir: Directory to save model
    """
    logger.info("Initializing GNN predictor...")
    
    predictor = GNNPredictor(config)
    
    # Train
    logger.info("Starting training...")
    
    training_stats = predictor.train(
        train_data=train_data,
        num_epochs=config.get("num_epochs", 100),
        batch_size=config.get("batch_size", 32),
        early_stopping_patience=config.get("early_stopping_patience", 20),
    )
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gnn_predictor.pt"
    predictor.save(str(model_path))
    
    # Save training stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Training stats saved to {stats_path}")
    
    return predictor, training_stats


def evaluate_predictor(
    predictor: GNNPredictor,
    test_data: List[Tuple[ModelGraph, float]],
) -> dict:
    """
    Evaluate predictor on test set.
    
    Args:
        predictor: Trained predictor
        test_data: Test data
    
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating on {len(test_data)} test samples...")
    
    predictions = []
    actuals = []
    
    for graph, actual_acc in test_data:
        pred_acc = predictor.predict(graph)
        predictions.append(pred_acc)
        actuals.append(actual_acc)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Correlation
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    # Accuracy within threshold
    threshold = 0.05
    within_threshold = np.mean(np.abs(predictions - actuals) < threshold)
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "accuracy_within_5pct": float(within_threshold),
        "num_samples": len(test_data),
    }
    
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info(f"  MAE:                {mae:.4f}")
    logger.info(f"  RMSE:               {rmse:.4f}")
    logger.info(f"  Correlation:        {correlation:.4f}")
    logger.info(f"  Within 5% accuracy: {within_threshold:.2%}")
    logger.info("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN Performance Predictor"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to real training data (JSON). If not provided, uses synthetic data.",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_models",
        help="Output directory for trained model",
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (if not using real data)",
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="GNN hidden dimension",
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of GNN layers",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(verbose=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("GNN Performance Predictor Training")
    logger.info("=" * 60)
    
    # Load or generate data
    if args.data_path:
        data = load_real_training_data(args.data_path)
    else:
        logger.info("No real data provided, generating synthetic data...")
        
        # Create search space
        search_space = SearchSpace(
            name="demo_space",
            layers=[
                Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
                Layer.maxpool(pool_size=[2]),
                Layer.dense(units=[128, 256, 512]),
                Layer.dropout(rate=[0.2, 0.3, 0.5]),
            ],
        )
        
        data = generate_synthetic_training_data(
            search_space, args.num_samples
        )
    
    # Train/test split
    split_idx = int(len(data) * (1 - args.test_split))
    random.shuffle(data)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")
    
    # Training configuration
    config = {
        "node_feature_dim": 128,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": 4,
        "dropout": 0.3,
        "lr": args.learning_rate,
        "weight_decay": 1e-5,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": 20,
    }
    
    # Train
    output_dir = Path(args.output_dir)
    predictor, training_stats = train_predictor(
        train_data, config, output_dir
    )
    
    # Evaluate
    test_metrics = evaluate_predictor(predictor, test_data)
    
    # Save test metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"\nTest metrics saved to {metrics_path}")
    logger.info("\nTraining complete! ✓")
    
    # Usage instructions
    logger.info("\n" + "=" * 60)
    logger.info("To use the trained predictor:")
    logger.info("=" * 60)
    logger.info("from morphml.meta_learning.predictors import GNNPredictor")
    logger.info("from morphml.core.graph import ModelGraph")
    logger.info("")
    logger.info("predictor = GNNPredictor()")
    logger.info(f"predictor.load('{output_dir / 'gnn_predictor.pt'}')")
    logger.info("")
    logger.info("# Predict accuracy for new architecture")
    logger.info("predicted_acc = predictor.predict(new_graph)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
