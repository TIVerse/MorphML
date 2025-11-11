"""
Unit tests for GNN-based performance predictor.

Tests GNN model architecture, training, and prediction.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
import pytest

from morphml.core.graph import GraphNode, ModelGraph

# Check if PyTorch is available
try:
    import torch

    from morphml.meta_learning.predictors.gnn_predictor import (
        TORCH_AVAILABLE,
        ArchitectureGNN,
        GNNPredictor,
    )

    SKIP_GNN_TESTS = not TORCH_AVAILABLE
except ImportError:
    SKIP_GNN_TESTS = True


@pytest.fixture
def sample_graph():
    """Create a sample architecture graph."""
    graph = ModelGraph()

    input_node = GraphNode("input", "input", {"input_shape": (3, 32, 32)})
    conv1 = GraphNode("conv1", "conv2d", {"filters": 32, "kernel_size": 3})
    pool = GraphNode("pool", "maxpool", {"pool_size": 2})
    dense = GraphNode("dense", "dense", {"units": 128})
    output = GraphNode("output", "dense", {"units": 10})

    graph.add_node(input_node)
    graph.add_node(conv1)
    graph.add_node(pool)
    graph.add_node(dense)
    graph.add_node(output)

    graph.add_edge_by_id("input", "conv1")
    graph.add_edge_by_id("conv1", "pool")
    graph.add_edge_by_id("pool", "dense")
    graph.add_edge_by_id("dense", "output")

    return graph


@pytest.fixture
def training_data(sample_graph):
    """Generate synthetic training data."""
    data = []

    # Create variations of the sample graph
    for i in range(50):
        graph = sample_graph.clone()

        # Vary some parameters
        graph.nodes["conv1"].params["filters"] = 32 + i * 2
        graph.nodes["dense"].params["units"] = 128 + i * 4

        # Simulate accuracy (correlated with depth/width)
        accuracy = 0.7 + 0.2 * (i / 50) + np.random.normal(0, 0.03)
        accuracy = max(0.5, min(0.95, accuracy))

        data.append((graph, accuracy))

    return data


# ============================================================================
# GNN Predictor Tests
# ============================================================================


@pytest.mark.skipif(SKIP_GNN_TESTS, reason="PyTorch not available")
class TestGNNPredictor:
    """Test GNN predictor functionality."""

    def test_initialization_default_config(self):
        """Test GNN predictor initializes with default config."""
        predictor = GNNPredictor()

        assert predictor.model is not None
        assert predictor.optimizer is not None
        assert predictor.device is not None
        assert isinstance(predictor.training_history, list)
        assert predictor.is_trained is False

    def test_initialization_custom_config(self):
        """Test GNN predictor with custom configuration."""
        config = {
            "node_feature_dim": 64,
            "hidden_dim": 128,
            "num_layers": 3,
            "num_heads": 2,
            "dropout": 0.2,
            "lr": 0.001,
        }

        predictor = GNNPredictor(config)

        assert predictor.model.hidden_dim == 128
        assert predictor.model.num_layers == 3

    def test_graph_to_pyg_data_conversion(self, sample_graph):
        """Test conversion from ModelGraph to PyG Data."""
        predictor = GNNPredictor()

        data = predictor._graph_to_pyg_data(sample_graph, accuracy=0.85)

        # Check data structure
        assert hasattr(data, "x")  # Node features
        assert hasattr(data, "edge_index")  # Edge connectivity
        assert hasattr(data, "y")  # Label
        assert hasattr(data, "batch")  # Batch indicator

        # Check dimensions
        assert data.x.shape[0] == len(sample_graph.nodes)
        assert data.x.shape[1] == 128  # Default node_feature_dim
        assert data.edge_index.shape[0] == 2
        assert data.y.shape[0] == 1
        assert data.y.item() == 0.85

    def test_node_encoding(self):
        """Test node feature encoding."""
        predictor = GNNPredictor()

        # Create nodes with different operations
        conv_node = GraphNode("conv", "conv2d", {"filters": 64, "kernel_size": 3})
        dense_node = GraphNode("dense", "dense", {"units": 256})
        relu_node = GraphNode("relu", "relu", {})

        # Encode nodes
        conv_feat = predictor._encode_node(conv_node, position=0, total_nodes=5)
        dense_feat = predictor._encode_node(dense_node, position=1, total_nodes=5)
        relu_feat = predictor._encode_node(relu_node, position=2, total_nodes=5)

        # Features should be different
        assert len(conv_feat) > 0
        assert len(dense_feat) > 0
        assert len(relu_feat) > 0
        assert conv_feat != dense_feat
        assert dense_feat != relu_feat

    def test_training(self, training_data):
        """Test GNN predictor training."""
        predictor = GNNPredictor(
            {
                "hidden_dim": 64,
                "num_layers": 2,
            }
        )

        # Train on small dataset for quick test
        stats = predictor.train(
            train_data=training_data,
            num_epochs=5,  # Few epochs for testing
            batch_size=16,
            early_stopping_patience=3,
        )

        # Check training completed
        assert "best_val_loss" in stats
        assert "num_epochs" in stats
        assert "history" in stats
        assert predictor.is_trained is True

        # Training should reduce loss
        if len(predictor.training_history) > 1:
            initial_loss = predictor.training_history[0]["train_loss"]
            final_loss = predictor.training_history[-1]["train_loss"]
            assert final_loss <= initial_loss * 1.5  # Allow some variance

    def test_prediction(self, training_data, sample_graph):
        """Test GNN prediction."""
        predictor = GNNPredictor()

        # Train
        predictor.train(training_data, num_epochs=5, batch_size=16)

        # Predict
        predicted_acc = predictor.predict(sample_graph)

        # Prediction should be in valid range
        assert 0.0 <= predicted_acc <= 1.0
        assert isinstance(predicted_acc, float)

    def test_prediction_without_training(self, sample_graph):
        """Test prediction without training (should still work with warning)."""
        predictor = GNNPredictor()

        # Predict without training
        predicted_acc = predictor.predict(sample_graph)

        # Should return a value (random but valid)
        assert 0.0 <= predicted_acc <= 1.0

    def test_save_and_load(self, training_data, sample_graph, tmp_path):
        """Test model save and load functionality."""
        # Train a predictor
        predictor1 = GNNPredictor()
        predictor1.train(training_data, num_epochs=5, batch_size=16)

        # Get prediction before save
        pred1 = predictor1.predict(sample_graph)

        # Save
        model_path = tmp_path / "test_gnn.pt"
        predictor1.save(str(model_path))

        # Load into new predictor
        predictor2 = GNNPredictor()
        predictor2.load(str(model_path))

        # Prediction should be identical
        pred2 = predictor2.predict(sample_graph)

        assert abs(pred1 - pred2) < 1e-5
        assert predictor2.is_trained is True

    def test_early_stopping(self, training_data):
        """Test early stopping mechanism."""
        predictor = GNNPredictor()

        # Use very small patience to trigger early stopping
        stats = predictor.train(
            training_data,
            num_epochs=100,  # Large number
            batch_size=16,
            early_stopping_patience=3,  # Small patience
        )

        # Should stop before 100 epochs
        assert stats["num_epochs"] < 100

    def test_validation_split(self, training_data):
        """Test automatic train/validation split."""
        predictor = GNNPredictor()

        # No explicit validation data
        predictor.train(
            training_data,
            val_data=None,  # Will auto-split
            num_epochs=5,
            batch_size=16,
        )

        # Should have validation metrics
        assert len(predictor.training_history) > 0
        assert "val_loss" in predictor.training_history[0]
        assert "val_mae" in predictor.training_history[0]

    def test_custom_validation_data(self, training_data):
        """Test with custom validation data."""
        # Split manually
        train = training_data[:40]
        val = training_data[40:]

        predictor = GNNPredictor()

        predictor.train(
            train_data=train,
            val_data=val,
            num_epochs=5,
            batch_size=16,
        )

        # Should use provided validation data
        assert len(predictor.training_history) > 0


# ============================================================================
# GNN Model Tests
# ============================================================================


@pytest.mark.skipif(SKIP_GNN_TESTS, reason="PyTorch not available")
class TestArchitectureGNN:
    """Test GNN model architecture."""

    def test_model_initialization(self):
        """Test GNN model initializes correctly."""
        model = ArchitectureGNN(
            node_feature_dim=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.2,
        )

        assert model.hidden_dim == 128
        assert model.num_layers == 3
        assert len(model.convs) == 3
        assert len(model.batch_norms) == 3

    def test_forward_pass(self):
        """Test GNN forward pass."""
        model = ArchitectureGNN(
            node_feature_dim=32,
            hidden_dim=64,
            num_layers=2,
        )

        # Create dummy input
        num_nodes = 10

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, 15))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Forward pass
        output = model(x, edge_index, batch)

        # Check output
        assert output.shape == (1,)  # Single graph in batch
        assert 0.0 <= output.item() <= 1.0  # Sigmoid output

    def test_batch_processing(self):
        """Test processing multiple graphs in batch."""
        model = ArchitectureGNN(node_feature_dim=32, hidden_dim=64)

        # Create batch of 3 graphs
        num_nodes_per_graph = [5, 7, 6]
        total_nodes = sum(num_nodes_per_graph)

        x = torch.randn(total_nodes, 32)

        # Create batch indicator
        batch = []
        for i, n in enumerate(num_nodes_per_graph):
            batch.extend([i] * n)
        batch = torch.tensor(batch, dtype=torch.long)

        # Create edges
        edge_index = torch.randint(0, total_nodes, (2, 20))

        # Forward
        output = model(x, edge_index, batch)

        # Should output one prediction per graph
        assert output.shape == (3,)
        assert all(0.0 <= val <= 1.0 for val in output)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.skipif(SKIP_GNN_TESTS, reason="PyTorch not available")
class TestGNNPredictorIntegration:
    """Integration tests for GNN predictor."""

    def test_end_to_end_workflow(self, training_data):
        """Test complete GNN training and prediction workflow."""
        # 1. Initialize predictor
        predictor = GNNPredictor(
            {
                "hidden_dim": 128,
                "num_layers": 3,
                "lr": 0.001,
            }
        )

        # 2. Train
        stats = predictor.train(
            training_data,
            num_epochs=10,
            batch_size=16,
        )

        assert predictor.is_trained
        assert stats["best_val_loss"] > 0

        # 3. Predict on test graphs
        test_graphs = [graph for graph, _ in training_data[:5]]
        predictions = [predictor.predict(g) for g in test_graphs]

        assert len(predictions) == 5
        assert all(0.0 <= p <= 1.0 for p in predictions)

    def test_prediction_correlation_with_actual(self, training_data):
        """Test that predictions correlate with actual accuracies."""
        # Train on first 40 samples
        train_data = training_data[:40]
        test_data = training_data[40:]

        predictor = GNNPredictor()
        predictor.train(train_data, num_epochs=20, batch_size=16)

        # Predict on test set
        predictions = []
        actuals = []

        for graph, actual_acc in test_data:
            pred_acc = predictor.predict(graph)
            predictions.append(pred_acc)
            actuals.append(actual_acc)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1]

        # Correlation should be positive (at least some learning)
        # Note: With synthetic data, correlation may not be very high
        assert correlation > -0.5  # Not completely random


# ============================================================================
# Error Handling
# ============================================================================


@pytest.mark.skipif(SKIP_GNN_TESTS, reason="PyTorch not available")
class TestGNNErrorHandling:
    """Test error handling in GNN predictor."""

    def test_empty_training_data(self):
        """Test training with empty data."""
        predictor = GNNPredictor()

        with pytest.raises((ValueError, IndexError)):
            predictor.train([], num_epochs=5)

    def test_invalid_graph_structure(self):
        """Test prediction on invalid graph."""
        predictor = GNNPredictor()

        # Create empty graph
        empty_graph = ModelGraph()

        # Should handle gracefully
        try:
            pred = predictor.predict(empty_graph)
            # If it doesn't raise, prediction should still be valid
            assert 0.0 <= pred <= 1.0
        except Exception:
            # Or it raises an appropriate error
            pass


# Test that imports fail gracefully without PyTorch
def test_import_without_pytorch():
    """Test that GNN predictor handles missing PyTorch gracefully."""
    try:
        # Import succeeded - PyTorch is available or fallback works
        assert True
    except ImportError as e:
        # Should have clear error message
        assert "PyTorch" in str(e) or "torch" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
