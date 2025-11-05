"""Tests for performance predictors."""

import numpy as np
import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.meta_learning.predictors import (
    ProxyMetricPredictor,
    LearningCurvePredictor,
    EnsemblePredictor,
)


class TestProxyMetricPredictor:
    """Test proxy metric predictor."""
    
    @pytest.fixture
    def sample_graphs(self):
        """Create sample graphs."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.conv2d(filters=128),
            Layer.maxpool2d(pool_size=2),
            Layer.flatten(),
            Layer.dense(units=256),
            Layer.output(units=10),
        )
        
        graphs = [space.sample() for _ in range(20)]
        return graphs
    
    def test_feature_extraction(self, sample_graphs):
        """Test feature extraction."""
        predictor = ProxyMetricPredictor()
        
        graph = sample_graphs[0]
        features = predictor.extract_features(graph)
        
        assert 'num_layers' in features
        assert 'num_parameters' in features
        assert 'num_conv' in features
        assert features['num_layers'] > 0
    
    def test_training(self, sample_graphs):
        """Test training predictor."""
        predictor = ProxyMetricPredictor()
        
        # Create training data with synthetic labels
        training_data = [
            (graph, 0.7 + np.random.rand() * 0.2)
            for graph in sample_graphs
        ]
        
        metrics = predictor.train(training_data, validation_split=0.2)
        
        assert predictor.is_trained
        if 'error' not in metrics:
            assert 'train_score' in metrics
    
    def test_prediction(self, sample_graphs):
        """Test making predictions."""
        predictor = ProxyMetricPredictor()
        
        # Train
        training_data = [(g, 0.8) for g in sample_graphs[:15]]
        predictor.train(training_data)
        
        # Predict
        test_graph = sample_graphs[16]
        prediction = predictor.predict(test_graph)
        
        assert 0.0 <= prediction <= 1.0
    
    def test_batch_prediction(self, sample_graphs):
        """Test batch prediction."""
        predictor = ProxyMetricPredictor()
        
        training_data = [(g, 0.8) for g in sample_graphs[:15]]
        predictor.train(training_data)
        
        test_graphs = sample_graphs[15:]
        predictions = predictor.batch_predict(test_graphs)
        
        assert len(predictions) == len(test_graphs)
        assert all(0.0 <= p <= 1.0 for p in predictions)


class TestLearningCurvePredictor:
    """Test learning curve predictor."""
    
    def test_power_law_curve(self):
        """Test power law fitting."""
        predictor = LearningCurvePredictor(curve_type="power_law")
        
        # Synthetic learning curve
        epochs = list(range(1, 11))
        accuracies = [0.3, 0.45, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.79]
        
        final_acc = predictor.predict_final_accuracy(
            accuracies, epochs, final_epoch=200
        )
        
        assert 0.0 <= final_acc <= 1.0
        assert final_acc >= accuracies[-1]  # Should be at least as good
    
    def test_exponential_curve(self):
        """Test exponential fitting."""
        predictor = LearningCurvePredictor(curve_type="exponential")
        
        epochs = list(range(1, 11))
        accuracies = [0.3, 0.45, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.79]
        
        final_acc = predictor.predict_final_accuracy(
            accuracies, epochs, final_epoch=200
        )
        
        assert 0.0 <= final_acc <= 1.0
    
    def test_early_stopping(self):
        """Test early stopping decision."""
        predictor = LearningCurvePredictor()
        
        # Poor performance trajectory
        poor_accuracies = [0.2, 0.25, 0.28, 0.30, 0.31, 0.32]
        
        should_stop = predictor.should_early_stop(
            poor_accuracies, threshold=0.8
        )
        
        # Should recommend stopping since predicted < 0.8
        assert isinstance(should_stop, bool)
    
    def test_fit_curve(self):
        """Test curve fitting for visualization."""
        predictor = LearningCurvePredictor()
        
        accuracies = [0.3, 0.45, 0.55, 0.62, 0.67]
        
        fitted_epochs, fitted_acc = predictor.fit_curve(accuracies)
        
        assert len(fitted_epochs) == len(fitted_acc)
        assert len(fitted_epochs) > len(accuracies)


class TestEnsemblePredictor:
    """Test ensemble predictor."""
    
    @pytest.fixture
    def sample_graphs(self):
        """Create sample graphs."""
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10),
        )
        return [space.sample() for _ in range(10)]
    
    def test_initialization(self):
        """Test ensemble initialization."""
        predictor1 = ProxyMetricPredictor()
        predictor2 = ProxyMetricPredictor()
        
        ensemble = EnsemblePredictor([predictor1, predictor2])
        
        assert len(ensemble.predictors) == 2
        assert len(ensemble.weights) == 2
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6
    
    def test_custom_weights(self):
        """Test custom weights."""
        predictor1 = ProxyMetricPredictor()
        predictor2 = ProxyMetricPredictor()
        
        ensemble = EnsemblePredictor(
            [predictor1, predictor2],
            weights=[0.7, 0.3]
        )
        
        assert abs(ensemble.weights[0] - 0.7) < 1e-6
        assert abs(ensemble.weights[1] - 0.3) < 1e-6
    
    def test_prediction(self, sample_graphs):
        """Test ensemble prediction."""
        # Train predictors
        predictor1 = ProxyMetricPredictor()
        predictor1.train([(g, 0.8) for g in sample_graphs[:8]])
        
        predictor2 = ProxyMetricPredictor()
        predictor2.train([(g, 0.75) for g in sample_graphs[:8]])
        
        # Ensemble
        ensemble = EnsemblePredictor([predictor1, predictor2])
        
        test_graph = sample_graphs[8]
        prediction = ensemble.predict(test_graph)
        
        assert 0.0 <= prediction <= 1.0
    
    def test_aggregation_methods(self, sample_graphs):
        """Test different aggregation methods."""
        predictor1 = ProxyMetricPredictor()
        predictor1.train([(g, 0.8) for g in sample_graphs[:8]])
        
        test_graph = sample_graphs[8]
        
        for method in ['weighted_average', 'max', 'min', 'median']:
            ensemble = EnsemblePredictor([predictor1], aggregation=method)
            pred = ensemble.predict(test_graph)
            assert 0.0 <= pred <= 1.0
    
    def test_batch_prediction(self, sample_graphs):
        """Test batch prediction."""
        predictor1 = ProxyMetricPredictor()
        predictor1.train([(g, 0.8) for g in sample_graphs[:7]])
        
        ensemble = EnsemblePredictor([predictor1])
        
        test_graphs = sample_graphs[7:]
        predictions = ensemble.batch_predict(test_graphs)
        
        assert len(predictions) == len(test_graphs)
        assert all(0.0 <= p <= 1.0 for p in predictions)


def test_predictors_imports():
    """Test that predictors can be imported."""
    from morphml.meta_learning import (
        ProxyMetricPredictor,
        LearningCurvePredictor,
        EnsemblePredictor,
    )
    
    assert ProxyMetricPredictor is not None
    assert LearningCurvePredictor is not None
    assert EnsemblePredictor is not None
