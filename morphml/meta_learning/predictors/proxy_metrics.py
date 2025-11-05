"""Proxy metric-based performance prediction.

Fast prediction using cheap architectural features.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ProxyMetricPredictor:
    """
    Predict performance using cheap proxy metrics.
    
    Uses architectural features that can be computed instantly:
    - Number of parameters
    - Network depth
    - Network width
    - Operation diversity
    - Skip connections
    
    Args:
        use_scaler: Whether to normalize features
    
    Example:
        >>> predictor = ProxyMetricPredictor()
        >>> 
        >>> # Train on historical data
        >>> training_data = [(graph1, 0.92), (graph2, 0.87), ...]
        >>> predictor.train(training_data)
        >>> 
        >>> # Predict new architecture
        >>> predicted_acc = predictor.predict(new_graph)
    """
    
    def __init__(self, use_scaler: bool = True):
        """Initialize predictor."""
        self.model = None
        self.scaler = StandardScaler() if use_scaler else None
        self.feature_names = []
        self.is_trained = False
        
        logger.info("Initialized ProxyMetricPredictor")
    
    def extract_features(self, graph: ModelGraph) -> Dict[str, float]:
        """
        Extract proxy features from architecture.
        
        Args:
            graph: Architecture graph
        
        Returns:
            Dictionary of proxy metrics
        """
        features = {}
        
        # Basic metrics
        features['num_layers'] = len(graph.layers)
        features['num_parameters'] = graph.count_parameters()
        
        # Operation counts
        op_counts = {}
        for layer in graph.layers:
            op_type = layer.layer_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        features['num_conv'] = op_counts.get('conv2d', 0)
        features['num_dense'] = op_counts.get('dense', 0)
        features['num_pool'] = op_counts.get('maxpool2d', 0) + op_counts.get('avgpool2d', 0)
        features['num_norm'] = op_counts.get('batchnorm', 0)
        features['num_activation'] = op_counts.get('relu', 0) + op_counts.get('tanh', 0)
        features['num_dropout'] = op_counts.get('dropout', 0)
        
        # Diversity
        features['operation_diversity'] = len(op_counts)
        
        # Network shape metrics
        layer_widths = []
        for layer in graph.layers:
            if layer.layer_type == 'conv2d':
                filters = layer.config.get('filters', 64)
                layer_widths.append(filters)
            elif layer.layer_type == 'dense':
                units = layer.config.get('units', 128)
                layer_widths.append(units)
        
        if layer_widths:
            features['avg_width'] = np.mean(layer_widths)
            features['max_width'] = np.max(layer_widths)
            features['min_width'] = np.min(layer_widths)
        else:
            features['avg_width'] = 0
            features['max_width'] = 0
            features['min_width'] = 0
        
        # Depth-to-width ratio
        if features['avg_width'] > 0:
            features['depth_to_width_ratio'] = features['num_layers'] / features['avg_width']
        else:
            features['depth_to_width_ratio'] = 0
        
        # Parameter efficiency
        if features['num_layers'] > 0:
            features['params_per_layer'] = features['num_parameters'] / features['num_layers']
        else:
            features['params_per_layer'] = 0
        
        return features
    
    def train(
        self,
        training_data: List[Tuple[ModelGraph, float]],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train predictor on historical data.
        
        Args:
            training_data: List of (architecture, accuracy) pairs
            validation_split: Fraction for validation
        
        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using dummy predictor")
            self.is_trained = True
            return {"error": "sklearn_not_available"}
        
        logger.info(f"Training ProxyMetricPredictor on {len(training_data)} samples")
        
        # Extract features
        X = []
        y = []
        
        for graph, accuracy in training_data:
            features = self.extract_features(graph)
            
            # Store feature names from first sample
            if not self.feature_names:
                self.feature_names = sorted(features.keys())
            
            # Convert to vector
            feature_vec = [features[name] for name in self.feature_names]
            X.append(feature_vec)
            y.append(accuracy)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        n_train = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Scale features
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if len(X_val) > 0 else 0.0
        
        # Feature importance
        importances = self.model.feature_importances_
        top_features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        logger.info(f"Training complete: R²={train_score:.3f}, Val R²={val_score:.3f}")
        logger.info(f"Top features: {top_features}")
        
        return {
            "train_score": train_score,
            "val_score": val_score,
            "top_features": dict(top_features),
        }
    
    def predict(self, graph: ModelGraph) -> float:
        """
        Predict accuracy for architecture.
        
        Args:
            graph: Architecture to evaluate
        
        Returns:
            Predicted accuracy (0-1)
        """
        if not self.is_trained:
            logger.warning("Predictor not trained, returning 0.5")
            return 0.5
        
        # Extract features
        features = self.extract_features(graph)
        
        # Convert to vector
        feature_vec = np.array([[features[name] for name in self.feature_names]])
        
        # Scale
        if self.scaler:
            feature_vec = self.scaler.transform(feature_vec)
        
        # Predict
        prediction = self.model.predict(feature_vec)[0]
        
        # Clip to valid range
        return np.clip(prediction, 0.0, 1.0)
    
    def batch_predict(self, graphs: List[ModelGraph]) -> np.ndarray:
        """
        Predict for multiple architectures efficiently.
        
        Args:
            graphs: List of architectures
        
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            return np.full(len(graphs), 0.5)
        
        # Extract all features
        X = []
        for graph in graphs:
            features = self.extract_features(graph)
            feature_vec = [features[name] for name in self.feature_names]
            X.append(feature_vec)
        
        X = np.array(X)
        
        # Scale
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X)
        
        return np.clip(predictions, 0.0, 1.0)
