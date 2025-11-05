"""Ensemble predictor combining multiple methods.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional

import numpy as np

from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class EnsemblePredictor:
    """
    Ensemble of multiple performance predictors.

    Combines predictions from:
    - Proxy metric predictor
    - Learning curve predictor
    - GNN predictor (if available)

    Args:
        predictors: List of predictor instances
        weights: Weights for each predictor (default: equal)
        aggregation: How to combine ('weighted_average', 'max', 'min')

    Example:
        >>> from morphml.meta_learning.predictors import (
        ...     ProxyMetricPredictor,
        ...     EnsemblePredictor
        ... )
        >>>
        >>> # Create predictors
        >>> proxy_pred = ProxyMetricPredictor()
        >>> proxy_pred.train(training_data)
        >>>
        >>> # Ensemble
        >>> ensemble = EnsemblePredictor(
        ...     predictors=[proxy_pred],
        ...     weights=[1.0]
        ... )
        >>>
        >>> prediction = ensemble.predict(new_graph)
    """

    def __init__(
        self,
        predictors: List[Any],
        weights: Optional[List[float]] = None,
        aggregation: str = "weighted_average",
    ):
        """Initialize ensemble predictor."""
        self.predictors = predictors

        # Default: equal weights
        if weights is None:
            weights = [1.0 / len(predictors)] * len(predictors)

        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

        self.aggregation = aggregation

        logger.info(
            f"Initialized EnsemblePredictor with {len(predictors)} predictors "
            f"(aggregation={aggregation})"
        )

    def predict(self, graph: ModelGraph, **predictor_kwargs) -> float:
        """
        Predict using ensemble.

        Args:
            graph: Architecture to evaluate
            **predictor_kwargs: Additional kwargs for specific predictors

        Returns:
            Ensemble prediction
        """
        predictions = []

        for i, predictor in enumerate(self.predictors):
            try:
                # Get prediction
                if hasattr(predictor, "predict"):
                    pred = predictor.predict(graph, **predictor_kwargs)
                    predictions.append(pred)
                else:
                    logger.warning(f"Predictor {i} has no predict method, skipping")
            except Exception as e:
                logger.warning(f"Predictor {i} failed: {e}")

        if not predictions:
            logger.warning("No valid predictions, returning 0.5")
            return 0.5

        # Aggregate
        if self.aggregation == "weighted_average":
            # Use weights corresponding to valid predictions
            valid_weights = self.weights[: len(predictions)]
            total_weight = sum(valid_weights)
            valid_weights = [w / total_weight for w in valid_weights]

            result = sum(w * p for w, p in zip(valid_weights, predictions))

        elif self.aggregation == "max":
            result = max(predictions)

        elif self.aggregation == "min":
            result = min(predictions)

        elif self.aggregation == "median":
            result = np.median(predictions)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return float(result)

    def batch_predict(self, graphs: List[ModelGraph], **predictor_kwargs) -> np.ndarray:
        """
        Batch prediction for efficiency.

        Args:
            graphs: List of architectures
            **predictor_kwargs: Additional kwargs

        Returns:
            Array of predictions
        """
        # Try batch prediction if available
        all_predictions = []

        for predictor in self.predictors:
            try:
                if hasattr(predictor, "batch_predict"):
                    preds = predictor.batch_predict(graphs, **predictor_kwargs)
                else:
                    # Fallback to sequential
                    preds = np.array([predictor.predict(g, **predictor_kwargs) for g in graphs])

                all_predictions.append(preds)
            except Exception as e:
                logger.warning(f"Batch prediction failed for predictor: {e}")

        if not all_predictions:
            return np.full(len(graphs), 0.5)

        # Aggregate
        all_predictions = np.array(all_predictions)  # Shape: (n_predictors, n_graphs)

        if self.aggregation == "weighted_average":
            weights = np.array(self.weights[: len(all_predictions)])
            weights = weights / weights.sum()
            result = np.average(all_predictions, axis=0, weights=weights)

        elif self.aggregation == "max":
            result = np.max(all_predictions, axis=0)

        elif self.aggregation == "min":
            result = np.min(all_predictions, axis=0)

        elif self.aggregation == "median":
            result = np.median(all_predictions, axis=0)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return result

    def get_individual_predictions(self, graph: ModelGraph, **predictor_kwargs) -> Dict[str, float]:
        """
        Get predictions from each individual predictor.

        Useful for debugging and analysis.

        Args:
            graph: Architecture to evaluate
            **predictor_kwargs: Additional kwargs

        Returns:
            Dictionary mapping predictor name to prediction
        """
        predictions = {}

        for i, predictor in enumerate(self.predictors):
            name = predictor.__class__.__name__
            try:
                pred = predictor.predict(graph, **predictor_kwargs)
                predictions[f"{name}_{i}"] = pred
            except Exception as e:
                logger.warning(f"Predictor {name} failed: {e}")
                predictions[f"{name}_{i}"] = None

        # Add ensemble prediction
        predictions["ensemble"] = self.predict(graph, **predictor_kwargs)

        return predictions

    def update_weights(self, new_weights: List[float]) -> None:
        """
        Update predictor weights.

        Useful for adaptive weighting based on performance.

        Args:
            new_weights: New weights (will be normalized)
        """
        if len(new_weights) != len(self.predictors):
            raise ValueError(f"Expected {len(self.predictors)} weights, got {len(new_weights)}")

        # Normalize
        total = sum(new_weights)
        self.weights = [w / total for w in new_weights]

        logger.info(f"Updated weights: {self.weights}")
