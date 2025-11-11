"""Performance predictors for architecture evaluation.

Provides fast performance estimation without full training:
- Proxy metrics (parameters, FLOPs, depth)
- Learning curve extrapolation
- GNN-based prediction (when PyTorch available)
- Ensemble methods

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.meta_learning.predictors.ensemble import EnsemblePredictor
from morphml.meta_learning.predictors.learning_curve import LearningCurvePredictor
from morphml.meta_learning.predictors.proxy_metrics import ProxyMetricPredictor

__all__ = [
    "ProxyMetricPredictor",
    "LearningCurvePredictor",
    "EnsemblePredictor",
]

# Optional GNN predictor (requires PyTorch)
try:
    __all__.append("GNNPredictor")
except ImportError:
    pass
