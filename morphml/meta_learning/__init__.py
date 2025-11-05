"""Meta-learning and transfer learning for MorphML.

Enables intelligent search through:
- Warm-starting from past experiments
- Performance prediction
- Transfer learning across tasks
- Knowledge base management
- Strategy evolution

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from morphml.meta_learning.architecture_similarity import ArchitectureSimilarity
from morphml.meta_learning.experiment_database import ExperimentDatabase, TaskMetadata
from morphml.meta_learning.warm_start import WarmStarter
from morphml.meta_learning.predictors import (
    ProxyMetricPredictor,
    LearningCurvePredictor,
    EnsemblePredictor,
)
from morphml.meta_learning.knowledge_base import (
    KnowledgeBase,
    ArchitectureEmbedder,
    MetaFeatureExtractor,
    VectorStore,
)
from morphml.meta_learning.strategy_evolution import (
    UCBSelector,
    ThompsonSamplingSelector,
    AdaptiveOptimizer,
    PortfolioOptimizer,
)
from morphml.meta_learning.transfer import (
    ArchitectureTransfer,
    FineTuningStrategy,
    MultiTaskNAS,
)

__all__ = [
    # Warm-starting
    "WarmStarter",
    "TaskMetadata",
    "ExperimentDatabase",
    "ArchitectureSimilarity",
    # Performance prediction
    "ProxyMetricPredictor",
    "LearningCurvePredictor",
    "EnsemblePredictor",
    # Knowledge base
    "KnowledgeBase",
    "ArchitectureEmbedder",
    "MetaFeatureExtractor",
    "VectorStore",
    # Strategy evolution
    "UCBSelector",
    "ThompsonSamplingSelector",
    "AdaptiveOptimizer",
    "PortfolioOptimizer",
    # Transfer learning
    "ArchitectureTransfer",
    "FineTuningStrategy",
    "MultiTaskNAS",
]

__version__ = "0.1.0"
