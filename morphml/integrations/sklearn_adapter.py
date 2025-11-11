"""Scikit-learn adapter for MorphML.

Converts ModelGraph to scikit-learn Pipeline for classical ML.

Example:
    >>> from morphml.integrations import SklearnAdapter
    >>> adapter = SklearnAdapter()
    >>> pipeline = adapter.build_pipeline(graph)
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
"""

from typing import Dict, Any, Optional

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    Pipeline = None

from morphml.core.graph import ModelGraph, GraphNode
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class SklearnAdapter:
    """
    Convert ModelGraph to scikit-learn Pipeline.
    
    Supports classical ML algorithms and preprocessing steps.
    
    Example:
        >>> adapter = SklearnAdapter()
        >>> pipeline = adapter.build_pipeline(graph)
        >>> pipeline.fit(X_train, y_train)
        >>> score = pipeline.score(X_test, y_test)
    """
    
    def __init__(self):
        """Initialize Scikit-learn adapter."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn is required for SklearnAdapter. "
                "Install with: pip install scikit-learn"
            )
        logger.info("Initialized SklearnAdapter")
    
    def build_pipeline(
        self,
        graph: ModelGraph,
        config: Optional[Dict[str, Any]] = None
    ) -> Pipeline:
        """
        Build scikit-learn pipeline from graph.
        
        Args:
            graph: ModelGraph to convert
            config: Optional configuration
            
        Returns:
            sklearn Pipeline instance
            
        Example:
            >>> pipeline = adapter.build_pipeline(graph)
        """
        steps = []
        
        for node in graph.topological_sort():
            step = self._create_step(node)
            if step is not None:
                steps.append(step)
        
        if not steps:
            # Default pipeline
            steps = [
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier())
            ]
        
        pipeline = Pipeline(steps)
        
        logger.info(f"Created sklearn pipeline with {len(steps)} steps")
        
        return pipeline
    
    def _create_step(self, node: GraphNode) -> Optional[tuple]:
        """
        Create pipeline step from node.
        
        Args:
            node: GraphNode to convert
            
        Returns:
            Tuple of (name, estimator) or None
        """
        op = node.operation
        params = node.params
        
        if op == "input":
            return None
        
        elif op == "scaler":
            return (f"scaler_{node.id}", StandardScaler())
        
        elif op == "pca":
            n_components = params.get("n_components", 50)
            return (f"pca_{node.id}", PCA(n_components=n_components))
        
        elif op == "random_forest":
            return (f"rf_{node.id}", RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=params.get("random_state", 42)
            ))
        
        elif op == "gradient_boosting":
            return (f"gb_{node.id}", GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 3),
                random_state=params.get("random_state", 42)
            ))
        
        elif op == "logistic_regression":
            return (f"lr_{node.id}", LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 100),
                random_state=params.get("random_state", 42)
            ))
        
        elif op == "svm":
            return (f"svm_{node.id}", SVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale"),
                random_state=params.get("random_state", 42)
            ))
        
        elif op == "knn":
            return (f"knn_{node.id}", KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform")
            ))
        
        else:
            logger.warning(f"Unknown operation for sklearn: {op}")
            return None
    
    def create_search_space_for_sklearn(self, graph: ModelGraph) -> Dict[str, Any]:
        """
        Create hyperparameter search space for sklearn pipeline.
        
        Args:
            graph: ModelGraph
            
        Returns:
            Dictionary of hyperparameter distributions
            
        Example:
            >>> search_space = adapter.create_search_space_for_sklearn(graph)
            >>> # Use with GridSearchCV or RandomizedSearchCV
        """
        param_grid = {}
        
        for node in graph.topological_sort():
            op = node.operation
            params = node.params
            
            if op == "random_forest":
                param_grid[f"rf_{node.id}__n_estimators"] = [50, 100, 200]
                param_grid[f"rf_{node.id}__max_depth"] = [None, 10, 20, 30]
            
            elif op == "gradient_boosting":
                param_grid[f"gb_{node.id}__n_estimators"] = [50, 100, 200]
                param_grid[f"gb_{node.id}__learning_rate"] = [0.01, 0.1, 0.2]
            
            elif op == "logistic_regression":
                param_grid[f"lr_{node.id}__C"] = [0.1, 1.0, 10.0]
            
            elif op == "svm":
                param_grid[f"svm_{node.id}__C"] = [0.1, 1.0, 10.0]
                param_grid[f"svm_{node.id}__kernel"] = ["linear", "rbf"]
            
            elif op == "pca":
                param_grid[f"pca_{node.id}__n_components"] = [10, 20, 50, 100]
        
        return param_grid
