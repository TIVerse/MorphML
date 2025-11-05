"""Transfer learning across tasks for neural architecture search.

Enables architecture transfer, fine-tuning, and domain adaptation.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Any, Dict, List, Optional

import numpy as np

from morphml.core.dsl import SearchSpace
from morphml.core.graph import ModelGraph
from morphml.logging_config import get_logger
from morphml.meta_learning.architecture_similarity import compute_task_similarity
from morphml.meta_learning.experiment_database import TaskMetadata

logger = get_logger(__name__)


class ArchitectureTransfer:
    """
    Transfer architectures between related tasks.

    Strategies:
    1. **Direct Transfer**: Use architecture as-is
    2. **Adaptation**: Modify input/output layers for new task
    3. **Capacity Adjustment**: Scale model size based on task complexity
    4. **Progressive Transfer**: Gradually adapt through intermediate tasks

    Example:
        >>> # Transfer CIFAR-10 architecture to CIFAR-100
        >>> source_task = TaskMetadata(
        ...     task_id='cifar10',
        ...     dataset_name='CIFAR-10',
        ...     num_classes=10,
        ...     input_size=(3, 32, 32),
        ...     num_samples=50000
        ... )
        >>>
        >>> target_task = TaskMetadata(
        ...     task_id='cifar100',
        ...     dataset_name='CIFAR-100',
        ...     num_classes=100,
        ...     input_size=(3, 32, 32),
        ...     num_samples=50000
        ... )
        >>>
        >>> transferred = ArchitectureTransfer.transfer_architecture(
        ...     source_arch=best_arch_cifar10,
        ...     source_task=source_task,
        ...     target_task=target_task,
        ...     adaptation_strategy='modify_head'
        ... )
    """

    @staticmethod
    def transfer_architecture(
        source_arch: ModelGraph,
        source_task: TaskMetadata,
        target_task: TaskMetadata,
        adaptation_strategy: str = "modify_head",
        capacity_scale: float = 1.0,
    ) -> ModelGraph:
        """
        Adapt architecture for new task.

        Args:
            source_arch: Architecture from source task
            source_task: Source task metadata
            target_task: Target task metadata
            adaptation_strategy: How to adapt ('direct', 'modify_head', 'full_adapt')
            capacity_scale: Scale model capacity (e.g., 1.5 for larger model)

        Returns:
            Transferred architecture adapted for target task
        """
        logger.info(
            f"Transferring architecture from {source_task.dataset_name} "
            f"to {target_task.dataset_name} (strategy={adaptation_strategy})"
        )

        # Clone architecture
        transferred = source_arch.clone()

        if adaptation_strategy == "direct":
            # No modifications - direct transfer
            logger.info("Direct transfer: no modifications")
            return transferred

        elif adaptation_strategy == "modify_head":
            # Modify input and output layers only
            transferred = ArchitectureTransfer._modify_io_layers(
                transferred, source_task, target_task
            )

        elif adaptation_strategy == "full_adapt":
            # Full adaptation: IO + capacity scaling
            transferred = ArchitectureTransfer._modify_io_layers(
                transferred, source_task, target_task
            )

            if capacity_scale != 1.0:
                transferred = ArchitectureTransfer._scale_capacity(transferred, capacity_scale)

        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")

        logger.info(
            f"Transfer complete. Nodes: {len(source_arch.nodes)} → {len(transferred.nodes)}"
        )

        return transferred

    @staticmethod
    def _modify_io_layers(
        graph: ModelGraph,
        source_task: TaskMetadata,
        target_task: TaskMetadata,
    ) -> ModelGraph:
        """Modify input and output layers for new task."""
        modified = graph.clone()

        # Modify input layer
        input_nodes = [n for n in modified.nodes.values() if n.operation == "input"]

        if input_nodes and source_task.input_size != target_task.input_size:
            for input_node in input_nodes:
                input_node.params["input_shape"] = target_task.input_size
                logger.debug(
                    f"Updated input shape: {source_task.input_size} → {target_task.input_size}"
                )

        # Modify output layer
        output_nodes = [n for n in modified.nodes.values() if n.operation == "output"]

        if not output_nodes:
            # Find dense/linear layers near the end
            topo_order = modified.topological_sort()
            if topo_order:
                last_nodes = [n for n in topo_order[-3:] if n.operation in ["dense", "linear"]]
                output_nodes = last_nodes[-1:] if last_nodes else []

        if output_nodes and source_task.num_classes != target_task.num_classes:
            for output_node in output_nodes:
                if "units" in output_node.params:
                    output_node.params["units"] = target_task.num_classes
                    logger.debug(
                        f"Updated output units: {source_task.num_classes} → {target_task.num_classes}"
                    )
                elif "out_features" in output_node.params:
                    output_node.params["out_features"] = target_task.num_classes

        return modified

    @staticmethod
    def _scale_capacity(graph: ModelGraph, scale: float) -> ModelGraph:
        """
        Scale model capacity by adjusting layer widths.

        Args:
            graph: Architecture to scale
            scale: Scaling factor (e.g., 1.5 = 50% wider)

        Returns:
            Scaled architecture
        """
        scaled = graph.clone()

        for node in scaled.nodes.values():
            # Scale convolutional filters
            if node.operation in ["conv2d", "conv1d"] and "filters" in node.params:
                original = node.params["filters"]
                node.params["filters"] = int(original * scale)
                logger.debug(f"Scaled {node.id} filters: {original} → {node.params['filters']}")

            # Scale dense units
            elif node.operation in ["dense", "linear"] and "units" in node.params:
                # Don't scale output layer
                if node.operation != "output":
                    original = node.params["units"]
                    node.params["units"] = int(original * scale)
                    logger.debug(f"Scaled {node.id} units: {original} → {node.params['units']}")

        return scaled

    @staticmethod
    def evaluate_transferability(
        source_task: TaskMetadata,
        target_task: TaskMetadata,
        method: str = "comprehensive",
    ) -> float:
        """
        Estimate how well architectures will transfer between tasks.

        Args:
            source_task: Source task metadata
            target_task: Target task metadata
            method: Scoring method ('comprehensive', 'simple', 'similarity')

        Returns:
            Transferability score (0-1, higher = better transfer expected)
        """
        if method == "simple":
            return ArchitectureTransfer._simple_transferability(source_task, target_task)
        elif method == "similarity":
            return compute_task_similarity(source_task, target_task)
        else:  # comprehensive
            return ArchitectureTransfer._comprehensive_transferability(source_task, target_task)

    @staticmethod
    def _simple_transferability(
        source_task: TaskMetadata,
        target_task: TaskMetadata,
    ) -> float:
        """Simple heuristic-based transferability."""
        # Same dataset = perfect transfer
        if source_task.dataset_name == target_task.dataset_name:
            return 1.0

        # Different problem types = poor transfer
        if source_task.problem_type != target_task.problem_type:
            return 0.3

        # Compare task properties
        size_ratio = min(source_task.num_samples, target_task.num_samples) / max(
            source_task.num_samples, target_task.num_samples
        )

        class_ratio = min(source_task.num_classes, target_task.num_classes) / max(
            source_task.num_classes, target_task.num_classes
        )

        # Average the ratios
        transferability = (size_ratio + class_ratio) / 2.0

        return float(np.clip(transferability, 0.0, 1.0))

    @staticmethod
    def _comprehensive_transferability(
        source_task: TaskMetadata,
        target_task: TaskMetadata,
    ) -> float:
        """Comprehensive transferability scoring."""
        scores = []

        # 1. Problem type match (0.3 weight)
        if source_task.problem_type == target_task.problem_type:
            scores.append((1.0, 0.3))
        else:
            scores.append((0.2, 0.3))

        # 2. Dataset family (0.2 weight)
        dataset_families = {
            "CIFAR-10": "cifar",
            "CIFAR-100": "cifar",
            "ImageNet": "imagenet",
            "ImageNet-16": "imagenet",
            "MNIST": "mnist",
            "Fashion-MNIST": "mnist",
        }

        source_family = dataset_families.get(source_task.dataset_name, "other")
        target_family = dataset_families.get(target_task.dataset_name, "other")

        if source_family == target_family:
            scores.append((1.0, 0.2))
        else:
            scores.append((0.5, 0.2))

        # 3. Input size similarity (0.2 weight)
        if source_task.input_size == target_task.input_size:
            input_score = 1.0
        else:
            # Compute dimensionality ratio
            source_dims = np.prod(source_task.input_size) if source_task.input_size else 1
            target_dims = np.prod(target_task.input_size) if target_task.input_size else 1
            ratio = min(source_dims, target_dims) / max(source_dims, target_dims)
            input_score = float(ratio)

        scores.append((input_score, 0.2))

        # 4. Class count similarity (0.15 weight)
        class_ratio = min(source_task.num_classes, target_task.num_classes) / max(
            source_task.num_classes, target_task.num_classes
        )
        scores.append((class_ratio, 0.15))

        # 5. Dataset size similarity (0.15 weight)
        size_ratio = min(source_task.num_samples, target_task.num_samples) / max(
            source_task.num_samples, target_task.num_samples
        )
        scores.append((size_ratio, 0.15))

        # Weighted average
        total_score = sum(score * weight for score, weight in scores)

        return float(np.clip(total_score, 0.0, 1.0))

    @staticmethod
    def recommend_transfer_strategy(
        source_task: TaskMetadata,
        target_task: TaskMetadata,
    ) -> Dict[str, Any]:
        """
        Recommend optimal transfer strategy.

        Returns:
            Dict with:
                - strategy: Recommended strategy name
                - capacity_scale: Recommended capacity scaling
                - reasoning: Explanation
        """
        transferability = ArchitectureTransfer.evaluate_transferability(source_task, target_task)

        # Determine strategy
        if transferability > 0.9:
            strategy = "direct"
            capacity_scale = 1.0
            reasoning = "Tasks are very similar - direct transfer recommended"

        elif transferability > 0.7:
            strategy = "modify_head"
            capacity_scale = 1.0
            reasoning = "Tasks are similar - only modify input/output layers"

        elif transferability > 0.4:
            # Check if target is larger
            if target_task.num_classes > source_task.num_classes * 2:
                strategy = "full_adapt"
                capacity_scale = 1.5
                reasoning = "Target task larger - increase capacity"
            elif target_task.num_classes < source_task.num_classes / 2:
                strategy = "full_adapt"
                capacity_scale = 0.7
                reasoning = "Target task smaller - reduce capacity"
            else:
                strategy = "full_adapt"
                capacity_scale = 1.0
                reasoning = "Moderate similarity - full adaptation"

        else:
            strategy = "full_adapt"
            capacity_scale = 1.2
            reasoning = "Tasks differ significantly - extensive adaptation needed"

        return {
            "strategy": strategy,
            "capacity_scale": capacity_scale,
            "transferability": transferability,
            "reasoning": reasoning,
        }


class FineTuningStrategy:
    """
    Fine-tuning protocols for transferred architectures.

    When training a transferred architecture on new task, different
    strategies can be employed:

    1. **Full Fine-Tuning**: Train all parameters
    2. **Freeze Early Layers**: Only train later layers
    3. **Differential Learning Rates**: Lower LR for early layers
    4. **Progressive Unfreezing**: Gradually unfreeze layers

    Note: This class provides configuration. Actual training requires
    a framework-specific implementation (PyTorch, TensorFlow, etc.)

    Example:
        >>> strategy = FineTuningStrategy.get_strategy(
        ...     transfer_type='similar_tasks',
        ...     model_depth=20
        ... )
        >>>
        >>> print(strategy)
        {
            'method': 'freeze_early',
            'freeze_ratio': 0.5,
            'learning_rate': 0.001,
            'num_epochs': 50,
            ...
        }
    """

    @staticmethod
    def get_strategy(
        transfer_type: str,
        model_depth: int,
        target_dataset_size: int = 50000,
    ) -> Dict[str, Any]:
        """
        Get recommended fine-tuning strategy.

        Args:
            transfer_type: Type of transfer
                - 'same_domain': Same dataset family
                - 'similar_tasks': Related but different tasks
                - 'distant_tasks': Very different tasks
            model_depth: Number of layers in model
            target_dataset_size: Size of target dataset

        Returns:
            Fine-tuning configuration dict
        """
        if transfer_type == "same_domain":
            # Minimal adaptation needed
            return {
                "method": "freeze_early",
                "freeze_ratio": 0.7,  # Freeze first 70% of layers
                "learning_rate": 1e-4,
                "num_epochs": 30,
                "warmup_epochs": 5,
                "description": "Freeze early layers, train classification head",
            }

        elif transfer_type == "similar_tasks":
            # Moderate adaptation
            if target_dataset_size < 10000:
                # Small dataset - be conservative
                return {
                    "method": "freeze_early",
                    "freeze_ratio": 0.5,
                    "learning_rate": 5e-4,
                    "num_epochs": 50,
                    "warmup_epochs": 10,
                    "description": "Freeze half, careful training to avoid overfitting",
                }
            else:
                # Larger dataset - can train more
                return {
                    "method": "differential_lr",
                    "early_lr": 1e-5,
                    "late_lr": 1e-3,
                    "num_epochs": 75,
                    "warmup_epochs": 10,
                    "description": "Lower LR for early layers, higher for later layers",
                }

        else:  # distant_tasks
            # Full adaptation needed
            return {
                "method": "progressive_unfreezing",
                "initial_freeze_ratio": 0.8,
                "unfreeze_schedule": [0.6, 0.4, 0.2, 0.0],  # Gradually unfreeze
                "learning_rate": 1e-3,
                "num_epochs": 100,
                "warmup_epochs": 15,
                "description": "Progressively unfreeze layers during training",
            }

    @staticmethod
    def generate_freeze_mask(
        num_layers: int,
        freeze_ratio: float,
    ) -> List[bool]:
        """
        Generate mask indicating which layers to freeze.

        Args:
            num_layers: Total number of layers
            freeze_ratio: Ratio of layers to freeze (0-1)

        Returns:
            List of booleans (True = freeze, False = train)
        """
        freeze_until = int(num_layers * freeze_ratio)

        mask = []
        for i in range(num_layers):
            mask.append(i < freeze_until)

        return mask


class MultiTaskNAS:
    """
    Neural Architecture Search for multiple tasks simultaneously.

    Finds architectures that perform well across a distribution of tasks,
    enabling better transfer learning and generalization.

    Args:
        tasks: List of tasks to optimize for
        search_space: Architecture search space
        task_weights: Optional weights for each task (default: equal)

    Example:
        >>> tasks = [
        ...     TaskMetadata(task_id='cifar10', ...),
        ...     TaskMetadata(task_id='cifar100', ...),
        ...     TaskMetadata(task_id='svhn', ...),
        ... ]
        >>>
        >>> multi_nas = MultiTaskNAS(
        ...     tasks=tasks,
        ...     search_space=space,
        ...     task_weights=[0.5, 0.3, 0.2]  # Prioritize CIFAR-10
        ... )
        >>>
        >>> # This would be integrated with an optimizer
        >>> # best_arch = multi_nas.search(optimizer, evaluator)
    """

    def __init__(
        self,
        tasks: List[TaskMetadata],
        search_space: SearchSpace,
        task_weights: Optional[List[float]] = None,
    ):
        """Initialize multi-task NAS."""
        self.tasks = tasks
        self.search_space = search_space

        # Normalize task weights
        if task_weights is None:
            task_weights = [1.0 / len(tasks)] * len(tasks)

        total = sum(task_weights)
        self.task_weights = [w / total for w in task_weights]

        logger.info(
            f"Initialized MultiTaskNAS with {len(tasks)} tasks " f"(weights={self.task_weights})"
        )

    def evaluate_multi_task_fitness(
        self,
        architecture: ModelGraph,
        evaluator_fn,
    ) -> Dict[str, float]:
        """
        Evaluate architecture on all tasks.

        Args:
            architecture: Architecture to evaluate
            evaluator_fn: Function to evaluate arch on a task
                          Signature: evaluator_fn(arch, task) -> fitness

        Returns:
            Dict with per-task fitness and weighted average
        """
        fitnesses = {}

        for task, _weight in zip(self.tasks, self.task_weights):
            # Adapt architecture for this task
            adapted = ArchitectureTransfer.transfer_architecture(
                source_arch=architecture,
                source_task=self.tasks[0],  # Use first task as "source"
                target_task=task,
                adaptation_strategy="modify_head",
            )

            # Evaluate
            fitness = evaluator_fn(adapted, task)
            fitnesses[task.task_id] = fitness

        # Weighted average
        weighted_fitness = sum(
            fitnesses[task.task_id] * weight for task, weight in zip(self.tasks, self.task_weights)
        )

        fitnesses["weighted_average"] = weighted_fitness

        return fitnesses

    def create_multi_task_evaluator(self, base_evaluator):
        """
        Create evaluator function for multi-task optimization.

        Returns function that can be used with any optimizer.
        """

        def multi_task_eval(architecture: ModelGraph) -> float:
            results = self.evaluate_multi_task_fitness(
                architecture,
                lambda arch, task: base_evaluator(arch),
            )
            return results["weighted_average"]

        return multi_task_eval
