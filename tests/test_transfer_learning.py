"""
Unit tests for transfer learning functionality.

Tests ArchitectureTransfer, FineTuningStrategy, and MultiTaskNAS.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import pytest

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import GraphNode, ModelGraph
from morphml.meta_learning.experiment_database import TaskMetadata
from morphml.meta_learning.transfer import (
    ArchitectureTransfer,
    FineTuningStrategy,
    MultiTaskNAS,
)


@pytest.fixture
def sample_architecture():
    """Create a sample architecture for testing."""
    graph = ModelGraph()

    # Simple CNN architecture
    input_node = GraphNode("input", "input", {"input_shape": (3, 32, 32)})
    conv1 = GraphNode("conv1", "conv2d", {"filters": 64, "kernel_size": 3})
    pool1 = GraphNode("pool1", "maxpool", {"pool_size": 2})
    conv2 = GraphNode("conv2", "conv2d", {"filters": 128, "kernel_size": 3})
    pool2 = GraphNode("pool2", "maxpool", {"pool_size": 2})
    dense1 = GraphNode("dense1", "dense", {"units": 256})
    dropout = GraphNode("dropout", "dropout", {"rate": 0.5})
    output_node = GraphNode("output", "dense", {"units": 10})

    graph.add_node(input_node)
    graph.add_node(conv1)
    graph.add_node(pool1)
    graph.add_node(conv2)
    graph.add_node(pool2)
    graph.add_node(dense1)
    graph.add_node(dropout)
    graph.add_node(output_node)

    graph.add_edge_by_id("input", "conv1")
    graph.add_edge_by_id("conv1", "pool1")
    graph.add_edge_by_id("pool1", "conv2")
    graph.add_edge_by_id("conv2", "pool2")
    graph.add_edge_by_id("pool2", "dense1")
    graph.add_edge_by_id("dense1", "dropout")
    graph.add_edge_by_id("dropout", "output")

    return graph


@pytest.fixture
def cifar10_task():
    """CIFAR-10 task metadata."""
    return TaskMetadata(
        task_id="cifar10_test",
        dataset_name="CIFAR-10",
        num_classes=10,
        input_size=(3, 32, 32),
        num_samples=50000,
        problem_type="image_classification",
    )


@pytest.fixture
def cifar100_task():
    """CIFAR-100 task metadata."""
    return TaskMetadata(
        task_id="cifar100_test",
        dataset_name="CIFAR-100",
        num_classes=100,
        input_size=(3, 32, 32),
        num_samples=50000,
        problem_type="image_classification",
    )


@pytest.fixture
def mnist_task():
    """MNIST task metadata."""
    return TaskMetadata(
        task_id="mnist_test",
        dataset_name="MNIST",
        num_classes=10,
        input_size=(1, 28, 28),
        num_samples=60000,
        problem_type="image_classification",
    )


# ============================================================================
# ArchitectureTransfer Tests
# ============================================================================


class TestArchitectureTransfer:
    """Test architecture transfer functionality."""

    def test_transfer_architecture_same_input_different_classes(
        self, sample_architecture, cifar10_task, cifar100_task
    ):
        """Test transfer with same input size but different number of classes."""
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=cifar100_task,
            adaptation_strategy="modify_head",
        )

        # Check that architecture was cloned
        assert transferred is not sample_architecture
        assert len(transferred.nodes) == len(sample_architecture.nodes)

        # Check output layer was modified
        output_node = transferred.nodes["output"]
        assert output_node.params["units"] == 100  # CIFAR-100 has 100 classes

    def test_transfer_architecture_different_input(
        self, sample_architecture, cifar10_task, mnist_task
    ):
        """Test transfer with different input size."""
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=mnist_task,
            adaptation_strategy="modify_head",
        )

        # Check input layer was modified
        input_node = transferred.nodes["input"]
        assert input_node.params["input_shape"] == (1, 28, 28)

        # Check output layer remains 10 (same number of classes)
        output_node = transferred.nodes["output"]
        assert output_node.params["units"] == 10

    def test_direct_transfer_strategy(self, sample_architecture, cifar10_task, cifar100_task):
        """Test direct transfer (no modifications)."""
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=cifar100_task,
            adaptation_strategy="direct",
        )

        # Should be identical to source
        assert transferred.nodes["output"].params["units"] == 10  # Unchanged

    def test_capacity_scaling(self, sample_architecture, cifar10_task, cifar100_task):
        """Test capacity scaling during transfer."""
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=cifar100_task,
            adaptation_strategy="full_adapt",
            capacity_scale=1.5,
        )

        # Check conv layers scaled
        assert transferred.nodes["conv1"].params["filters"] == int(64 * 1.5)
        assert transferred.nodes["conv2"].params["filters"] == int(128 * 1.5)

        # Check dense layer scaled
        assert transferred.nodes["dense1"].params["units"] == int(256 * 1.5)

        # Output layer should be 100 (target classes), not scaled
        assert transferred.nodes["output"].params["units"] == 100

    def test_evaluate_transferability_same_dataset(self, cifar10_task):
        """Test transferability score for same dataset."""
        score = ArchitectureTransfer.evaluate_transferability(cifar10_task, cifar10_task)

        assert score == 1.0  # Perfect transfer

    def test_evaluate_transferability_similar_tasks(self, cifar10_task, cifar100_task):
        """Test transferability score for similar tasks."""
        score = ArchitectureTransfer.evaluate_transferability(cifar10_task, cifar100_task)

        # Should be high (same domain, input size)
        assert 0.7 <= score <= 1.0

    def test_evaluate_transferability_different_domains(self, cifar10_task, mnist_task):
        """Test transferability score for different domains."""
        score = ArchitectureTransfer.evaluate_transferability(cifar10_task, mnist_task)

        # Should be moderate (different input, same problem type)
        assert 0.3 <= score <= 0.8

    def test_recommend_transfer_strategy_similar(self, cifar10_task, cifar100_task):
        """Test strategy recommendation for similar tasks."""
        rec = ArchitectureTransfer.recommend_transfer_strategy(cifar10_task, cifar100_task)

        assert "strategy" in rec
        assert "capacity_scale" in rec
        assert "transferability" in rec
        assert "reasoning" in rec

        # For similar tasks, should recommend modify_head or direct
        assert rec["strategy"] in ["direct", "modify_head"]

    def test_recommend_transfer_strategy_distant(self, cifar10_task, mnist_task):
        """Test strategy recommendation for distant tasks."""
        rec = ArchitectureTransfer.recommend_transfer_strategy(cifar10_task, mnist_task)

        # For distant tasks, should recommend full_adapt
        assert rec["strategy"] == "full_adapt"


# ============================================================================
# FineTuningStrategy Tests
# ============================================================================


class TestFineTuningStrategy:
    """Test fine-tuning strategy selection."""

    def test_get_strategy_same_domain(self):
        """Test strategy for same domain transfer."""
        strategy = FineTuningStrategy.get_strategy(
            transfer_type="same_domain",
            model_depth=20,
            target_dataset_size=50000,
        )

        assert strategy["method"] == "freeze_early"
        assert "freeze_ratio" in strategy
        assert "learning_rate" in strategy
        assert strategy["freeze_ratio"] == 0.7  # Freeze most layers

    def test_get_strategy_similar_tasks_small_data(self):
        """Test strategy for similar tasks with small dataset."""
        strategy = FineTuningStrategy.get_strategy(
            transfer_type="similar_tasks",
            model_depth=15,
            target_dataset_size=5000,  # Small dataset
        )

        assert strategy["method"] == "freeze_early"
        assert strategy["freeze_ratio"] == 0.5
        # Small data -> more conservative

    def test_get_strategy_similar_tasks_large_data(self):
        """Test strategy for similar tasks with large dataset."""
        strategy = FineTuningStrategy.get_strategy(
            transfer_type="similar_tasks",
            model_depth=15,
            target_dataset_size=100000,  # Large dataset
        )

        assert strategy["method"] == "differential_lr"
        assert "early_lr" in strategy
        assert "late_lr" in strategy
        # Can afford more training with large data

    def test_get_strategy_distant_tasks(self):
        """Test strategy for distant tasks."""
        strategy = FineTuningStrategy.get_strategy(
            transfer_type="distant_tasks",
            model_depth=30,
            target_dataset_size=50000,
        )

        assert strategy["method"] == "progressive_unfreezing"
        assert "unfreeze_schedule" in strategy
        assert "initial_freeze_ratio" in strategy

    def test_generate_freeze_mask(self):
        """Test freeze mask generation."""
        num_layers = 20
        freeze_ratio = 0.5

        mask = FineTuningStrategy.generate_freeze_mask(num_layers, freeze_ratio)

        assert len(mask) == num_layers
        assert all(isinstance(x, bool) for x in mask)

        # First 50% should be frozen
        num_frozen = sum(mask)
        expected_frozen = int(num_layers * freeze_ratio)
        assert num_frozen == expected_frozen

        # Check pattern: early layers frozen, later layers trainable
        assert all(mask[:expected_frozen])  # All True
        assert not any(mask[expected_frozen:])  # All False


# ============================================================================
# MultiTaskNAS Tests
# ============================================================================


class TestMultiTaskNAS:
    """Test multi-task NAS functionality."""

    def test_initialization(self, cifar10_task, cifar100_task):
        """Test MultiTaskNAS initialization."""
        tasks = [cifar10_task, cifar100_task]

        search_space = SearchSpace(
            name="test_space",
            layers=[
                Layer.conv2d(filters=[32, 64], kernel_size=[3]),
                Layer.dense(units=[128, 256]),
            ],
        )

        multi_nas = MultiTaskNAS(
            tasks=tasks,
            search_space=search_space,
            task_weights=[0.6, 0.4],
        )

        assert len(multi_nas.tasks) == 2
        assert len(multi_nas.task_weights) == 2
        assert abs(sum(multi_nas.task_weights) - 1.0) < 1e-6  # Normalized

    def test_default_weights(self, cifar10_task, cifar100_task, mnist_task):
        """Test default equal weights."""
        tasks = [cifar10_task, cifar100_task, mnist_task]

        search_space = SearchSpace(
            name="test_space",
            layers=[Layer.dense(units=[128])],
        )

        multi_nas = MultiTaskNAS(tasks=tasks, search_space=search_space)

        # Should have equal weights
        assert all(abs(w - 1.0 / 3) < 1e-6 for w in multi_nas.task_weights)

    def test_evaluate_multi_task_fitness(self, sample_architecture, cifar10_task, cifar100_task):
        """Test multi-task fitness evaluation."""
        tasks = [cifar10_task, cifar100_task]

        search_space = SearchSpace(
            name="test_space",
            layers=[Layer.dense(units=[128])],
        )

        multi_nas = MultiTaskNAS(
            tasks=tasks,
            search_space=search_space,
            task_weights=[0.7, 0.3],
        )

        # Mock evaluator
        def mock_evaluator(arch, task):
            # Return different fitness for different tasks
            if task.task_id == "cifar10_test":
                return 0.9
            else:
                return 0.8

        results = multi_nas.evaluate_multi_task_fitness(sample_architecture, mock_evaluator)

        # Check all tasks evaluated
        assert "cifar10_test" in results
        assert "cifar100_test" in results
        assert "weighted_average" in results

        # Check weighted average
        expected = 0.7 * 0.9 + 0.3 * 0.8
        assert abs(results["weighted_average"] - expected) < 1e-6

    def test_create_multi_task_evaluator(self, sample_architecture, cifar10_task, cifar100_task):
        """Test multi-task evaluator creation."""
        tasks = [cifar10_task, cifar100_task]

        search_space = SearchSpace(
            name="test_space",
            layers=[Layer.dense(units=[128])],
        )

        multi_nas = MultiTaskNAS(tasks=tasks, search_space=search_space)

        # Mock base evaluator
        def base_evaluator(arch):
            return 0.85

        evaluator = multi_nas.create_multi_task_evaluator(base_evaluator)

        # Test that evaluator works
        fitness = evaluator(sample_architecture)

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestTransferLearningIntegration:
    """Integration tests for complete transfer learning workflow."""

    def test_end_to_end_transfer(self, sample_architecture, cifar10_task, cifar100_task):
        """Test complete transfer workflow."""
        # 1. Evaluate transferability
        score = ArchitectureTransfer.evaluate_transferability(cifar10_task, cifar100_task)
        assert 0.0 <= score <= 1.0

        # 2. Get recommendation
        rec = ArchitectureTransfer.recommend_transfer_strategy(cifar10_task, cifar100_task)
        assert "strategy" in rec

        # 3. Transfer architecture
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=cifar100_task,
            adaptation_strategy=rec["strategy"],
            capacity_scale=rec["capacity_scale"],
        )

        # 4. Get fine-tuning strategy
        ft_strategy = FineTuningStrategy.get_strategy(
            transfer_type="similar_tasks",
            model_depth=len(transferred.nodes),
            target_dataset_size=50000,
        )

        assert ft_strategy["method"] in [
            "freeze_early",
            "differential_lr",
            "full",
            "progressive_unfreezing",
        ]

        # 5. Verify transferred architecture is valid
        assert transferred.is_valid_dag()
        assert transferred.nodes["output"].params["units"] == 100

    def test_progressive_transfer_chain(
        self, sample_architecture, mnist_task, cifar10_task, cifar100_task
    ):
        """Test progressive transfer through task chain."""
        task_chain = [mnist_task, cifar10_task, cifar100_task]

        current_arch = sample_architecture

        for i in range(len(task_chain) - 1):
            source = task_chain[i]
            target = task_chain[i + 1]

            # Transfer to next task
            current_arch = ArchitectureTransfer.transfer_architecture(
                source_arch=current_arch,
                source_task=source,
                target_task=target,
                adaptation_strategy="modify_head",
            )

            # Verify valid
            assert current_arch.is_valid_dag()

        # Final architecture should match last task
        final_task = task_chain[-1]
        assert current_arch.nodes["output"].params["units"] == final_task.num_classes


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_adaptation_strategy(self, sample_architecture, cifar10_task, cifar100_task):
        """Test invalid adaptation strategy raises error."""
        with pytest.raises(ValueError, match="Unknown adaptation strategy"):
            ArchitectureTransfer.transfer_architecture(
                source_arch=sample_architecture,
                source_task=cifar10_task,
                target_task=cifar100_task,
                adaptation_strategy="invalid_strategy",
            )

    def test_zero_capacity_scale(self, sample_architecture, cifar10_task, cifar100_task):
        """Test very small capacity scale."""
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=cifar100_task,
            adaptation_strategy="full_adapt",
            capacity_scale=0.5,  # Reduce by half
        )

        # Layers should be smaller
        assert transferred.nodes["conv1"].params["filters"] < 64
        assert transferred.nodes["dense1"].params["units"] < 256

    def test_large_capacity_scale(self, sample_architecture, cifar10_task, cifar100_task):
        """Test large capacity scale."""
        transferred = ArchitectureTransfer.transfer_architecture(
            source_arch=sample_architecture,
            source_task=cifar10_task,
            target_task=cifar100_task,
            adaptation_strategy="full_adapt",
            capacity_scale=2.0,  # Double size
        )

        # Layers should be larger
        assert transferred.nodes["conv1"].params["filters"] > 64
        assert transferred.nodes["dense1"].params["units"] > 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
