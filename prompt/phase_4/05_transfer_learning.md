# Component 5: Transfer Learning Across Tasks

**Duration:** Week 8  
**LOC Target:** ~2,000  
**Dependencies:** Components 1-4

---

## 🎯 Objective

Transfer architectures across related tasks:
1. **Architecture Adaptation** - Modify for new task
2. **Fine-tuning Strategies** - Transfer learning protocols
3. **Domain Adaptation** - Handle distribution shifts
4. **Multi-task Learning** - Shared architecture search

---

## 📋 Files to Create

### 1. `meta_learning/transfer.py` (~1,000 LOC)

```python
class ArchitectureTransfer:
    """
    Transfer architectures between tasks.
    
    Methods:
    1. Direct transfer: Use architecture as-is
    2. Adaptation: Modify layers for new task
    3. Progressive transfer: Gradually adapt
    """
    
    @staticmethod
    def transfer_architecture(
        source_arch: ModelGraph,
        source_task: TaskMetadata,
        target_task: TaskMetadata
    ) -> ModelGraph:
        """
        Adapt architecture for new task.
        
        Changes:
        - Update input layer for new resolution
        - Update output layer for new number of classes
        - Optionally adjust capacity
        """
        transferred = source_arch.clone()
        
        # Modify input
        if source_task.input_size != target_task.input_size:
            input_node = transferred.get_input_node()
            input_node.params['input_shape'] = target_task.input_size
        
        # Modify output
        if source_task.num_classes != target_task.num_classes:
            output_node = transferred.get_output_node()
            output_node.params['units'] = target_task.num_classes
        
        logger.info(
            f"Transferred architecture from {source_task.dataset_name} "
            f"to {target_task.dataset_name}"
        )
        
        return transferred
    
    @staticmethod
    def evaluate_transferability(
        source_task: TaskMetadata,
        target_task: TaskMetadata
    ) -> float:
        """
        Estimate how well architectures will transfer.
        
        Based on task similarity.
        """
        # Simple heuristic
        if source_task.problem_type != target_task.problem_type:
            return 0.3  # Different problem types transfer poorly
        
        # Similar datasets transfer well
        if source_task.dataset_name == target_task.dataset_name:
            return 1.0
        
        # Check similarity
        size_ratio = min(source_task.num_samples, target_task.num_samples) / \
                    max(source_task.num_samples, target_task.num_samples)
        
        class_ratio = min(source_task.num_classes, target_task.num_classes) / \
                     max(source_task.num_classes, target_task.num_classes)
        
        transferability = (size_ratio + class_ratio) / 2
        
        return transferability
```

---

### 2. `meta_learning/fine_tuning.py` (~500 LOC)

```python
class FineTuningStrategy:
    """
    Fine-tuning protocols for transferred architectures.
    
    Strategies:
    1. Full fine-tuning
    2. Freeze early layers
    3. Layer-wise learning rates
    """
    
    @staticmethod
    def fine_tune_transferred(
        model: nn.Module,
        target_dataset: Dataset,
        strategy: str = 'freeze_early'
    ) -> nn.Module:
        """Fine-tune transferred model."""
        if strategy == 'freeze_early':
            # Freeze first 50% of layers
            layers = list(model.parameters())
            freeze_until = len(layers) // 2
            
            for i, param in enumerate(layers):
                if i < freeze_until:
                    param.requires_grad = False
        
        # Train
        trainer = Trainer(model, target_dataset)
        trained_model = trainer.train(num_epochs=50)
        
        return trained_model
```

---

### 3. `meta_learning/multi_task.py` (~500 LOC)

```python
class MultiTaskNAS:
    """
    Search for architectures that work across multiple tasks.
    
    Optimizes for average performance on task distribution.
    """
    
    def __init__(
        self,
        tasks: List[TaskMetadata],
        search_space: SearchSpace,
        optimizer: BaseOptimizer
    ):
        self.tasks = tasks
        self.search_space = search_space
        self.optimizer = optimizer
    
    def search(self) -> ModelGraph:
        """
        Find architecture that works well across all tasks.
        
        Fitness = weighted average of performance on tasks
        """
        def multi_task_fitness(arch: ModelGraph) -> float:
            fitnesses = []
            
            for task in self.tasks:
                # Adapt architecture
                adapted = ArchitectureTransfer.transfer_architecture(
                    arch, self.tasks[0], task
                )
                
                # Evaluate
                fitness = evaluate_on_task(adapted, task)
                fitnesses.append(fitness)
            
            # Average performance
            return np.mean(fitnesses)
        
        self.optimizer.evaluate = multi_task_fitness
        
        best = self.optimizer.optimize()
        
        return best.genome
```

---

## 🧪 Tests

```python
def test_architecture_transfer():
    """Test architecture transfer."""
    source_task = TaskMetadata(
        dataset_name='CIFAR-10',
        num_classes=10,
        input_size=(3, 32, 32)
    )
    
    target_task = TaskMetadata(
        dataset_name='CIFAR-100',
        num_classes=100,
        input_size=(3, 32, 32)
    )
    
    arch = create_sample_architecture()
    
    transferred = ArchitectureTransfer.transfer_architecture(
        arch, source_task, target_task
    )
    
    # Check output layer updated
    output = transferred.get_output_node()
    assert output.params['units'] == 100
```

---

## ✅ Deliverables

- [ ] Architecture transfer methods
- [ ] Transferability estimation
- [ ] Fine-tuning strategies
- [ ] Multi-task NAS
- [ ] Demonstrate successful transfer

---

**Phase 4 Complete!** 🎉

Total Phase 4 LOC: ~15,000 production code

**Next Phase:** Phase 5 - Ecosystem & Polish
