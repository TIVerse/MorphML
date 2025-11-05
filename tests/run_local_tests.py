#!/usr/bin/env python3
"""Run comprehensive local tests for MorphML.

Tests all components without requiring external services.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class LocalTestRunner:
    """Run comprehensive local tests."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        
        print("\n" + "="*80)
        print(" "*25 + "🧪 MorphML Local Test Suite")
        print("="*80 + "\n")
    
    def run_test(self, name: str, test_func):
        """Run a single test."""
        print(f"Testing {name}...", end=" ")
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"✅ PASSED ({duration:.2f}s)")
                self.passed += 1
                self.results[name] = {"status": "passed", "duration": duration}
            else:
                print(f"❌ FAILED ({duration:.2f}s)")
                self.failed += 1
                self.results[name] = {"status": "failed", "duration": duration}
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.failed += 1
            self.results[name] = {"status": "error", "error": str(e)}
    
    def test_phase1_dsl(self) -> bool:
        """Test Phase 1: DSL and search space."""
        from morphml.core.dsl import Layer, SearchSpace
        
        # Create search space
        space = SearchSpace("test_space")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64, kernel_size=3),
            Layer.maxpool2d(pool_size=2),
            Layer.conv2d(filters=128, kernel_size=3),
            Layer.flatten(),
            Layer.dense(units=256),
            Layer.dropout(rate=0.5),
            Layer.output(units=10)
        )
        
        # Sample architecture
        graph = space.sample()
        
        # Verify structure
        assert graph is not None
        assert len(graph.layers) > 0
        
        # Test JSON serialization
        json_str = graph.to_json()
        assert len(json_str) > 0
        
        return True
    
    def test_phase1_optimizers(self) -> bool:
        """Test Phase 1: Basic optimizers."""
        from morphml.core.dsl import Layer, SearchSpace
        from morphml.optimizers import GeneticAlgorithm, RandomSearch
        
        # Create search space
        space = SearchSpace("test_optimizer")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=32),
            Layer.output(units=10)
        )
        
        # Define simple evaluator
        def evaluator(graph):
            # Simple fitness based on layer count
            return len(graph.layers) / 10.0
        
        # Test Random Search
        rs = RandomSearch(space, evaluator, num_samples=10)
        rs.search()
        assert len(rs.history) == 10
        
        # Test Genetic Algorithm
        ga = GeneticAlgorithm(space, evaluator, population_size=10, num_generations=3)
        ga.search()
        assert len(ga.history) > 0
        
        return True
    
    def test_phase2_advanced_optimizers(self) -> bool:
        """Test Phase 2: Advanced optimizers."""
        from morphml.core.dsl import Layer, SearchSpace
        from morphml.optimizers.evolutionary import DifferentialEvolution, CMA_ES
        
        # Create search space
        space = SearchSpace("test_advanced")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=32),
            Layer.output(units=10)
        )
        
        # Simple evaluator
        def evaluator(graph):
            return 0.8 + hash(str(graph.layers)) % 100 / 500.0
        
        # Test Differential Evolution
        de = DifferentialEvolution(space, evaluator, population_size=10, num_generations=3)
        de.search()
        assert len(de.history) > 0
        
        # Test CMA-ES (if available)
        try:
            cma = CMA_ES(space, evaluator, population_size=10, num_generations=3)
            cma.search()
            assert len(cma.history) > 0
        except ImportError:
            pass  # CMA not available
        
        return True
    
    def test_phase3_schedulers(self) -> bool:
        """Test Phase 3: Task schedulers."""
        from morphml.distributed import (
            FIFOScheduler,
            LoadBalancingScheduler,
            AdaptiveScheduler,
            WorkerInfo,
            Task,
        )
        from morphml.core.dsl import Layer, SearchSpace
        
        # Create test data
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        
        workers = [
            WorkerInfo("w1", "host1", 50052, 2, status="idle"),
            WorkerInfo("w2", "host2", 50052, 2, status="idle"),
        ]
        
        task = Task("task-1", space.sample())
        
        # Test FIFO
        fifo = FIFOScheduler()
        worker = fifo.assign_task(task, workers)
        assert worker is not None
        
        # Test Load Balancing
        lb = LoadBalancingScheduler()
        worker = lb.assign_task(task, workers)
        assert worker is not None
        
        # Test Adaptive
        adaptive = AdaptiveScheduler()
        worker = adaptive.assign_task(task, workers)
        assert worker is not None
        
        return True
    
    def test_phase3_fault_tolerance(self) -> bool:
        """Test Phase 3: Fault tolerance."""
        from morphml.distributed import (
            FaultToleranceManager,
            CircuitBreaker,
            FailureType,
            Task,
        )
        from morphml.core.dsl import Layer, SearchSpace
        
        # Create fault tolerance manager
        ft_manager = FaultToleranceManager({"max_retries": 3})
        
        # Create test task
        space = SearchSpace("test")
        space.add_layers(Layer.input(shape=(3, 32, 32)), Layer.output(units=10))
        task = Task("task-1", space.sample())
        
        # Test task failure handling
        should_retry = ft_manager.handle_task_failure(
            task, FailureType.NETWORK_ERROR, "timeout"
        )
        assert should_retry is True
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open() is True
        
        return True
    
    def test_phase3_health_monitor(self) -> bool:
        """Test Phase 3: Health monitoring."""
        from morphml.distributed import HealthMonitor, get_system_health
        
        # Create monitor
        monitor = HealthMonitor()
        
        # Get health metrics
        metrics = monitor.get_health_metrics()
        assert metrics is not None
        assert hasattr(metrics, "cpu_percent")
        assert hasattr(metrics, "memory_percent")
        
        # Test convenience function
        health = get_system_health()
        assert "cpu_percent" in health
        
        # Get system info
        info = monitor.get_system_info()
        assert "platform" in info
        
        return True
    
    def test_benchmarks(self) -> bool:
        """Test benchmark suite."""
        from benchmarks.distributed.benchmark_schedulers import SchedulerBenchmark
        
        # Run small benchmark
        benchmark = SchedulerBenchmark(num_tasks=50, num_workers=5)
        results = benchmark.benchmark_scheduler("fifo")
        
        assert "throughput" in results
        assert results["tasks_assigned"] > 0
        
        return True
    
    def test_storage_mocks(self) -> bool:
        """Test storage components (mock mode)."""
        from morphml.distributed.storage.checkpointing import CheckpointManager
        
        # Test checkpoint manager with mocks
        # (Real tests would need actual storage backends)
        # This just verifies the classes can be imported
        
        assert CheckpointManager is not None
        
        return True
    
    def run_all(self):
        """Run all tests."""
        start_time = time.time()
        
        # Phase 1 tests
        print("\n" + "─"*80)
        print("📦 PHASE 1: Foundation")
        print("─"*80)
        self.run_test("DSL and Search Space", self.test_phase1_dsl)
        self.run_test("Basic Optimizers", self.test_phase1_optimizers)
        
        # Phase 2 tests
        print("\n" + "─"*80)
        print("🧬 PHASE 2: Advanced Optimizers")
        print("─"*80)
        self.run_test("Advanced Optimizers", self.test_phase2_advanced_optimizers)
        
        # Phase 3 tests
        print("\n" + "─"*80)
        print("🌐 PHASE 3: Distributed System")
        print("─"*80)
        self.run_test("Task Schedulers", self.test_phase3_schedulers)
        self.run_test("Fault Tolerance", self.test_phase3_fault_tolerance)
        self.run_test("Health Monitoring", self.test_phase3_health_monitor)
        self.run_test("Storage Components", self.test_storage_mocks)
        
        # Benchmark tests
        print("\n" + "─"*80)
        print("📊 BENCHMARKS")
        print("─"*80)
        self.run_test("Benchmark Suite", self.test_benchmarks)
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Passed:  {self.passed}")
        print(f"❌ Failed:  {self.failed}")
        print(f"⏭️  Skipped: {self.skipped}")
        print(f"⏱️  Duration: {total_time:.2f}s")
        print("="*80 + "\n")
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! 🎉\n")
            print("MorphML is working correctly on your system.\n")
            return True
        else:
            print(f"⚠️  {self.failed} test(s) failed. See details above.\n")
            return False


def main():
    """Run local tests."""
    runner = LocalTestRunner()
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
