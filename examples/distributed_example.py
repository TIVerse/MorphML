"""
MorphML Distributed Execution Example

This example demonstrates distributed architecture search using master-worker pattern.

Run master:
    python examples/distributed_example.py --mode master --num-workers 2

Run worker (in separate terminals):
    python examples/distributed_example.py --mode worker --worker-id worker-1
    python examples/distributed_example.py --mode worker --worker-id worker-2

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
Repository: https://github.com/TIVerse/MorphML
"""

import argparse
import time

from morphml.core.dsl import Layer, SearchSpace
from morphml.core.graph import ModelGraph
from morphml.distributed import MasterNode, WorkerNode
from morphml.evaluation import HeuristicEvaluator
from morphml.optimizers import GeneticAlgorithm


def create_search_space() -> SearchSpace:
    """Create search space for CIFAR-10 classification."""
    space = SearchSpace("cifar10_distributed")
    
    space.add_layers(
        # Input
        Layer.input(shape=(3, 32, 32)),
        
        # Conv Block 1
        Layer.conv2d(filters=[32, 64, 128], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        
        # Conv Block 2
        Layer.conv2d(filters=[64, 128, 256], kernel_size=[3, 5]),
        Layer.relu(),
        Layer.batchnorm(),
        Layer.maxpool(pool_size=2),
        
        # Dense layers
        Layer.dense(units=[128, 256, 512]),
        Layer.dropout(rate=[0.3, 0.5]),
        Layer.dense(units=[64, 128]),
        
        # Output
        Layer.output(units=10),
    )
    
    return space


def run_master(num_workers: int = 2, num_generations: int = 20) -> None:
    """
    Run master node.
    
    Args:
        num_workers: Number of workers to wait for
        num_generations: Number of optimization generations
    """
    print("=" * 80)
    print("MORPHML DISTRIBUTED NAS - MASTER NODE")
    print("=" * 80)
    
    # Create search space
    print("\n[1/4] Creating search space...")
    space = create_search_space()
    print(f"      Search space: {space.name}")
    print(f"      Estimated architectures: ~10^6")
    
    # Create optimizer
    print("\n[2/4] Creating optimizer...")
    optimizer = GeneticAlgorithm(
        search_space=space,
        population_size=20,
        num_generations=num_generations,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism=3,
    )
    print(f"      Optimizer: Genetic Algorithm")
    print(f"      Population: 20")
    print(f"      Generations: {num_generations}")
    
    # Create master
    print("\n[3/4] Starting master node...")
    config = {
        "host": "0.0.0.0",
        "port": 50051,
        "num_workers": num_workers,
        "heartbeat_interval": 10,
        "task_timeout": 300,
    }
    
    master = MasterNode(optimizer, config)
    master.start()
    
    print(f"      Master listening on port {config['port']}")
    print(f"      Waiting for {num_workers} workers...")
    
    try:
        # Run experiment with progress callback
        print(f"\n[4/4] Running distributed NAS experiment...")
        print(f"      Target: {num_generations} generations")
        print()
        
        def progress_callback(generation: int, stats: dict) -> None:
            """Print progress."""
            print(
                f"  Gen {generation:3d}/{num_generations}: "
                f"best={stats['best_fitness']:.4f}, "
                f"avg={stats['avg_fitness']:.4f}, "
                f"min={stats['min_fitness']:.4f}"
            )
        
        start_time = time.time()
        best_architectures = master.run_experiment(
            num_generations=num_generations, callback=progress_callback
        )
        duration = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"\nDuration: {duration:.1f}s ({duration/60:.1f} minutes)")
        print(f"Total evaluations: {master.total_evaluations}")
        print(f"Throughput: {master.total_evaluations/duration:.2f} evals/second")
        
        # Statistics
        stats = master.get_statistics()
        print(f"\nTasks completed: {stats['tasks_completed']}")
        print(f"Tasks failed: {stats['tasks_failed']}")
        print(f"Workers used: {stats['workers_total']}")
        
        # Best architectures
        print(f"\nTop {len(best_architectures)} architectures found:")
        for i, ind in enumerate(best_architectures[:5], 1):
            print(f"  {i}. Fitness: {ind.fitness:.4f}, Params: {ind.graph.estimate_parameters():,}")
        
        # Save best
        if best_architectures:
            best = best_architectures[0]
            best.graph.save("best_architecture_distributed.json")
            print(f"\nBest architecture saved to: best_architecture_distributed.json")
    
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    
    finally:
        print("\nShutting down master...")
        master.stop()
        print("Master stopped")


def run_worker(worker_id: str, master_host: str = "localhost") -> None:
    """
    Run worker node.
    
    Args:
        worker_id: Unique worker identifier
        master_host: Master node hostname/IP
    """
    print("=" * 80)
    print(f"MORPHML DISTRIBUTED NAS - WORKER NODE")
    print("=" * 80)
    print(f"\nWorker ID: {worker_id}")
    print(f"Master: {master_host}:50051")
    
    # Create evaluator
    print("\n[1/2] Creating evaluator...")
    evaluator = HeuristicEvaluator()
    print("      Using heuristic evaluator (fast proxy)")
    
    # Define evaluation function
    def evaluate(graph: ModelGraph) -> dict:
        """Evaluate architecture."""
        fitness = evaluator(graph)
        params = graph.estimate_parameters()
        
        return {
            "fitness": fitness,
            "val_accuracy": fitness,
            "params": params,
        }
    
    # Create worker
    print("\n[2/2] Starting worker node...")
    config = {
        "worker_id": worker_id,
        "master_host": master_host,
        "master_port": 50051,
        "port": 50052 + hash(worker_id) % 100,  # Unique port per worker
        "num_gpus": 1,
        "evaluator": evaluate,
        "heartbeat_interval": 10,
    }
    
    worker = WorkerNode(config)
    
    try:
        worker.start()
        print(f"      Worker started successfully")
        print(f"      Listening on port {config['port']}")
        print(f"      GPUs: {config['num_gpus']}")
        print("\nWorker is ready. Waiting for tasks from master...")
        print("(Press Ctrl+C to stop)\n")
        
        # Keep worker running
        worker.wait_for_shutdown()
    
    except KeyboardInterrupt:
        print("\n\nWorker interrupted by user")
    
    except Exception as e:
        print(f"\n\nWorker error: {e}")
    
    finally:
        print("\nShutting down worker...")
        worker.stop()
        print("Worker stopped")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MorphML Distributed NAS Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["master", "worker"],
        required=True,
        help="Run as master or worker",
    )
    
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of workers (master only)"
    )
    
    parser.add_argument(
        "--num-generations",
        type=int,
        default=20,
        help="Number of generations (master only)",
    )
    
    parser.add_argument(
        "--worker-id", type=str, default="worker-1", help="Worker ID (worker only)"
    )
    
    parser.add_argument(
        "--master-host", type=str, default="localhost", help="Master host (worker only)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "master":
        run_master(num_workers=args.num_workers, num_generations=args.num_generations)
    else:
        run_worker(worker_id=args.worker_id, master_host=args.master_host)


if __name__ == "__main__":
    main()
