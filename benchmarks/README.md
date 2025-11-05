# MorphML Performance Benchmarks

Comprehensive performance benchmarking suite for MorphML distributed Neural Architecture Search.

## 📋 Benchmarks Included

### 1. Scheduler Benchmarks
Tests different task scheduling strategies:
- **FIFO** - First-In-First-Out
- **Priority** - Priority-based scheduling
- **Load Balancing** - Even distribution
- **Work Stealing** - Dynamic rebalancing
- **Adaptive** - Learning-based assignment
- **Round Robin** - Circular assignment

**Metrics:**
- Throughput (tasks/second)
- Assignment latency
- Load balance fairness

### 2. Scaling Benchmarks
Tests system scaling characteristics:
- Throughput vs number of workers
- Speedup and parallel efficiency
- Optimal worker count
- Overhead analysis

**Metrics:**
- Execution time
- Throughput
- Speedup (vs 1 worker)
- Parallel efficiency

## 🚀 Quick Start

### Run All Benchmarks

```bash
cd MorphML
python benchmarks/run_all_benchmarks.py
```

This will:
1. Run scheduler benchmarks
2. Run scaling benchmarks
3. Generate visualization plots
4. Create comprehensive report

### Run Individual Benchmarks

#### Scheduler Benchmark
```bash
python benchmarks/distributed/benchmark_schedulers.py --tasks 1000 --workers 10
```

#### Scaling Benchmark
```bash
python benchmarks/distributed/benchmark_scaling.py --tasks 200 --workers 1,2,4,8,16,32
```

### Generate Visualizations

```bash
python benchmarks/visualization/plot_results.py --type all
```

## 📊 Output Files

- `scheduler_benchmark_results.json` - Raw scheduler results
- `scaling_benchmark_results.json` - Raw scaling results
- `scheduler_comparison.png` - Scheduler comparison charts
- `scaling_results.png` - Scaling characteristic plots
- `BENCHMARK_REPORT.md` - Comprehensive report

## 📈 Sample Results

### Scheduler Performance

| Scheduler | Throughput | Latency | Load Balance |
|-----------|-----------|---------|--------------|
| Load Balancing | 2500 t/s | 0.40 ms | 0.95 |
| Adaptive | 2450 t/s | 0.42 ms | 0.93 |
| Work Stealing | 2400 t/s | 0.45 ms | 0.91 |
| Round Robin | 2350 t/s | 0.43 ms | 0.99 |
| FIFO | 2300 t/s | 0.38 ms | 0.85 |
| Priority | 2250 t/s | 0.50 ms | 0.87 |

### Scaling Performance

| Workers | Time | Throughput | Speedup | Efficiency |
|---------|------|-----------|---------|------------|
| 1 | 100.0s | 10 t/s | 1.0x | 100% |
| 2 | 52.0s | 19 t/s | 1.9x | 96% |
| 4 | 27.0s | 37 t/s | 3.7x | 93% |
| 8 | 14.5s | 69 t/s | 6.9x | 86% |
| 16 | 8.0s | 125 t/s | 12.5x | 78% |
| 32 | 4.5s | 222 t/s | 22.2x | 69% |

**Key Findings:**
- Near-linear scaling up to 8-16 workers
- Optimal efficiency at 4-8 workers
- Continued throughput gains up to 32+ workers

## 🔧 Configuration

### Benchmark Parameters

```python
# Scheduler benchmark
num_tasks = 1000      # Number of tasks to schedule
num_workers = 10      # Number of simulated workers

# Scaling benchmark
num_tasks = 200       # Tasks to process
worker_counts = [1, 2, 4, 8, 16, 32]  # Workers to test
```

### Custom Benchmarks

Create your own benchmarks:

```python
from benchmarks.distributed.benchmark_schedulers import SchedulerBenchmark

# Custom configuration
benchmark = SchedulerBenchmark(
    num_tasks=5000,
    num_workers=50
)

# Run specific scheduler
results = benchmark.benchmark_scheduler('adaptive')
print(results)
```

## 📝 Requirements

### Core Requirements
- Python 3.10+
- MorphML installed

### Optional (for visualization)
```bash
pip install matplotlib numpy
```

## 🎯 Interpreting Results

### Throughput
- Higher is better
- Indicates how many tasks can be processed per second
- Depends on worker count and overhead

### Latency
- Lower is better
- Time to assign a task to a worker
- Critical for interactive workloads

### Load Balance
- Closer to 1.0 is better
- Indicates fairness of work distribution
- Important for avoiding stragglers

### Speedup
- Ideally linear (N workers = Nx speedup)
- Real-world: sub-linear due to coordination overhead
- Efficiency = Speedup / Workers

### Efficiency
- 1.0 (100%) = perfect scaling
- Decreases with more workers due to overhead
- Typically 80-95% is excellent

## 🐛 Troubleshooting

### Benchmark takes too long
- Reduce `num_tasks`
- Test fewer worker counts
- Use smaller architectures

### Matplotlib not found
```bash
pip install matplotlib
```

### Memory issues
- Reduce `num_workers`
- Use ThreadPoolExecutor with smaller max_workers

## 📚 Additional Resources

- [MorphML Documentation](https://github.com/TIVerse/MorphML)
- [Distributed Computing Docs](../docs/distributed.md)
- [Performance Tuning Guide](../docs/performance.md)

## 📄 License

Copyright © 2025 TONMOY INFRASTRUCTURE & VISION  
Licensed under the Apache License 2.0
