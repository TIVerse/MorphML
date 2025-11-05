"""Performance metrics for benchmarking.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import Dict, List, Optional

import numpy as np

from morphml.logging_config import get_logger

logger = get_logger(__name__)


def sample_efficiency(history: List[Dict], target_fitness: float) -> int:
    """
    Calculate number of evaluations to reach target fitness.
    
    Args:
        history: Optimization history
        target_fitness: Target fitness threshold
        
    Returns:
        Number of evaluations to reach target, or -1 if not reached
    """
    for i, entry in enumerate(history):
        if entry.get('best_fitness', 0) >= target_fitness:
            return i + 1
    return -1


def convergence_rate(history: List[Dict]) -> float:
    """
    Calculate convergence rate (improvement per evaluation).
    
    Args:
        history: Optimization history
        
    Returns:
        Average improvement per evaluation
    """
    if len(history) < 2:
        return 0.0
    
    fitnesses = [entry.get('best_fitness', 0) for entry in history]
    improvements = np.diff(fitnesses)
    
    return np.mean(improvements[improvements > 0]) if len(improvements) > 0 else 0.0


def final_best_fitness(history: List[Dict]) -> float:
    """
    Get final best fitness achieved.
    
    Args:
        history: Optimization history
        
    Returns:
        Best fitness value
    """
    if not history:
        return 0.0
    
    return max(entry.get('best_fitness', 0) for entry in history)


def time_efficiency(history: List[Dict], target_fitness: float) -> float:
    """
    Calculate time to reach target fitness.
    
    Args:
        history: Optimization history
        target_fitness: Target fitness threshold
        
    Returns:
        Time in seconds, or -1 if not reached
    """
    for entry in history:
        if entry.get('best_fitness', 0) >= target_fitness:
            return entry.get('time_elapsed', -1)
    return -1.0


def stability_score(histories: List[List[Dict]]) -> float:
    """
    Calculate stability across multiple runs.
    
    Args:
        histories: List of optimization histories from multiple runs
        
    Returns:
        Stability score (1 - coefficient of variation of final fitness)
    """
    if not histories:
        return 0.0
    
    final_fitnesses = [final_best_fitness(h) for h in histories]
    
    mean = np.mean(final_fitnesses)
    std = np.std(final_fitnesses)
    
    if mean == 0:
        return 0.0
    
    cv = std / mean  # Coefficient of variation
    stability = max(0.0, 1.0 - cv)
    
    return stability


def area_under_curve(history: List[Dict]) -> float:
    """
    Calculate area under convergence curve (AUC).
    
    Higher AUC indicates faster convergence.
    
    Args:
        history: Optimization history
        
    Returns:
        AUC value
    """
    if not history:
        return 0.0
    
    fitnesses = [entry.get('best_fitness', 0) for entry in history]
    
    # Trapezoidal integration
    auc = np.trapz(fitnesses)
    
    return auc


def success_rate(histories: List[List[Dict]], target_fitness: float) -> float:
    """
    Calculate success rate (fraction of runs reaching target).
    
    Args:
        histories: List of optimization histories
        target_fitness: Target fitness threshold
        
    Returns:
        Success rate [0, 1]
    """
    if not histories:
        return 0.0
    
    successes = sum(1 for h in histories if final_best_fitness(h) >= target_fitness)
    
    return successes / len(histories)


def compute_all_metrics(
    history: List[Dict],
    target_fitness: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all available metrics for an optimization run.
    
    Args:
        history: Optimization history
        target_fitness: Optional target fitness for efficiency metrics
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'final_best_fitness': final_best_fitness(history),
        'convergence_rate': convergence_rate(history),
        'auc': area_under_curve(history),
        'num_evaluations': len(history),
    }
    
    if target_fitness is not None:
        metrics['sample_efficiency'] = sample_efficiency(history, target_fitness)
        metrics['time_efficiency'] = time_efficiency(history, target_fitness)
    
    return metrics


def compare_optimizers(
    results: Dict[str, List[Dict]],
    target_fitness: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple optimizers across metrics.
    
    Args:
        results: Dict mapping optimizer names to their histories
        target_fitness: Optional target for efficiency metrics
        
    Returns:
        Dict of optimizer names to their metric dictionaries
    """
    comparison = {}
    
    for optimizer_name, history in results.items():
        comparison[optimizer_name] = compute_all_metrics(history, target_fitness)
    
    return comparison
