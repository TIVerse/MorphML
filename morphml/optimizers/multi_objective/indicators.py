"""Quality indicators for multi-objective optimization.

This module provides metrics to assess the quality of Pareto fronts including:
- Hypervolume (S-metric): Volume of objective space dominated by Pareto front
- Inverted Generational Distance (IGD): Average distance to reference Pareto front
- Spacing: Distribution uniformity of solutions
- Spread: Extent of Pareto front coverage

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

from typing import List, Optional, Tuple

import numpy as np

from morphml.logging_config import get_logger
from morphml.optimizers.multi_objective.nsga2 import MultiObjectiveIndividual

logger = get_logger(__name__)


class QualityIndicators:
    """
    Quality indicators for multi-objective optimization.
    
    Provides metrics to evaluate and compare Pareto fronts.
    
    Example:
        >>> indicators = QualityIndicators()
        >>> hv = indicators.hypervolume(pareto_front, reference_point)
        >>> print(f"Hypervolume: {hv:.4f}")
    """
    
    @staticmethod
    def hypervolume(
        pareto_front: List[MultiObjectiveIndividual],
        reference_point: Optional[np.ndarray] = None,
        objective_names: Optional[List[str]] = None
    ) -> float:
        """
        Calculate hypervolume (S-metric) indicator.
        
        Hypervolume measures the volume of objective space dominated by the
        Pareto front. Higher is better.
        
        Args:
            pareto_front: List of Pareto-optimal individuals
            reference_point: Reference point (nadir point). If None, uses worst values.
            objective_names: List of objective names to consider
            
        Returns:
            Hypervolume value
            
        Note:
            For 2D/3D, uses efficient algorithms. For >3D, uses Monte Carlo approximation.
            
        Example:
            >>> hv = QualityIndicators.hypervolume(
            ...     pareto_front,
            ...     reference_point=np.array([0.0, 100.0, 1e7])
            ... )
        """
        if not pareto_front:
            return 0.0
        
        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())
        
        points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in pareto_front
        ])
        
        n_objectives = points.shape[1]
        
        # Set reference point if not provided
        if reference_point is None:
            reference_point = np.min(points, axis=0) - 1.0
        
        # Different algorithms based on dimensionality
        if n_objectives == 2:
            hv = QualityIndicators._hypervolume_2d(points, reference_point)
        elif n_objectives == 3:
            hv = QualityIndicators._hypervolume_3d(points, reference_point)
        else:
            hv = QualityIndicators._hypervolume_monte_carlo(points, reference_point)
        
        return hv
    
    @staticmethod
    def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
        """
        Efficient 2D hypervolume calculation.
        
        Sorts points by first objective and calculates area.
        Complexity: O(n log n)
        """
        # Sort by first objective (descending for maximization)
        sorted_indices = np.argsort(-points[:, 0])
        sorted_points = points[sorted_indices]
        
        hv = 0.0
        prev_y = reference[1]
        
        for point in sorted_points:
            width = point[0] - reference[0]
            height = point[1] - prev_y
            
            if width > 0 and height > 0:
                hv += width * height
            
            prev_y = max(prev_y, point[1])
        
        return abs(hv)
    
    @staticmethod
    def _hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
        """
        3D hypervolume calculation using WFG algorithm.
        
        Complexity: O(n² log n)
        """
        # Simplified 3D calculation
        # For production, use pymoo or pygmo for exact WFG implementation
        
        # Monte Carlo approximation for now
        return QualityIndicators._hypervolume_monte_carlo(points, reference, n_samples=10000)
    
    @staticmethod
    def _hypervolume_monte_carlo(
        points: np.ndarray,
        reference: np.ndarray,
        n_samples: int = 100000
    ) -> float:
        """
        Monte Carlo approximation of hypervolume for high dimensions.
        
        Samples random points in objective space and checks domination.
        
        Args:
            points: Pareto front points
            reference: Reference point
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Approximated hypervolume
        """
        # Define bounds
        upper_bound = np.max(points, axis=0)
        lower_bound = reference
        
        # Volume of bounding box
        box_volume = np.prod(upper_bound - lower_bound)
        
        if box_volume <= 0:
            return 0.0
        
        # Sample random points
        random_points = np.random.uniform(
            lower_bound,
            upper_bound,
            size=(n_samples, len(lower_bound))
        )
        
        # Count dominated points
        dominated_count = 0
        for sample in random_points:
            # Check if any Pareto point dominates this sample
            dominated = np.any(np.all(points >= sample, axis=1))
            if dominated:
                dominated_count += 1
        
        # Estimate hypervolume
        hv = (dominated_count / n_samples) * box_volume
        return hv
    
    @staticmethod
    def inverted_generational_distance(
        pareto_front: List[MultiObjectiveIndividual],
        reference_pareto: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None
    ) -> float:
        """
        Calculate Inverted Generational Distance (IGD).
        
        IGD measures average distance from reference Pareto front to
        obtained front. Lower is better.
        
        Args:
            pareto_front: Obtained Pareto front
            reference_pareto: True/reference Pareto front
            objective_names: Objectives to consider
            
        Returns:
            IGD value
            
        Example:
            >>> igd = QualityIndicators.inverted_generational_distance(
            ...     obtained_front,
            ...     true_pareto_front
            ... )
        """
        if not pareto_front or not reference_pareto:
            return float('inf')
        
        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())
        
        obtained_points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in pareto_front
        ])
        
        reference_points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in reference_pareto
        ])
        
        # For each reference point, find minimum distance to obtained front
        distances = []
        for ref_point in reference_points:
            min_dist = np.min(np.linalg.norm(obtained_points - ref_point, axis=1))
            distances.append(min_dist)
        
        igd = np.mean(distances)
        return igd
    
    @staticmethod
    def spacing(
        pareto_front: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None
    ) -> float:
        """
        Calculate spacing metric.
        
        Spacing measures the uniformity of distribution of solutions
        in the Pareto front. Lower is better (more uniform).
        
        Args:
            pareto_front: Pareto front
            objective_names: Objectives to consider
            
        Returns:
            Spacing value
        """
        if len(pareto_front) < 2:
            return 0.0
        
        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())
        
        points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in pareto_front
        ])
        
        # Calculate pairwise distances
        n = len(points)
        distances = []
        
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(points[i] - points[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        # Spacing = standard deviation of distances
        mean_dist = np.mean(distances)
        spacing_value = np.sqrt(np.mean((np.array(distances) - mean_dist) ** 2))
        
        return spacing_value
    
    @staticmethod
    def spread(
        pareto_front: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None
    ) -> float:
        """
        Calculate spread (delta) metric.
        
        Spread measures the extent of coverage of the Pareto front.
        Lower is better (better coverage).
        
        Args:
            pareto_front: Pareto front
            objective_names: Objectives to consider
            
        Returns:
            Spread value
        """
        if len(pareto_front) < 2:
            return 0.0
        
        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())
        
        points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in pareto_front
        ])
        
        # Sort by first objective
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        # Calculate consecutive distances
        distances = []
        for i in range(len(sorted_points) - 1):
            dist = np.linalg.norm(sorted_points[i + 1] - sorted_points[i])
            distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Extreme distances (to ideal corners)
        d_f = np.linalg.norm(sorted_points[0] - np.max(points, axis=0))
        d_l = np.linalg.norm(sorted_points[-1] - np.max(points, axis=0))
        
        # Mean distance
        d_mean = np.mean(distances)
        
        # Spread metric
        numerator = d_f + d_l + np.sum(np.abs(np.array(distances) - d_mean))
        denominator = d_f + d_l + len(distances) * d_mean
        
        if denominator == 0:
            return 0.0
        
        spread_value = numerator / denominator
        return spread_value
    
    @staticmethod
    def generational_distance(
        pareto_front: List[MultiObjectiveIndividual],
        reference_pareto: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None,
        p: int = 2
    ) -> float:
        """
        Calculate Generational Distance (GD).
        
        GD measures average distance from obtained front to reference front.
        Lower is better (closer to true Pareto front).
        
        Args:
            pareto_front: Obtained Pareto front
            reference_pareto: Reference Pareto front
            objective_names: Objectives to consider
            p: Distance metric exponent (typically 2 for Euclidean)
            
        Returns:
            GD value
        """
        if not pareto_front or not reference_pareto:
            return float('inf')
        
        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())
        
        obtained_points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in pareto_front
        ])
        
        reference_points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in reference_pareto
        ])
        
        # For each obtained point, find minimum distance to reference front
        distances = []
        for point in obtained_points:
            min_dist = np.min(np.linalg.norm(reference_points - point, axis=1))
            distances.append(min_dist ** p)
        
        gd = (np.sum(distances) / len(distances)) ** (1.0 / p)
        return gd
    
    @staticmethod
    def epsilon_indicator(
        pareto_front: List[MultiObjectiveIndividual],
        reference_pareto: List[MultiObjectiveIndividual],
        objective_names: Optional[List[str]] = None
    ) -> float:
        """
        Calculate additive epsilon indicator.
        
        Measures minimum epsilon by which obtained front must be translated
        to dominate reference front.
        
        Args:
            pareto_front: Obtained Pareto front
            reference_pareto: Reference Pareto front
            objective_names: Objectives to consider
            
        Returns:
            Epsilon value
        """
        if not pareto_front or not reference_pareto:
            return float('inf')
        
        # Extract objective values
        if objective_names is None:
            objective_names = list(pareto_front[0].objectives.keys())
        
        obtained_points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in pareto_front
        ])
        
        reference_points = np.array([
            [ind.objectives[name] for name in objective_names]
            for ind in reference_pareto
        ])
        
        # For each reference point, find minimum epsilon
        epsilons = []
        for ref_point in reference_points:
            min_epsilon = float('inf')
            for obt_point in obtained_points:
                epsilon = np.max(ref_point - obt_point)
                min_epsilon = min(min_epsilon, epsilon)
            epsilons.append(min_epsilon)
        
        # Maximum of minimum epsilons
        epsilon_indicator = np.max(epsilons)
        return epsilon_indicator


def calculate_all_indicators(
    pareto_front: List[MultiObjectiveIndividual],
    reference_pareto: Optional[List[MultiObjectiveIndividual]] = None,
    reference_point: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate all quality indicators for a Pareto front.
    
    Args:
        pareto_front: Obtained Pareto front
        reference_pareto: Optional reference Pareto front for IGD/GD/epsilon
        reference_point: Optional reference point for hypervolume
        
    Returns:
        Dictionary of indicator values
        
    Example:
        >>> indicators = calculate_all_indicators(pareto_front)
        >>> print(f"Hypervolume: {indicators['hypervolume']:.4f}")
        >>> print(f"Spacing: {indicators['spacing']:.4f}")
    """
    qi = QualityIndicators()
    
    results = {
        'hypervolume': qi.hypervolume(pareto_front, reference_point),
        'spacing': qi.spacing(pareto_front),
        'spread': qi.spread(pareto_front),
        'pareto_size': len(pareto_front)
    }
    
    if reference_pareto is not None:
        results['igd'] = qi.inverted_generational_distance(pareto_front, reference_pareto)
        results['gd'] = qi.generational_distance(pareto_front, reference_pareto)
        results['epsilon'] = qi.epsilon_indicator(pareto_front, reference_pareto)
    
    return results


def compare_pareto_fronts(
    front1: List[MultiObjectiveIndividual],
    front2: List[MultiObjectiveIndividual],
    front1_name: str = "Front 1",
    front2_name: str = "Front 2"
) -> None:
    """
    Compare two Pareto fronts and print results.
    
    Args:
        front1: First Pareto front
        front2: Second Pareto front
        front1_name: Name for first front
        front2_name: Name for second front
        
    Example:
        >>> compare_pareto_fronts(ga_front, nsga2_front, "GA", "NSGA-II")
    """
    qi = QualityIndicators()
    
    # Calculate indicators
    hv1 = qi.hypervolume(front1)
    hv2 = qi.hypervolume(front2)
    
    spacing1 = qi.spacing(front1)
    spacing2 = qi.spacing(front2)
    
    spread1 = qi.spread(front1)
    spread2 = qi.spread(front2)
    
    # Print comparison
    print("\n" + "="*60)
    print("Pareto Front Comparison")
    print("="*60)
    
    print(f"\n{front1_name:20s} | {front2_name:20s}")
    print("-"*60)
    
    print(f"Size: {len(front1):14d} | {len(front2):14d}")
    print(f"Hypervolume: {hv1:12.4f} | {hv2:12.4f} {'✓' if hv1 > hv2 else ''}")
    print(f"Spacing: {spacing1:16.4f} | {spacing2:16.4f} {'✓' if spacing1 < spacing2 else ''}")
    print(f"Spread: {spread1:17.4f} | {spread2:17.4f} {'✓' if spread1 < spread2 else ''}")
    
    print("="*60 + "\n")
