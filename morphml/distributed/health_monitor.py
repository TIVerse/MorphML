"""System health monitoring for workers.

Tracks CPU, memory, GPU, disk, and network health metrics.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil
    
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    
    PYNVML_AVAILABLE = True
    try:
        pynvml.nvmlInit()
    except Exception:
        PYNVML_AVAILABLE = False
except ImportError:
    PYNVML_AVAILABLE = False

from morphml.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthMetrics:
    """System health metrics."""
    
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_percent: float = 0.0
    disk_free_gb: float = 0.0
    gpu_stats: List[Dict[str, Any]] = field(default_factory=list)
    network_latency_ms: Optional[float] = None
    is_healthy: bool = True
    issues: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Monitor system health metrics for workers.
    
    Tracks:
    - CPU utilization
    - Memory usage
    - GPU utilization and memory
    - Disk space
    - Optional: Network latency
    
    Args:
        thresholds: Health thresholds dictionary
            - cpu_critical: CPU % to mark unhealthy (default: 95)
            - memory_critical: Memory % to mark unhealthy (default: 95)
            - disk_critical: Disk % to mark unhealthy (default: 95)
            - gpu_temp_critical: GPU temperature °C (default: 85)
            - gpu_memory_critical: GPU memory % (default: 95)
    
    Example:
        >>> monitor = HealthMonitor()
        >>> metrics = monitor.get_health_metrics()
        >>> if not metrics.is_healthy:
        ...     print(f"Health issues: {metrics.issues}")
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize health monitor."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, health monitoring limited")
        
        thresholds = thresholds or {}
        self.cpu_critical = thresholds.get("cpu_critical", 95.0)
        self.memory_critical = thresholds.get("memory_critical", 95.0)
        self.disk_critical = thresholds.get("disk_critical", 95.0)
        self.gpu_temp_critical = thresholds.get("gpu_temp_critical", 85.0)
        self.gpu_memory_critical = thresholds.get("gpu_memory_critical", 95.0)
        
        # Initialize GPU monitoring
        self.gpu_count = 0
        if PYNVML_AVAILABLE:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"Detected {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to detect GPUs: {e}")
        
        logger.info("Initialized HealthMonitor")
    
    def get_health_metrics(self) -> HealthMetrics:
        """
        Get current system health metrics.
        
        Returns:
            HealthMetrics object
        """
        metrics = HealthMetrics()
        
        if not PSUTIL_AVAILABLE:
            metrics.is_healthy = True  # Assume healthy if can't check
            return metrics
        
        # CPU
        try:
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
        
        # Memory
        try:
            mem = psutil.virtual_memory()
            metrics.memory_percent = mem.percent
            metrics.memory_available_gb = mem.available / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
        
        # Disk
        try:
            disk = psutil.disk_usage("/")
            metrics.disk_percent = disk.percent
            metrics.disk_free_gb = disk.free / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get disk metrics: {e}")
        
        # GPU
        if PYNVML_AVAILABLE and self.gpu_count > 0:
            metrics.gpu_stats = self._get_gpu_stats()
        
        # Check health
        metrics.is_healthy, metrics.issues = self._check_health(metrics)
        
        return metrics
    
    def _get_gpu_stats(self) -> List[Dict[str, Any]]:
        """Get GPU statistics using pynvml."""
        gpu_stats = []
        
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                gpu_stats.append(
                    {
                        "id": i,
                        "name": name,
                        "load": util.gpu,
                        "memory_used_mb": mem_info.used / (1024 ** 2),
                        "memory_total_mb": mem_info.total / (1024 ** 2),
                        "memory_percent": (mem_info.used / mem_info.total) * 100,
                        "temperature": temp,
                    }
                )
            
            except Exception as e:
                logger.warning(f"Failed to get stats for GPU {i}: {e}")
        
        return gpu_stats
    
    def _check_health(
        self, metrics: HealthMetrics
    ) -> tuple[bool, List[str]]:
        """
        Check if system is healthy based on metrics.
        
        Returns:
            (is_healthy, list of issues)
        """
        issues = []
        
        # Check CPU
        if metrics.cpu_percent > self.cpu_critical:
            issues.append(f"CPU overload: {metrics.cpu_percent:.1f}%")
        
        # Check memory
        if metrics.memory_percent > self.memory_critical:
            issues.append(f"Memory critical: {metrics.memory_percent:.1f}%")
        
        # Check disk
        if metrics.disk_percent > self.disk_critical:
            issues.append(
                f"Disk critical: {metrics.disk_percent:.1f}% "
                f"({metrics.disk_free_gb:.1f}GB free)"
            )
        
        # Check GPUs
        for gpu in metrics.gpu_stats:
            gpu_id = gpu["id"]
            
            # Temperature
            if gpu["temperature"] > self.gpu_temp_critical:
                issues.append(
                    f"GPU {gpu_id} overheating: {gpu['temperature']}°C"
                )
            
            # Memory
            if gpu["memory_percent"] > self.gpu_memory_critical:
                issues.append(
                    f"GPU {gpu_id} memory critical: {gpu['memory_percent']:.1f}%"
                )
        
        is_healthy = len(issues) == 0
        
        return is_healthy, issues
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get static system information.
        
        Returns:
            System info dictionary
        """
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info["cpu_count_logical"] = psutil.cpu_count(logical=True)
                info["cpu_count_physical"] = psutil.cpu_count(logical=False)
                info["memory_total_gb"] = psutil.virtual_memory().total / (1024 ** 3)
                info["disk_total_gb"] = psutil.disk_usage("/").total / (1024 ** 3)
            except Exception as e:
                logger.warning(f"Failed to get system info: {e}")
        
        if PYNVML_AVAILABLE and self.gpu_count > 0:
            gpu_info = []
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info.append(
                        {
                            "id": i,
                            "name": name,
                            "memory_total_gb": mem_info.total / (1024 ** 3),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to get info for GPU {i}: {e}")
            
            info["gpus"] = gpu_info
        else:
            info["gpus"] = []
        
        return info
    
    def monitor_continuously(
        self, interval: float = 60.0, callback: Optional[callable] = None
    ) -> None:
        """
        Continuously monitor health (blocking).
        
        Args:
            interval: Monitoring interval in seconds
            callback: Optional callback function(metrics)
        """
        logger.info(f"Starting continuous monitoring (interval={interval}s)")
        
        try:
            while True:
                metrics = self.get_health_metrics()
                
                if not metrics.is_healthy:
                    logger.warning(f"Health issues detected: {metrics.issues}")
                
                if callback:
                    callback(metrics)
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Stopping continuous monitoring")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def get_system_health() -> Dict[str, Any]:
    """
    Convenience function to get system health.
    
    Returns:
        Health metrics dictionary
    """
    monitor = HealthMonitor()
    metrics = monitor.get_health_metrics()
    
    return {
        "timestamp": metrics.timestamp,
        "cpu_percent": metrics.cpu_percent,
        "memory_percent": metrics.memory_percent,
        "memory_available_gb": metrics.memory_available_gb,
        "disk_percent": metrics.disk_percent,
        "disk_free_gb": metrics.disk_free_gb,
        "gpus": metrics.gpu_stats,
        "is_healthy": metrics.is_healthy,
        "issues": metrics.issues,
    }


def is_system_healthy(thresholds: Optional[Dict[str, float]] = None) -> bool:
    """
    Quick health check.
    
    Args:
        thresholds: Optional custom thresholds
    
    Returns:
        True if system is healthy
    """
    monitor = HealthMonitor(thresholds)
    metrics = monitor.get_health_metrics()
    return metrics.is_healthy
