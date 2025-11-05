"""Tests for health monitoring."""


from morphml.distributed import HealthMonitor, get_system_health, is_system_healthy


class TestHealthMonitor:
    """Test health monitor."""

    def test_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor()

        assert monitor.cpu_critical == 95.0
        assert monitor.memory_critical == 95.0

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        monitor = HealthMonitor({"cpu_critical": 80.0, "memory_critical": 85.0})

        assert monitor.cpu_critical == 80.0
        assert monitor.memory_critical == 85.0

    def test_get_health_metrics(self):
        """Test getting health metrics."""
        monitor = HealthMonitor()
        metrics = monitor.get_health_metrics()

        assert hasattr(metrics, "cpu_percent")
        assert hasattr(metrics, "memory_percent")
        assert hasattr(metrics, "disk_percent")
        assert hasattr(metrics, "is_healthy")
        assert hasattr(metrics, "issues")

    def test_get_system_info(self):
        """Test getting system info."""
        monitor = HealthMonitor()
        info = monitor.get_system_info()

        assert "platform" in info
        assert "architecture" in info
        assert "python_version" in info

    def test_health_check_logic(self):
        """Test health check logic."""
        from morphml.distributed.health_monitor import HealthMetrics

        monitor = HealthMonitor({"cpu_critical": 80.0})

        # Create metrics with high CPU
        metrics = HealthMetrics()
        metrics.cpu_percent = 90.0
        metrics.memory_percent = 50.0
        metrics.disk_percent = 50.0

        is_healthy, issues = monitor._check_health(metrics)

        assert not is_healthy
        assert len(issues) > 0
        assert any("CPU" in issue for issue in issues)


def test_get_system_health():
    """Test convenience function."""
    health = get_system_health()

    assert "cpu_percent" in health
    assert "memory_percent" in health
    assert "is_healthy" in health


def test_is_system_healthy():
    """Test quick health check."""
    # Should not raise exception
    healthy = is_system_healthy()

    assert isinstance(healthy, bool)


def test_health_monitor_imports():
    """Test that health monitor classes can be imported."""
    from morphml.distributed import (
        HealthMetrics,
        HealthMonitor,
        get_system_health,
        is_system_healthy,
    )

    assert HealthMonitor is not None
    assert HealthMetrics is not None
    assert get_system_health is not None
    assert is_system_healthy is not None
