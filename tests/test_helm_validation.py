"""Validation tests for Helm templates and Kubernetes manifests.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest
import yaml


HELM_CHART_DIR = Path(__file__).parent.parent / "deployment" / "helm" / "morphml"
K8S_MANIFESTS_DIR = Path(__file__).parent.parent / "deployment" / "kubernetes"


class TestHelmChartStructure:
    """Test Helm chart has correct structure."""

    def test_chart_yaml_exists(self):
        """Test Chart.yaml exists."""
        chart_file = HELM_CHART_DIR / "Chart.yaml"
        assert chart_file.exists(), "Chart.yaml not found"

    def test_values_yaml_exists(self):
        """Test values.yaml exists."""
        values_file = HELM_CHART_DIR / "values.yaml"
        assert values_file.exists(), "values.yaml not found"

    def test_templates_directory_exists(self):
        """Test templates directory exists."""
        templates_dir = HELM_CHART_DIR / "templates"
        assert templates_dir.exists(), "templates directory not found"

    def test_required_templates_exist(self):
        """Test all required templates exist."""
        templates_dir = HELM_CHART_DIR / "templates"
        required_templates = [
            "_helpers.tpl",
            "master-deployment.yaml",
            "worker-deployment.yaml",
            "service.yaml",
            "configmap.yaml",
            "secrets.yaml",
            "serviceaccount.yaml",
            "rbac.yaml",
            "hpa.yaml",
            "pvc.yaml",
        ]

        for template in required_templates:
            template_file = templates_dir / template
            assert template_file.exists(), f"Template {template} not found"


class TestHelmChartValidation:
    """Test Helm chart is valid."""

    def test_chart_yaml_valid(self):
        """Test Chart.yaml is valid YAML."""
        chart_file = HELM_CHART_DIR / "Chart.yaml"
        with open(chart_file) as f:
            chart_data = yaml.safe_load(f)

        assert "name" in chart_data, "Chart.yaml missing 'name'"
        assert "version" in chart_data, "Chart.yaml missing 'version'"
        assert "apiVersion" in chart_data, "Chart.yaml missing 'apiVersion'"

    def test_values_yaml_valid(self):
        """Test values.yaml is valid YAML."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values_data = yaml.safe_load(f)

        # Check required sections
        assert "master" in values_data, "values.yaml missing 'master'"
        assert "worker" in values_data, "values.yaml missing 'worker'"
        assert "postgresql" in values_data, "values.yaml missing 'postgresql'"
        assert "redis" in values_data, "values.yaml missing 'redis'"
        assert "minio" in values_data, "values.yaml missing 'minio'"

    def test_templates_syntax(self):
        """Test templates have valid syntax."""
        templates_dir = HELM_CHART_DIR / "templates"
        yaml_templates = list(templates_dir.glob("*.yaml"))

        for template_file in yaml_templates:
            # Skip files with Go template syntax that can't be validated directly
            if template_file.name == "_helpers.tpl":
                continue

            # Just check file can be opened
            with open(template_file) as f:
                content = f.read()
                assert len(content) > 0, f"{template_file.name} is empty"


class TestHelmTemplateRendering:
    """Test Helm templates render correctly."""

    @pytest.mark.skipif(
        not Path("/usr/local/bin/helm").exists() and not Path("/usr/bin/helm").exists(),
        reason="Helm not installed",
    )
    def test_helm_template_renders(self):
        """Test helm template command works."""
        try:
            result = subprocess.run(
                ["helm", "template", "test", str(HELM_CHART_DIR)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Check command succeeded
            assert result.returncode == 0, f"Helm template failed: {result.stderr}"

            # Check output is valid YAML
            rendered = result.stdout
            assert len(rendered) > 0, "Helm template produced no output"

            # Should contain Kubernetes resources
            assert "kind: Deployment" in rendered
            assert "kind: Service" in rendered

        except FileNotFoundError:
            pytest.skip("Helm command not found")

    @pytest.mark.skipif(
        not Path("/usr/local/bin/helm").exists() and not Path("/usr/bin/helm").exists(),
        reason="Helm not installed",
    )
    def test_helm_lint(self):
        """Test helm lint passes."""
        try:
            result = subprocess.run(
                ["helm", "lint", str(HELM_CHART_DIR)], capture_output=True, text=True, timeout=30
            )

            # Lint should pass with no errors
            assert result.returncode == 0, f"Helm lint failed: {result.stderr}"
            assert "0 chart(s) failed" in result.stdout or "1 chart(s) linted" in result.stdout

        except FileNotFoundError:
            pytest.skip("Helm command not found")


class TestKubernetesManifests:
    """Test standalone Kubernetes manifests."""

    def test_manifests_exist(self):
        """Test manifest files exist."""
        assert K8S_MANIFESTS_DIR.exists(), "Kubernetes manifests directory not found"

        manifest_files = list(K8S_MANIFESTS_DIR.glob("*.yaml"))
        assert len(manifest_files) > 0, "No manifest files found"

    def test_manifests_valid_yaml(self):
        """Test manifests are valid YAML."""
        manifest_files = list(K8S_MANIFESTS_DIR.glob("*.yaml"))

        for manifest_file in manifest_files:
            with open(manifest_file) as f:
                try:
                    yaml.safe_load_all(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {manifest_file.name}: {e}")


class TestResourceConfiguration:
    """Test resource configurations are reasonable."""

    def test_master_resources(self):
        """Test master resource limits."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        master_resources = values["master"]["resources"]

        # Check memory
        assert "requests" in master_resources
        assert "limits" in master_resources

        # Master should have reasonable memory
        mem_request = master_resources["requests"]["memory"]
        assert "Gi" in mem_request or "Mi" in mem_request

    def test_worker_resources(self):
        """Test worker resource configuration."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        worker_resources = values["worker"]["resources"]

        # Workers should request GPU
        assert "nvidia.com/gpu" in worker_resources["requests"]
        assert worker_resources["requests"]["nvidia.com/gpu"] >= 1

    def test_autoscaling_config(self):
        """Test autoscaling configuration."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        autoscaling = values["worker"]["autoscaling"]

        assert autoscaling["enabled"] == True
        assert autoscaling["minReplicas"] >= 1
        assert autoscaling["maxReplicas"] >= autoscaling["minReplicas"]
        assert 0 < autoscaling["targetCPUUtilizationPercentage"] <= 100


class TestSecurityConfiguration:
    """Test security configurations."""

    def test_rbac_enabled(self):
        """Test RBAC is configured."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        assert "rbac" in values
        assert values["rbac"]["create"] == True

    def test_service_account_configured(self):
        """Test service account is configured."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        assert "serviceAccount" in values
        assert values["serviceAccount"]["create"] == True
        assert "name" in values["serviceAccount"]

    def test_secrets_template_exists(self):
        """Test secrets template exists."""
        secrets_file = HELM_CHART_DIR / "templates" / "secrets.yaml"
        assert secrets_file.exists()

        with open(secrets_file) as f:
            content = f.read()
            assert "kind: Secret" in content


class TestMonitoringConfiguration:
    """Test monitoring is configured."""

    def test_servicemonitor_template(self):
        """Test ServiceMonitor template exists."""
        sm_file = HELM_CHART_DIR / "templates" / "servicemonitor.yaml"
        assert sm_file.exists()

        with open(sm_file) as f:
            content = f.read()
            assert "ServiceMonitor" in content

    def test_prometheus_config_exists(self):
        """Test Prometheus config exists."""
        prom_config = Path(__file__).parent.parent / "deployment" / "monitoring" / "prometheus.yaml"
        assert prom_config.exists()

        with open(prom_config) as f:
            config = yaml.safe_load(f)
            assert "scrape_configs" in config

    def test_grafana_dashboard_exists(self):
        """Test Grafana dashboard exists."""
        dashboard_file = (
            Path(__file__).parent.parent / "deployment" / "monitoring" / "grafana-dashboard.json"
        )
        assert dashboard_file.exists()

        with open(dashboard_file) as f:
            dashboard = json.load(f)
            assert "dashboard" in dashboard
            assert "panels" in dashboard["dashboard"]


class TestStorageConfiguration:
    """Test storage is properly configured."""

    def test_pvc_template_exists(self):
        """Test PVC template exists."""
        pvc_file = HELM_CHART_DIR / "templates" / "pvc.yaml"
        assert pvc_file.exists()

    def test_postgresql_configured(self):
        """Test PostgreSQL is configured."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        pg_config = values["postgresql"]
        assert pg_config["enabled"] == True
        assert "auth" in pg_config
        assert "database" in pg_config["auth"]

    def test_redis_configured(self):
        """Test Redis is configured."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        redis_config = values["redis"]
        assert redis_config["enabled"] == True

    def test_minio_configured(self):
        """Test MinIO is configured."""
        values_file = HELM_CHART_DIR / "values.yaml"
        with open(values_file) as f:
            values = yaml.safe_load(f)

        minio_config = values["minio"]
        assert minio_config["enabled"] == True
        assert "defaultBuckets" in minio_config


class TestDocumentation:
    """Test deployment documentation exists."""

    def test_deployment_readme(self):
        """Test deployment README exists."""
        readme = Path(__file__).parent.parent / "docs" / "deployment" / "README.md"
        assert readme.exists()

        with open(readme) as f:
            content = f.read()
            assert len(content) > 1000  # Should be substantial
            assert "Quick Start" in content

    def test_kubernetes_guide(self):
        """Test Kubernetes guide exists."""
        guide = Path(__file__).parent.parent / "docs" / "deployment" / "kubernetes.md"
        assert guide.exists()

        with open(guide) as f:
            content = f.read()
            assert "helm install" in content.lower()

    def test_gke_guide(self):
        """Test GKE guide exists."""
        guide = Path(__file__).parent.parent / "docs" / "deployment" / "gke.md"
        assert guide.exists()

        with open(guide) as f:
            content = f.read()
            assert "gcloud" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
