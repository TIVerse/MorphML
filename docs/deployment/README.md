# MorphML Deployment Guide

**Author:** Eshan Roy <eshanized@proton.me>  
**Organization:** TONMOY INFRASTRUCTURE & VISION  
**Last Updated:** November 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Deployment Options](#deployment-options)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Production Checklist](#production-checklist)

---

## Overview

MorphML supports multiple deployment strategies:

- **Local Development**: Single machine with Docker Compose
- **Kubernetes**: Production-grade distributed deployment
- **Cloud Providers**: GKE, EKS, AKS with Helm charts
- **Hybrid**: Mix of local master and cloud workers

### Architecture

```
┌─────────────────────────────────────────────┐
│           Load Balancer (Optional)          │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────┴────────┐
         │   Master Node   │  (ClusterIP Service)
         │   Port: 50051   │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───┴────┐   ┌───┴────┐   ┌───┴────┐
│Worker 1│   │Worker 2│   │Worker N│
│+ GPU   │   │+ GPU   │   │+ GPU   │
└────────┘   └────────┘   └────────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
┌───┴────────┐  ┌──────────┐  ┌┴─────────┐
│ PostgreSQL │  │  Redis   │  │  MinIO   │
│ (Results)  │  │ (Cache)  │  │(Artifacts)│
└────────────┘  └──────────┘  └──────────┘
```

---

## Prerequisites

### For Local Development
- Docker 20.10+
- Docker Compose 2.0+
- 16GB+ RAM
- 50GB+ free disk space

### For Kubernetes
- Kubernetes 1.24+
- kubectl configured
- Helm 3.8+
- GPU support (NVIDIA GPU Operator for GPU nodes)
- Persistent volume provisioner
- 100GB+ storage per worker node

### Optional
- Prometheus Operator (for monitoring)
- Grafana (for dashboards)
- Ingress Controller (for external access)

---

## Quick Start

### Option 1: Docker Compose (Development)

```bash
# Clone repository
git clone https://github.com/TIVerse/MorphML.git
cd MorphML

# Build images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f master

# Stop services
docker-compose down
```

### Option 2: Kubernetes with Helm (Production)

```bash
# Add dependencies
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install MorphML
helm install morphml ./deployment/helm/morphml \
  --namespace morphml \
  --create-namespace \
  --values ./deployment/helm/morphml/values.yaml

# Check deployment
kubectl get pods -n morphml

# View master logs
kubectl logs -f -n morphml deployment/morphml-master

# Access master service (port-forward for testing)
kubectl port-forward -n morphml svc/morphml-master 50051:50051
```

---

## Deployment Options

### 1. Local Docker Compose

**Best for**: Development, testing, small experiments

**Steps**:

```bash
cd deployment/docker
docker-compose -f docker-compose.yml up -d
```

**Services**:
- Master: `localhost:50051`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- MinIO: `localhost:9000`

### 2. Standalone Kubernetes

**Best for**: Self-hosted production, on-premise

**Steps**:

```bash
# Apply manifests
kubectl create namespace morphml
kubectl apply -f deployment/kubernetes/ -n morphml

# Verify
kubectl get all -n morphml
```

### 3. Helm Chart

**Best for**: Production with easy configuration management

**Steps**:

```bash
# Customize values
cp deployment/helm/morphml/values.yaml my-values.yaml
# Edit my-values.yaml with your settings

# Install
helm install morphml ./deployment/helm/morphml \
  -f my-values.yaml \
  -n morphml \
  --create-namespace

# Upgrade
helm upgrade morphml ./deployment/helm/morphml \
  -f my-values.yaml \
  -n morphml

# Uninstall
helm uninstall morphml -n morphml
```

### 4. Cloud Providers

See specific guides:
- [Google Kubernetes Engine (GKE)](./gke.md)
- [Amazon Elastic Kubernetes Service (EKS)](./eks.md)
- [Azure Kubernetes Service (AKS)](./aks.md)

---

## Configuration

### Master Node Configuration

**values.yaml**:
```yaml
master:
  replicas: 1  # Always 1 for now
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
  persistence:
    enabled: true
    size: 10Gi
```

### Worker Node Configuration

**values.yaml**:
```yaml
worker:
  replicas: 4  # Initial workers
  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
      nvidia.com/gpu: 1  # Request 1 GPU per worker
    limits:
      memory: "16Gi"
      cpu: "8"
      nvidia.com/gpu: 1
  
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 50  # Scale up to 50 workers
    targetCPUUtilizationPercentage: 80
```

### Storage Configuration

**PostgreSQL** (experiment results):
```yaml
postgresql:
  enabled: true
  auth:
    database: morphml
    username: morphml
    password: CHANGE_ME  # Use strong password
  primary:
    persistence:
      size: 20Gi
```

**Redis** (caching):
```yaml
redis:
  enabled: true
  auth:
    password: CHANGE_ME
  master:
    persistence:
      size: 5Gi
```

**MinIO** (artifacts):
```yaml
minio:
  enabled: true
  auth:
    rootUser: minioadmin
    rootPassword: CHANGE_ME
  persistence:
    size: 100Gi
```

### Secrets Management

**Create secrets manually**:
```bash
kubectl create secret generic morphml-secrets \
  --from-literal=postgres-password='YOUR_PASSWORD' \
  --from-literal=redis-password='YOUR_PASSWORD' \
  --from-literal=minio-access-key='YOUR_KEY' \
  --from-literal=minio-secret-key='YOUR_SECRET' \
  -n morphml
```

**Or use external secret management**:
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault
- Sealed Secrets

---

## Monitoring

### Enable Prometheus & Grafana

**values.yaml**:
```yaml
monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

### Access Dashboards

```bash
# Port-forward Grafana
kubectl port-forward -n morphml svc/grafana 3000:80

# Open browser
open http://localhost:3000

# Default credentials
# Username: admin
# Password: <from secret>
```

### Available Metrics

**Master Metrics**:
- `morphml_workers_active` - Number of active workers
- `morphml_tasks_queued` - Tasks in queue
- `morphml_tasks_completed_total` - Total completed tasks
- `morphml_tasks_failed_total` - Total failed tasks
- `morphml_best_fitness` - Current best fitness score

**Worker Metrics**:
- `morphml_worker_cpu_percent` - CPU usage
- `morphml_worker_memory_percent` - Memory usage
- `morphml_worker_gpu_percent` - GPU utilization
- `morphml_worker_tasks_completed` - Tasks completed by worker
- `morphml_task_duration_seconds` - Task execution time

### Grafana Dashboards

Pre-built dashboard at:
```
deployment/monitoring/grafana-dashboard.json
```

Import in Grafana UI: Dashboards → Import → Upload JSON

---

## Troubleshooting

### Workers Not Connecting

**Problem**: Workers can't connect to master

**Check**:
```bash
# Verify master is running
kubectl get pods -n morphml -l component=master

# Check master logs
kubectl logs -n morphml -l component=master

# Test connectivity from worker
kubectl exec -n morphml <worker-pod> -- nc -zv morphml-master 50051
```

**Solutions**:
- Verify service name in worker environment variables
- Check network policies
- Ensure master port is correct (50051)

### Database Connection Errors

**Problem**: Can't connect to PostgreSQL

**Check**:
```bash
# Verify PostgreSQL is running
kubectl get pods -n morphml -l app.kubernetes.io/name=postgresql

# Test connection
kubectl exec -n morphml <master-pod> -- \
  psql -h morphml-postgresql -U morphml -d morphml -c "SELECT 1"
```

**Solutions**:
- Verify credentials in secrets
- Check PostgreSQL service name
- Ensure database is initialized

### GPU Not Available

**Problem**: Workers report no GPU

**Check**:
```bash
# Verify NVIDIA GPU operator
kubectl get pods -n gpu-operator

# Check node labels
kubectl get nodes -o json | jq '.items[].metadata.labels' | grep nvidia

# Exec into worker
kubectl exec -it -n morphml <worker-pod> -- nvidia-smi
```

**Solutions**:
- Install NVIDIA GPU operator
- Add node labels for GPU nodes
- Verify device plugin is running

### Out of Memory

**Problem**: Worker pods killed due to OOM

**Check**:
```bash
# Check events
kubectl describe pod -n morphml <worker-pod>

# Monitor memory
kubectl top pods -n morphml
```

**Solutions**:
- Increase worker memory limits
- Reduce batch size in experiments
- Enable memory limits in training config

### Slow Task Execution

**Problem**: Tasks taking too long

**Check**:
```bash
# Check resource utilization
kubectl top pods -n morphml

# View worker metrics
kubectl port-forward -n morphml svc/morphml-master 8000:8000
curl http://localhost:8000/metrics
```

**Solutions**:
- Increase worker resources (CPU/GPU)
- Add more workers (scale up)
- Optimize evaluation function
- Enable caching

---

## Production Checklist

### Security

- [ ] Change all default passwords
- [ ] Enable RBAC
- [ ] Use network policies
- [ ] Enable pod security policies
- [ ] Use private container registry
- [ ] Enable TLS for gRPC
- [ ] Rotate credentials regularly

### High Availability

- [ ] Enable persistence for master
- [ ] Configure PostgreSQL HA
- [ ] Configure Redis HA (sentinel)
- [ ] Use LoadBalancer for master service
- [ ] Enable pod disruption budgets
- [ ] Configure anti-affinity rules

### Monitoring

- [ ] Enable Prometheus
- [ ] Configure Grafana dashboards
- [ ] Set up alerting rules
- [ ] Monitor disk usage
- [ ] Track GPU utilization
- [ ] Set up log aggregation

### Performance

- [ ] Enable horizontal pod autoscaling
- [ ] Configure resource requests/limits
- [ ] Use GPU nodes for workers
- [ ] Enable caching (Redis)
- [ ] Optimize storage (SSD for PostgreSQL)
- [ ] Configure network policies for low latency

### Backup & Recovery

- [ ] Backup PostgreSQL regularly
- [ ] Backup MinIO buckets
- [ ] Test restore procedures
- [ ] Document recovery steps
- [ ] Configure snapshot schedules

### Cost Optimization

- [ ] Use spot/preemptible instances for workers
- [ ] Enable cluster autoscaler
- [ ] Set resource quotas
- [ ] Monitor cloud costs
- [ ] Scale down during off-hours

---

## Next Steps

1. **Test Deployment**: Run a simple experiment
2. **Configure Monitoring**: Set up dashboards
3. **Scale Workers**: Add more compute resources
4. **Run Benchmarks**: Validate performance
5. **Production Hardening**: Follow checklist above

---

## Support

- **Documentation**: https://github.com/TIVerse/MorphML/docs
- **Issues**: https://github.com/TIVerse/MorphML/issues
- **Discussions**: https://github.com/TIVerse/MorphML/discussions
- **Email**: eshanized@proton.me

---

**Happy Searching! 🧬**
