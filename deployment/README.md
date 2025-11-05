# MorphML Kubernetes Deployment

Production-ready Kubernetes deployment for MorphML distributed Neural Architecture Search.

## 📋 Prerequisites

- Kubernetes cluster (1.20+)
- `kubectl` CLI tool
- `helm` (3.0+)
- Docker (for building images)
- NVIDIA GPU nodes (for workers)
- NVIDIA device plugin installed on cluster

## 🚀 Quick Start

### 1. Build Docker Images

```bash
# Build and tag images
cd /path/to/MorphML
./deployment/scripts/build.sh

# Push to registry (optional)
PUSH=true REGISTRY=your-registry ./deployment/scripts/build.sh
```

### 2. Deploy with Helm

```bash
# Deploy everything (PostgreSQL, Redis, MinIO, MorphML)
./deployment/scripts/deploy.sh

# Or with custom values
helm install morphml ./deployment/helm/morphml \
  --namespace morphml \
  --create-namespace \
  --values your-values.yaml
```

### 3. Verify Deployment

```bash
# Check pods
kubectl get pods -n morphml

# Check services
kubectl get svc -n morphml

# View logs
kubectl logs -f deployment/morphml-master -n morphml
kubectl logs -f deployment/morphml-worker -n morphml
```

## 📦 Components

### Docker Images

- **morphml-master**: Master node (coordinator)
- **morphml-worker**: Worker node (evaluator with GPU)

### Kubernetes Resources

- **Deployments**: Master (1 replica), Workers (auto-scaling 2-50)
- **Services**: ClusterIP for internal communication
- **ConfigMaps**: Configuration parameters
- **Secrets**: Credentials for databases
- **PVCs**: Persistent storage for data
- **HPA**: Horizontal Pod Autoscaler for workers

### Dependencies

- **PostgreSQL**: Experiment results database
- **Redis**: Distributed cache
- **MinIO**: S3-compatible object storage
- **Prometheus** (optional): Metrics collection
- **Grafana** (optional): Dashboards

## ⚙️ Configuration

### values.yaml

Key configuration options:

```yaml
# Worker scaling
worker:
  replicas: 4
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 50
    targetCPUUtilizationPercentage: 80

# Resource limits
worker:
  resources:
    requests:
      nvidia.com/gpu: 1
      memory: "8Gi"
      cpu: "4"

# Storage
postgresql:
  primary:
    persistence:
      size: 20Gi

minio:
  persistence:
    size: 100Gi
```

### Environment Variables

Set via ConfigMap or Secrets:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
- `REDIS_HOST`, `REDIS_PORT`
- `S3_ENDPOINT`, `S3_BUCKET`
- `MASTER_HOST`, `MASTER_PORT`

## 📊 Monitoring

### Prometheus

Deploy Prometheus for metrics:

```bash
kubectl apply -f deployment/monitoring/prometheus-config.yaml
```

Access Prometheus UI:

```bash
kubectl port-forward svc/prometheus 9090:9090 -n morphml
```

### Metrics Exported

- Task completion rate
- Worker utilization
- Architecture evaluations/sec
- Failure rates
- Resource usage (CPU, memory, GPU)

## 🔧 Operations

### Scale Workers

```bash
# Manual scaling
kubectl scale deployment morphml-worker --replicas=10 -n morphml

# Check HPA status
kubectl get hpa -n morphml
```

### Update Images

```bash
# Build new images
./deployment/scripts/build.sh

# Rolling update
kubectl set image deployment/morphml-master \
  master=tiverse/morphml-master:v2 -n morphml

kubectl set image deployment/morphml-worker \
  worker=tiverse/morphml-worker:v2 -n morphml
```

### View Logs

```bash
# Master logs
kubectl logs -f deployment/morphml-master -n morphml

# Worker logs (all pods)
kubectl logs -f -l component=worker -n morphml

# Specific pod
kubectl logs morphml-worker-abc123 -n morphml
```

### Debug

```bash
# Execute commands in pod
kubectl exec -it morphml-master-abc123 -n morphml -- /bin/bash

# Check resource usage
kubectl top pods -n morphml
kubectl top nodes

# Describe pod
kubectl describe pod morphml-worker-abc123 -n morphml
```

### Backup

```bash
# Backup PostgreSQL
kubectl exec -it postgres-postgresql-0 -n morphml -- \
  pg_dump -U morphml morphml > backup.sql

# Backup MinIO
kubectl port-forward svc/minio 9000:9000 -n morphml
mc alias set morphml http://localhost:9000 minioadmin changeme
mc mirror morphml/morphml-artifacts ./backup
```

## 🌐 Cloud Provider Specific

### Google Kubernetes Engine (GKE)

```bash
# Create cluster with GPUs
gcloud container clusters create morphml \
  --accelerator type=nvidia-tesla-v100,count=4 \
  --machine-type n1-standard-8 \
  --num-nodes 3 \
  --zone us-central1-a

# Install NVIDIA driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Amazon EKS

```bash
# Create cluster
eksctl create cluster \
  --name morphml \
  --version 1.27 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10

# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### Azure AKS

```bash
# Create cluster with GPU nodes
az aks create \
  --resource-group morphml \
  --name morphml \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-addons monitoring

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

## 🔐 Security

### Best Practices

1. **Change default passwords** in `secrets.yaml`
2. **Use RBAC** for fine-grained access control
3. **Enable network policies** to restrict traffic
4. **Use private registry** for Docker images
5. **Enable pod security policies**
6. **Regular security updates** of base images

### Example: Update Secrets

```bash
kubectl create secret generic morphml-secrets \
  --from-literal=postgres-password='YOUR_SECURE_PASSWORD' \
  --from-literal=redis-password='YOUR_SECURE_PASSWORD' \
  --from-literal=s3-secret-key='YOUR_SECURE_KEY' \
  --namespace morphml \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 🐛 Troubleshooting

### Workers not connecting to master

```bash
# Check master service
kubectl get svc morphml-master -n morphml

# Check master logs
kubectl logs deployment/morphml-master -n morphml

# Verify DNS resolution from worker
kubectl exec morphml-worker-abc123 -n morphml -- \
  nslookup morphml-master.morphml.svc.cluster.local
```

### GPU not detected

```bash
# Check NVIDIA device plugin
kubectl get daemonset -n kube-system | grep nvidia

# Verify GPU nodes
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# Check worker logs
kubectl logs morphml-worker-abc123 -n morphml | grep -i gpu
```

### Out of memory

```bash
# Increase worker memory limits
kubectl set resources deployment morphml-worker \
  --limits=memory=32Gi -n morphml

# Or update values.yaml and helm upgrade
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [MorphML Documentation](https://github.com/TIVerse/MorphML)

## 📄 License

Copyright © 2025 TONMOY INFRASTRUCTURE & VISION  
Licensed under the Apache License 2.0
