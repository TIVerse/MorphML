# Kubernetes Deployment Guide

Complete guide for deploying MorphML on Kubernetes clusters.

---

## Architecture on Kubernetes

```
Namespace: morphml
├── Master Deployment (1 replica)
│   ├── Pod: morphml-master-xxx
│   └── Service: morphml-master (ClusterIP:50051)
├── Worker Deployment (4+ replicas, autoscaling)
│   ├── Pod: morphml-worker-xxx-1
│   ├── Pod: morphml-worker-xxx-2
│   └── ...
├── PostgreSQL StatefulSet (via Bitnami chart)
│   └── PVC: data-morphml-postgresql-0
├── Redis StatefulSet (via Bitnami chart)
│   └── PVC: redis-data-morphml-redis-master-0
└── MinIO Deployment (via Bitnami chart)
    └── PVC: morphml-minio
```

---

## Prerequisites

### 1. Kubernetes Cluster

**Minimum requirements**:
- Kubernetes 1.24+
- 3+ nodes (1 master node, 2+ worker nodes)
- 100GB+ storage per worker node
- GPU support (optional but recommended)

**Recommended**:
- Kubernetes 1.27+
- 5+ nodes
- NVIDIA GPU operator installed
- High-performance storage (SSD/NVMe)

### 2. kubectl & Helm

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
kubectl version --client
helm version
```

### 3. Storage Provisioner

Ensure your cluster has a storage class:

```bash
kubectl get storageclass

# Example output:
# NAME                 PROVISIONER             RECLAIMPOLICY
# standard (default)   kubernetes.io/gce-pd    Delete
# fast-ssd             kubernetes.io/gce-pd    Delete
```

If none exists, install one:
```bash
# For local development (hostPath)
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml

# Set as default
kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

### 4. GPU Support (Optional)

Install NVIDIA GPU Operator:

```bash
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true
```

Verify:
```bash
kubectl get pods -n gpu-operator
kubectl describe nodes | grep nvidia.com/gpu
```

---

## Installation

### Method 1: Helm Chart (Recommended)

**Step 1: Add Helm repositories**

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

**Step 2: Create namespace**

```bash
kubectl create namespace morphml
```

**Step 3: Customize values**

```bash
cd MorphML
cp deployment/helm/morphml/values.yaml my-values.yaml

# Edit my-values.yaml
# - Change default passwords
# - Adjust resource requests/limits
# - Configure storage classes
# - Set worker replicas
```

**Step 4: Install**

```bash
helm install morphml ./deployment/helm/morphml \
  --namespace morphml \
  --values my-values.yaml \
  --timeout 10m
```

**Step 5: Verify**

```bash
# Check all pods are running
kubectl get pods -n morphml

# Expected output:
# NAME                               READY   STATUS    RESTARTS   AGE
# morphml-master-xxx                 1/1     Running   0          2m
# morphml-worker-xxx-1               1/1     Running   0          2m
# morphml-worker-xxx-2               1/1     Running   0          2m
# morphml-postgresql-0               1/1     Running   0          2m
# morphml-redis-master-0             1/1     Running   0          2m
# morphml-minio-xxx                  1/1     Running   0          2m
```

### Method 2: Standalone Manifests

**Step 1: Create namespace**

```bash
kubectl create namespace morphml
```

**Step 2: Create secrets**

```bash
kubectl create secret generic morphml-secrets \
  --from-literal=postgres-username=morphml \
  --from-literal=postgres-password='YOUR_STRONG_PASSWORD' \
  --from-literal=redis-password='YOUR_REDIS_PASSWORD' \
  --from-literal=minio-access-key='YOUR_ACCESS_KEY' \
  --from-literal=minio-secret-key='YOUR_SECRET_KEY' \
  --namespace morphml
```

**Step 3: Apply manifests**

```bash
kubectl apply -f deployment/kubernetes/ --namespace morphml
```

**Step 4: Verify**

```bash
kubectl get all -n morphml
```

---

## Configuration

### Resource Allocation

**Master Node**:
```yaml
master:
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
```

**Worker Nodes (with GPU)**:
```yaml
worker:
  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
      nvidia.com/gpu: 1
    limits:
      memory: "16Gi"
      cpu: "8"
      nvidia.com/gpu: 1
```

### Autoscaling

**Horizontal Pod Autoscaler**:
```yaml
worker:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 50
    targetCPUUtilizationPercentage: 80
    targetMemoryUtilizationPercentage: 85
```

**Cluster Autoscaler** (for node-level scaling):
```bash
# GKE
gcloud container clusters update CLUSTER_NAME \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --zone ZONE

# AWS EKS - configure in cluster.yaml
```

### Node Affinity

**Place workers on GPU nodes**:
```yaml
worker:
  nodeSelector:
    accelerator: nvidia-gpu
  
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

**Separate master and workers**:
```yaml
master:
  nodeSelector:
    node-role: master
  
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            component: worker
        topologyKey: kubernetes.io/hostname
```

---

## Operations

### Scaling

**Manual scaling**:
```bash
# Scale workers
kubectl scale deployment morphml-worker --replicas=10 -n morphml

# Verify
kubectl get pods -n morphml -l component=worker
```

**Check HPA status**:
```bash
kubectl get hpa -n morphml
```

### Updates

**Helm upgrade**:
```bash
# Update image version
helm upgrade morphml ./deployment/helm/morphml \
  --namespace morphml \
  --values my-values.yaml \
  --set worker.image.tag=v0.2.0
```

**Rolling update strategy**:
```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Monitoring

**View logs**:
```bash
# Master logs
kubectl logs -f -n morphml deployment/morphml-master

# Worker logs
kubectl logs -f -n morphml deployment/morphml-worker

# All worker logs
kubectl logs -f -n morphml -l component=worker --all-containers=true
```

**Port forward to master metrics**:
```bash
kubectl port-forward -n morphml svc/morphml-master 8000:8000
curl http://localhost:8000/metrics
```

**Access Grafana**:
```bash
kubectl port-forward -n morphml svc/grafana 3000:80
```

### Debugging

**Exec into pod**:
```bash
# Master
kubectl exec -it -n morphml deployment/morphml-master -- bash

# Worker
kubectl exec -it -n morphml deployment/morphml-worker -- bash
```

**Check events**:
```bash
kubectl get events -n morphml --sort-by='.lastTimestamp'
```

**Describe resources**:
```bash
kubectl describe pod -n morphml morphml-master-xxx
kubectl describe deployment -n morphml morphml-worker
```

---

## Storage

### Persistent Volumes

**Master data volume**:
```yaml
master:
  persistence:
    enabled: true
    storageClass: "fast-ssd"
    size: 10Gi
```

**PostgreSQL**:
```yaml
postgresql:
  primary:
    persistence:
      storageClass: "fast-ssd"
      size: 50Gi
```

**Backup volumes**:
```bash
# Create backup
kubectl exec -n morphml morphml-postgresql-0 -- \
  pg_dump -U morphml morphml > backup.sql

# Restore
kubectl exec -i -n morphml morphml-postgresql-0 -- \
  psql -U morphml morphml < backup.sql
```

---

## Security

### RBAC

**Service account permissions**:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: morphml
  namespace: morphml
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
```

### Network Policies

**Restrict traffic**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: morphml-master
  namespace: morphml
spec:
  podSelector:
    matchLabels:
      component: master
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          component: worker
    ports:
    - protocol: TCP
      port: 50051
```

### Pod Security

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
    - ALL
```

---

## Troubleshooting

### Common Issues

**1. Pods stuck in Pending**

Check:
```bash
kubectl describe pod -n morphml <pod-name>
```

Common causes:
- Insufficient resources
- No nodes with GPUs
- PVC provisioning failed

**2. CrashLoopBackOff**

Check logs:
```bash
kubectl logs -n morphml <pod-name> --previous
```

Common causes:
- Wrong environment variables
- Can't connect to database
- Missing secrets

**3. Workers can't connect to master**

Test connectivity:
```bash
kubectl exec -n morphml <worker-pod> -- \
  nc -zv morphml-master 50051
```

Check:
- Service name correct
- Network policies
- Master pod is running

---

## Best Practices

1. **Use resource limits** - Prevent resource exhaustion
2. **Enable monitoring** - Track metrics and logs
3. **Configure autoscaling** - Scale based on demand
4. **Use persistent storage** - Don't lose data
5. **Regular backups** - Backup PostgreSQL and MinIO
6. **Security hardening** - Follow security checklist
7. **Test disaster recovery** - Practice restoring from backups

---

## Next Steps

- [Deploy on GKE](./gke.md)
- [Deploy on EKS](./eks.md)
- [Deploy on AKS](./aks.md)
- [Set up monitoring](./monitoring.md)
