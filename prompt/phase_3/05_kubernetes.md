# Component 5: Kubernetes Deployment

**Duration:** Weeks 7-8  
**LOC Target:** ~4,000  
**Dependencies:** All Phase 3 components

---

## 🎯 Objective

Deploy MorphML on Kubernetes for production:
1. **Docker Images** - Containerize master and worker
2. **K8s Manifests** - Deployments, Services, ConfigMaps
3. **Helm Charts** - Parameterized deployments
4. **Auto-scaling** - HPA for workers
5. **Monitoring** - Prometheus + Grafana

---

## 📋 Files to Create

### 1. `deployment/docker/Dockerfile.master` (~50 LOC)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy source
COPY morphml/ ./morphml/

# Expose port
EXPOSE 50051

# Run master
CMD ["python", "-m", "morphml.distributed.master"]
```

---

### 2. `deployment/docker/Dockerfile.worker` (~60 LOC)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy source
COPY morphml/ ./morphml/

# Expose port
EXPOSE 50052

# Run worker
CMD ["python3", "-m", "morphml.distributed.worker"]
```

---

### 3. `deployment/kubernetes/master-deployment.yaml` (~100 LOC)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morphml-master
  labels:
    app: morphml
    component: master
spec:
  replicas: 1
  selector:
    matchLabels:
      app: morphml
      component: master
  template:
    metadata:
      labels:
        app: morphml
        component: master
    spec:
      containers:
      - name: master
        image: tiverse/morphml-master:latest
        ports:
        - containerPort: 50051
          name: grpc
        env:
        - name: MASTER_PORT
          value: "50051"
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: morphml-config
              key: postgres_host
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: morphml-config
              key: redis_host
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: morphml-master
spec:
  selector:
    app: morphml
    component: master
  ports:
  - port: 50051
    targetPort: 50051
    name: grpc
  type: ClusterIP
```

---

### 4. `deployment/kubernetes/worker-deployment.yaml` (~120 LOC)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morphml-worker
  labels:
    app: morphml
    component: worker
spec:
  replicas: 4
  selector:
    matchLabels:
      app: morphml
      component: worker
  template:
    metadata:
      labels:
        app: morphml
        component: worker
    spec:
      containers:
      - name: worker
        image: tiverse/morphml-worker:latest
        ports:
        - containerPort: 50052
          name: grpc
        env:
        - name: MASTER_HOST
          value: "morphml-master"
        - name: MASTER_PORT
          value: "50051"
        - name: WORKER_PORT
          value: "50052"
        - name: NUM_GPUS
          value: "1"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
      nodeSelector:
        accelerator: nvidia-tesla-v100
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: morphml-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: morphml-worker
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
```

---

### 5. `deployment/kubernetes/configmap.yaml` (~40 LOC)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: morphml-config
data:
  postgres_host: "postgres.default.svc.cluster.local"
  postgres_port: "5432"
  postgres_db: "morphml"
  redis_host: "redis.default.svc.cluster.local"
  redis_port: "6379"
  s3_endpoint: "http://minio.default.svc.cluster.local:9000"
  s3_bucket: "morphml-artifacts"
```

---

### 6. `deployment/helm/morphml/Chart.yaml` (~20 LOC)

```yaml
apiVersion: v2
name: morphml
description: A Helm chart for MorphML distributed NAS
type: application
version: 0.1.0
appVersion: "1.0.0"
keywords:
  - machine-learning
  - neural-architecture-search
  - distributed-computing
maintainers:
  - name: Eshan Roy
    email: eshanized@proton.me
```

---

### 7. `deployment/helm/morphml/values.yaml` (~150 LOC)

```yaml
# Master configuration
master:
  image:
    repository: tiverse/morphml-master
    tag: latest
    pullPolicy: IfNotPresent
  
  replicas: 1
  
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
  
  service:
    type: ClusterIP
    port: 50051

# Worker configuration
worker:
  image:
    repository: tiverse/morphml-worker
    tag: latest
    pullPolicy: IfNotPresent
  
  replicas: 4
  
  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
      nvidia.com/gpu: 1
    limits:
      memory: "16Gi"
      cpu: "8"
      nvidia.com/gpu: 1
  
  nodeSelector:
    accelerator: nvidia-tesla-v100
  
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 50
    targetCPUUtilizationPercentage: 80

# PostgreSQL
postgresql:
  enabled: true
  host: postgres
  port: 5432
  database: morphml
  username: morphml
  password: changeme

# Redis
redis:
  enabled: true
  host: redis
  port: 6379

# MinIO (S3-compatible)
minio:
  enabled: true
  endpoint: http://minio:9000
  bucket: morphml-artifacts
  accessKey: minioadmin
  secretKey: minioadmin

# Monitoring
monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

---

### 8. `deployment/helm/morphml/templates/deployment.yaml` (~200 LOC)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "morphml.fullname" . }}-master
  labels:
    {{- include "morphml.labels" . | nindent 4 }}
    component: master
spec:
  replicas: {{ .Values.master.replicas }}
  selector:
    matchLabels:
      {{- include "morphml.selectorLabels" . | nindent 6 }}
      component: master
  template:
    metadata:
      labels:
        {{- include "morphml.selectorLabels" . | nindent 8 }}
        component: master
    spec:
      containers:
      - name: master
        image: "{{ .Values.master.image.repository }}:{{ .Values.master.image.tag }}"
        imagePullPolicy: {{ .Values.master.image.pullPolicy }}
        ports:
        - containerPort: {{ .Values.master.service.port }}
          name: grpc
        env:
        - name: POSTGRES_HOST
          value: {{ .Values.postgresql.host }}
        - name: REDIS_HOST
          value: {{ .Values.redis.host }}
        resources:
          {{- toYaml .Values.master.resources | nindent 10 }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "morphml.fullname" . }}-worker
  labels:
    {{- include "morphml.labels" . | nindent 4 }}
    component: worker
spec:
  replicas: {{ .Values.worker.replicas }}
  selector:
    matchLabels:
      {{- include "morphml.selectorLabels" . | nindent 6 }}
      component: worker
  template:
    metadata:
      labels:
        {{- include "morphml.selectorLabels" . | nindent 8 }}
        component: worker
    spec:
      containers:
      - name: worker
        image: "{{ .Values.worker.image.repository }}:{{ .Values.worker.image.tag }}"
        imagePullPolicy: {{ .Values.worker.image.pullPolicy }}
        env:
        - name: MASTER_HOST
          value: {{ include "morphml.fullname" . }}-master
        resources:
          {{- toYaml .Values.worker.resources | nindent 10 }}
      {{- with .Values.worker.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

---

### 9. `deployment/scripts/deploy.sh` (~100 LOC)

```bash
#!/bin/bash
# Deploy MorphML to Kubernetes

set -e

NAMESPACE=${NAMESPACE:-morphml}
RELEASE_NAME=${RELEASE_NAME:-morphml}

echo "Deploying MorphML to namespace: $NAMESPACE"

# Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Install dependencies
echo "Installing PostgreSQL..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --namespace $NAMESPACE \
  --set auth.database=morphml

echo "Installing Redis..."
helm install redis bitnami/redis \
  --namespace $NAMESPACE

echo "Installing MinIO..."
helm install minio bitnami/minio \
  --namespace $NAMESPACE

# Install MorphML
echo "Installing MorphML..."
helm install $RELEASE_NAME ./deployment/helm/morphml \
  --namespace $NAMESPACE \
  --values ./deployment/helm/morphml/values.yaml

echo "Deployment complete!"
echo "Check status: kubectl get pods -n $NAMESPACE"
```

---

### 10. `deployment/monitoring/prometheus.yaml` (~200 LOC)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
      - job_name: 'morphml-master'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_component]
            regex: master
            action: keep
      
      - job_name: 'morphml-worker'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_component]
            regex: worker
            action: keep
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
```

---

## 🧪 Usage

```bash
# Build Docker images
docker build -f deployment/docker/Dockerfile.master -t tiverse/morphml-master:latest .
docker build -f deployment/docker/Dockerfile.worker -t tiverse/morphml-worker:latest .

# Push to registry
docker push tiverse/morphml-master:latest
docker push tiverse/morphml-worker:latest

# Deploy with Helm
./deployment/scripts/deploy.sh

# Check status
kubectl get pods -n morphml

# Scale workers
kubectl scale deployment morphml-worker --replicas=10 -n morphml

# View logs
kubectl logs -f deployment/morphml-master -n morphml
```

---

## ✅ Deliverables

- [ ] Docker images for master and worker
- [ ] Kubernetes manifests
- [ ] Helm chart with parameterized configuration
- [ ] Auto-scaling configuration
- [ ] Deployment scripts
- [ ] Prometheus monitoring setup
- [ ] Documentation for cloud providers (GKE, EKS, AKS)

---

**Phase 3 Complete!** 🎉

Total Phase 3 LOC: ~20,000 production code

**Next Phase:** Phase 4 - Meta-Learning
