# MorphML on Google Kubernetes Engine (GKE)

Deploy MorphML on GKE with GPU support, autoscaling, and managed services.

---

## Prerequisites

- Google Cloud account
- `gcloud` CLI installed
- Project with billing enabled
- Necessary APIs enabled

---

## Setup

### 1. Install gcloud CLI

```bash
# Install
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable APIs

```bash
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
```

---

## Create GKE Cluster

### Standard Cluster with GPU

```bash
# Set variables
export PROJECT_ID=your-project-id
export CLUSTER_NAME=morphml-cluster
export REGION=us-central1
export ZONE=us-central1-a

# Create cluster
gcloud container clusters create $CLUSTER_NAME \
  --region=$REGION \
  --num-nodes=2 \
  --machine-type=n1-standard-4 \
  --disk-size=100 \
  --disk-type=pd-ssd \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --addons=HttpLoadBalancing,HorizontalPodAutoscaling \
  --enable-stackdriver-kubernetes

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=2 \
  --min-nodes=1 \
  --max-nodes=20 \
  --enable-autoscaling \
  --disk-size=200 \
  --disk-type=pd-ssd

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
```

### Install NVIDIA GPU drivers

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

---

## Deploy MorphML

### 1. Create namespace

```bash
kubectl create namespace morphml
```

### 2. Configure Cloud Storage for MinIO

**Option A: Use GCS bucket instead of MinIO**

```yaml
# values.yaml
minio:
  enabled: false

# Use GCS credentials
storage:
  type: gcs
  bucket: gs://your-bucket-name
  credentials: /path/to/service-account.json
```

**Option B: Keep MinIO with GCS backend**

```yaml
minio:
  enabled: true
  persistence:
    storageClass: "standard-rwo"
    size: 100Gi
```

### 3. Configure CloudSQL (Optional)

**Instead of PostgreSQL in cluster**:

```bash
# Create CloudSQL instance
gcloud sql instances create morphml-db \
  --database-version=POSTGRES_14 \
  --tier=db-n1-standard-2 \
  --region=$REGION

# Create database
gcloud sql databases create morphml --instance=morphml-db

# Create user
gcloud sql users create morphml \
  --instance=morphml-db \
  --password=YOUR_PASSWORD
```

Update values.yaml:
```yaml
postgresql:
  enabled: false

externalDatabase:
  host: <CLOUD_SQL_PROXY_SERVICE>
  port: 5432
  database: morphml
  user: morphml
  password: YOUR_PASSWORD
```

### 4. Install with Helm

```bash
# Customize values
cat > gke-values.yaml <<EOF
master:
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"

worker:
  replicas: 4
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-t4
  resources:
    requests:
      memory: "16Gi"
      cpu: "8"
      nvidia.com/gpu: 1
    limits:
      memory: "32Gi"
      cpu: "16"
      nvidia.com/gpu: 1
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 50

postgresql:
  primary:
    persistence:
      storageClass: "standard-rwo"
      size: 50Gi

redis:
  master:
    persistence:
      storageClass: "standard-rwo"
      size: 10Gi

minio:
  persistence:
    storageClass: "standard-rwo"
    size: 200Gi
EOF

# Install
helm install morphml ./deployment/helm/morphml \
  --namespace morphml \
  --values gke-values.yaml \
  --timeout 10m

# Verify
kubectl get pods -n morphml
```

---

## Configure Monitoring

### Enable GKE Monitoring

```bash
# Already enabled with --enable-stackdriver-kubernetes

# View metrics in Cloud Console
https://console.cloud.google.com/kubernetes/workload?project=YOUR_PROJECT
```

### Install Prometheus (Optional)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

---

## Configure External Access

### LoadBalancer

```yaml
# values.yaml
master:
  service:
    type: LoadBalancer
```

Get external IP:
```bash
kubectl get svc -n morphml morphml-master
```

### Ingress

```bash
# Install nginx ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace

# Create ingress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: morphml-ingress
  namespace: morphml
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: morphml.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: morphml-master
            port:
              number: 50051
EOF
```

---

## Cost Optimization

### Use Preemptible VMs

```bash
# Create preemptible node pool for workers
gcloud container node-pools create preemptible-gpu-pool \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --preemptible \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=0 \
  --min-nodes=0 \
  --max-nodes=20 \
  --enable-autoscaling
```

### Enable Cluster Autoscaler

```bash
# Already enabled with --enable-autoscaling
# Nodes will automatically scale based on pending pods
```

### Use Committed Use Discounts

```bash
# Purchase in Cloud Console
https://console.cloud.google.com/compute/commitments
```

---

## Backup & Disaster Recovery

### Backup Strategy

1. **Velero for cluster backup**:
```bash
# Install Velero
wget https://github.com/vmware-tanzu/velero/releases/download/v1.11.0/velero-v1.11.0-linux-amd64.tar.gz
tar -xvf velero-v1.11.0-linux-amd64.tar.gz
sudo mv velero-v1.11.0-linux-amd64/velero /usr/local/bin/

# Setup GCS bucket
gsutil mb gs://$PROJECT_ID-velero-backups/

# Install Velero server
velero install \
    --provider gcp \
    --plugins velero/velero-plugin-for-gcp:v1.7.0 \
    --bucket $PROJECT_ID-velero-backups \
    --secret-file ./credentials-velero

# Create backup
velero backup create morphml-backup --include-namespaces morphml
```

2. **CloudSQL automated backups**:
```bash
gcloud sql instances patch morphml-db \
    --backup-start-time=03:00 \
    --enable-bin-log \
    --retained-backups-count=7
```

---

## Security

### Workload Identity

```bash
# Enable on cluster
gcloud container clusters update $CLUSTER_NAME \
    --workload-pool=$PROJECT_ID.svc.id.goog \
    --region=$REGION

# Create service account
gcloud iam service-accounts create morphml-gsa

# Bind to Kubernetes service account
gcloud iam service-accounts add-iam-policy-binding \
    morphml-gsa@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[morphml/morphml]"

# Annotate K8s service account
kubectl annotate serviceaccount morphml \
    -n morphml \
    iam.gke.io/gcp-service-account=morphml-gsa@$PROJECT_ID.iam.gserviceaccount.com
```

### Network Policies

```bash
# Enable network policy on cluster
gcloud container clusters update $CLUSTER_NAME \
    --enable-network-policy \
    --region=$REGION
```

---

## Monitoring Dashboard

Access GKE dashboards:
- **Workloads**: https://console.cloud.google.com/kubernetes/workload
- **Services**: https://console.cloud.google.com/kubernetes/discovery
- **Logs**: https://console.cloud.google.com/logs
- **Monitoring**: https://console.cloud.google.com/monitoring

---

## Cleanup

```bash
# Delete Helm release
helm uninstall morphml -n morphml

# Delete cluster
gcloud container clusters delete $CLUSTER_NAME --region=$REGION

# Delete other resources
gcloud sql instances delete morphml-db
gsutil -m rm -r gs://your-bucket-name
```

---

## Cost Estimate

**Typical monthly costs** (us-central1):

| Resource | Configuration | Cost/month |
|----------|--------------|------------|
| GKE Cluster | Management fee | $73 |
| Master node pool | 2x n1-standard-4 | ~$150 |
| GPU worker pool | 4x n1-standard-8 + T4 | ~$1,200 |
| Persistent disks | 500GB SSD | ~$85 |
| CloudSQL (optional) | db-n1-standard-2 | ~$180 |
| Network egress | 100GB | ~$12 |
| **Total** | | **~$1,700** |

**Cost reduction strategies**:
- Use preemptible VMs: -70% on compute
- Use committed use discounts: -55% on compute
- Scale down during off-hours
- Use regional persistent disks

---

## Best Practices

1. Use separate node pools for master and workers
2. Enable autoscaling for workers
3. Use preemptible VMs for cost savings
4. Configure Workload Identity
5. Enable Cloud Monitoring
6. Use CloudSQL for production databases
7. Implement proper backup strategy
8. Use network policies for security

---

**Next**: [Monitoring Setup](./monitoring.md)
