#!/bin/bash
# Deploy MorphML to Kubernetes
# Author: Eshan Roy <eshanized@proton.me>
# Organization: TONMOY INFRASTRUCTURE & VISION

set -e

# Configuration
NAMESPACE=${NAMESPACE:-morphml}
RELEASE_NAME=${RELEASE_NAME:-morphml}
CHART_PATH=${CHART_PATH:-./deployment/helm/morphml}
VALUES_FILE=${VALUES_FILE:-./deployment/helm/morphml/values.yaml}

echo "========================================="
echo "  MorphML Kubernetes Deployment"
echo "========================================="
echo ""
echo "Namespace:    $NAMESPACE"
echo "Release:      $RELEASE_NAME"
echo "Chart Path:   $CHART_PATH"
echo "Values File:  $VALUES_FILE"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm is required but not installed. Aborting." >&2; exit 1; }
echo "✓ Prerequisites satisfied"
echo ""

# Create namespace
echo "Creating namespace: $NAMESPACE"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
echo "✓ Namespace created"
echo ""

# Add Helm repositories
echo "Adding Helm repositories..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
echo "✓ Helm repos updated"
echo ""

# Install MorphML with Helm
echo "Installing MorphML..."
helm upgrade --install $RELEASE_NAME $CHART_PATH \
  --namespace $NAMESPACE \
  --values $VALUES_FILE \
  --wait \
  --timeout 10m
echo "✓ MorphML installed"
echo ""

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/morphml-master -n $NAMESPACE || true
kubectl wait --for=condition=available --timeout=300s \
  deployment/morphml-worker -n $NAMESPACE || true
echo "✓ Deployments ready"
echo ""

# Display status
echo "========================================="
echo "  Deployment Status"
echo "========================================="
echo ""
kubectl get pods -n $NAMESPACE
echo ""
kubectl get svc -n $NAMESPACE
echo ""

echo "========================================="
echo "  Deployment Complete!"
echo "========================================="
echo ""
echo "Check logs:"
echo "  kubectl logs -f deployment/morphml-master -n $NAMESPACE"
echo "  kubectl logs -f deployment/morphml-worker -n $NAMESPACE"
echo ""
echo "Scale workers:"
echo "  kubectl scale deployment morphml-worker --replicas=10 -n $NAMESPACE"
echo ""
echo "Port forward master:"
echo "  kubectl port-forward svc/morphml-master 50051:50051 -n $NAMESPACE"
echo ""
