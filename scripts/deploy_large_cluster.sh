#!/bin/bash
# Deploy and validate MorphML on large cluster (50+ workers)
#
# This script automates:
# 1. Cluster setup and validation
# 2. MorphML deployment with optimized settings
# 3. Monitoring stack deployment
# 4. Validation test execution
# 5. Result collection and reporting
#
# Author: Eshan Roy <eshanized@proton.me>
# Organization: TONMOY INFRASTRUCTURE & VISION

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-morphml-large}"
RELEASE_NAME="${RELEASE_NAME:-morphml}"
NUM_WORKERS="${NUM_WORKERS:-50}"
HELM_CHART_PATH="${HELM_CHART_PATH:-./deployment/helm/morphml}"
VALUES_FILE="${VALUES_FILE:-./deployment/kubernetes/large-cluster-values.yaml}"
VALIDATION_DURATION="${VALIDATION_DURATION:-1}"  # hours
OUTPUT_DIR="${OUTPUT_DIR:-./validation_results}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm not found. Please install helm 3.x."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    
    # Check GPU availability
    local gpu_nodes=$(kubectl get nodes -o json | jq '[.items[] | select(.status.capacity."nvidia.com/gpu" != null)] | length')
    if [ "$gpu_nodes" -lt 10 ]; then
        log_warning "Found only $gpu_nodes GPU nodes. Recommended: at least 10 nodes for 50+ workers."
    fi
    
    log_success "Prerequisites check passed"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace $NAMESPACE
        log_success "Namespace created"
    fi
}

install_nvidia_device_plugin() {
    log_info "Checking NVIDIA device plugin..."
    
    if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &> /dev/null; then
        log_success "NVIDIA device plugin already installed"
    else
        log_info "Installing NVIDIA device plugin..."
        kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
        sleep 10
        log_success "NVIDIA device plugin installed"
    fi
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus
    log_info "Deploying Prometheus..."
    kubectl apply -f deployment/monitoring/prometheus-config.yaml -n $NAMESPACE || true
    
    # Deploy Grafana
    log_info "Deploying Grafana..."
    kubectl apply -f deployment/monitoring/grafana-dashboard.json -n $NAMESPACE || true
    
    log_success "Monitoring stack deployed"
}

deploy_morphml() {
    log_info "Deploying MorphML with $NUM_WORKERS workers..."
    
    # Update worker count in values
    local temp_values=$(mktemp)
    cp $VALUES_FILE $temp_values
    
    # Deploy with Helm
    helm upgrade --install $RELEASE_NAME $HELM_CHART_PATH \
        --namespace $NAMESPACE \
        --values $temp_values \
        --set worker.replicaCount=$NUM_WORKERS \
        --set worker.autoscaling.minReplicas=$NUM_WORKERS \
        --set worker.autoscaling.maxReplicas=$((NUM_WORKERS * 2)) \
        --wait \
        --timeout 15m
    
    rm -f $temp_values
    
    log_success "MorphML deployed"
}

wait_for_workers() {
    log_info "Waiting for workers to be ready..."
    
    local timeout=600  # 10 minutes
    local elapsed=0
    local ready_workers=0
    
    while [ $elapsed -lt $timeout ]; do
        ready_workers=$(kubectl get pods -n $NAMESPACE -l component=worker -o json | \
            jq '[.items[] | select(.status.phase=="Running")] | length')
        
        if [ "$ready_workers" -ge $NUM_WORKERS ]; then
            log_success "$ready_workers workers ready"
            return 0
        fi
        
        log_info "Workers ready: $ready_workers/$NUM_WORKERS"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    
    log_error "Timeout waiting for workers. Only $ready_workers/$NUM_WORKERS ready."
    return 1
}

run_connectivity_test() {
    log_info "Running connectivity test..."
    
    # Get master pod
    local master_pod=$(kubectl get pods -n $NAMESPACE -l component=master -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$master_pod" ]; then
        log_error "Master pod not found"
        return 1
    fi
    
    # Check master logs for worker connections
    log_info "Checking master logs for worker registrations..."
    kubectl logs $master_pod -n $NAMESPACE | grep -i "worker registered" | tail -20
    
    log_success "Connectivity test complete"
}

run_validation_suite() {
    log_info "Running validation suite (duration: ${VALIDATION_DURATION}h)..."
    
    # Create output directory
    mkdir -p $OUTPUT_DIR
    
    # Get master service endpoint
    local master_host=$(kubectl get svc -n $NAMESPACE morphml-master -o jsonpath='{.spec.clusterIP}')
    local master_port=50051
    
    log_info "Master endpoint: $master_host:$master_port"
    
    # Port forward for local validation client
    log_info "Setting up port forward..."
    kubectl port-forward -n $NAMESPACE svc/morphml-master 50051:50051 &
    local port_forward_pid=$!
    sleep 5
    
    # Run validation
    log_info "Executing validation tests..."
    python3 benchmarks/distributed/validate_large_cluster.py \
        --master-host localhost \
        --master-port 50051 \
        --workers $NUM_WORKERS \
        --output-dir $OUTPUT_DIR \
        --format markdown
    
    # Kill port forward
    kill $port_forward_pid 2>/dev/null || true
    
    log_success "Validation suite complete"
}

collect_metrics() {
    log_info "Collecting metrics..."
    
    # Prometheus metrics
    if kubectl get svc -n $NAMESPACE prometheus &> /dev/null; then
        log_info "Exporting Prometheus metrics..."
        kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090 &
        local prom_pid=$!
        sleep 5
        
        # Query key metrics
        curl -s "http://localhost:9090/api/v1/query?query=morphml_tasks_completed_total" > $OUTPUT_DIR/metrics_tasks.json
        curl -s "http://localhost:9090/api/v1/query?query=morphml_worker_utilization" > $OUTPUT_DIR/metrics_utilization.json
        
        kill $prom_pid 2>/dev/null || true
    fi
    
    # Pod metrics
    log_info "Collecting pod metrics..."
    kubectl top pods -n $NAMESPACE > $OUTPUT_DIR/pod_metrics.txt
    kubectl top nodes > $OUTPUT_DIR/node_metrics.txt
    
    # Event logs
    log_info "Collecting event logs..."
    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' > $OUTPUT_DIR/events.txt
    
    log_success "Metrics collected"
}

generate_summary() {
    log_info "Generating validation summary..."
    
    cat > $OUTPUT_DIR/VALIDATION_SUMMARY.md <<EOF
# MorphML Large Cluster Validation Summary

## Cluster Configuration
- **Namespace**: $NAMESPACE
- **Workers**: $NUM_WORKERS
- **Validation Duration**: ${VALIDATION_DURATION}h
- **Timestamp**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Deployment Status
\`\`\`
$(kubectl get all -n $NAMESPACE)
\`\`\`

## Worker Distribution
\`\`\`
$(kubectl get pods -n $NAMESPACE -l component=worker -o wide | head -20)
\`\`\`

## Resource Usage
\`\`\`
$(kubectl top pods -n $NAMESPACE | head -20)
\`\`\`

## Validation Results
See detailed reports in:
- \`$OUTPUT_DIR/validation_report_*.md\`
- \`$OUTPUT_DIR/validation_report_*.json\`

## Prometheus Metrics
Access Grafana dashboard:
\`\`\`bash
kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000
# Open http://localhost:3000
\`\`\`

## Next Steps
1. Review validation reports
2. Check Grafana dashboards for performance trends
3. Analyze failure patterns if any
4. Scale up/down as needed

---
*Generated by deploy_large_cluster.sh*
EOF

    log_success "Summary generated: $OUTPUT_DIR/VALIDATION_SUMMARY.md"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Kill any remaining port forwards
    pkill -f "kubectl port-forward" 2>/dev/null || true
    
    log_success "Cleanup complete"
}

main() {
    echo ""
    echo "=========================================="
    echo "MorphML Large Cluster Deployment"
    echo "=========================================="
    echo "Namespace:        $NAMESPACE"
    echo "Workers:          $NUM_WORKERS"
    echo "Validation:       ${VALIDATION_DURATION}h"
    echo "Output:           $OUTPUT_DIR"
    echo "=========================================="
    echo ""
    
    # Trap cleanup
    trap cleanup EXIT
    
    # Execute deployment pipeline
    check_prerequisites
    create_namespace
    install_nvidia_device_plugin
    deploy_monitoring
    deploy_morphml
    wait_for_workers
    run_connectivity_test
    run_validation_suite
    collect_metrics
    generate_summary
    
    echo ""
    log_success "=========================================="
    log_success "Deployment and validation complete!"
    log_success "=========================================="
    echo ""
    log_info "Results available in: $OUTPUT_DIR"
    log_info "View summary: cat $OUTPUT_DIR/VALIDATION_SUMMARY.md"
    echo ""
}

# Help text
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    cat <<EOF
Usage: $0 [options]

Deploy and validate MorphML on a large Kubernetes cluster (50+ workers).

Environment Variables:
  NAMESPACE               Kubernetes namespace (default: morphml-large)
  NUM_WORKERS            Number of workers (default: 50)
  VALIDATION_DURATION    Validation duration in hours (default: 1)
  OUTPUT_DIR             Output directory (default: ./validation_results)

Examples:
  # Deploy with 50 workers
  ./scripts/deploy_large_cluster.sh

  # Deploy with 100 workers
  NUM_WORKERS=100 ./scripts/deploy_large_cluster.sh

  # Deploy with extended validation (4 hours)
  VALIDATION_DURATION=4 ./scripts/deploy_large_cluster.sh

  # Custom namespace
  NAMESPACE=morphml-prod NUM_WORKERS=75 ./scripts/deploy_large_cluster.sh

EOF
    exit 0
fi

# Run main
main "$@"
