#!/bin/bash
# Build Docker images for MorphML
# Author: Eshan Roy <eshanized@proton.me>

set -e

# Configuration
REGISTRY=${REGISTRY:-tiverse}
VERSION=${VERSION:-latest}
BUILD_CONTEXT=${BUILD_CONTEXT:-.}

echo "========================================="
echo "  MorphML Docker Image Build"
echo "========================================="
echo ""
echo "Registry: $REGISTRY"
echo "Version:  $VERSION"
echo ""

# Build master image
echo "Building master image..."
docker build \
  -f deployment/docker/Dockerfile.master \
  -t ${REGISTRY}/morphml-master:${VERSION} \
  ${BUILD_CONTEXT}
echo "✓ Master image built: ${REGISTRY}/morphml-master:${VERSION}"
echo ""

# Build worker image
echo "Building worker image..."
docker build \
  -f deployment/docker/Dockerfile.worker \
  -t ${REGISTRY}/morphml-worker:${VERSION} \
  ${BUILD_CONTEXT}
echo "✓ Worker image built: ${REGISTRY}/morphml-worker:${VERSION}"
echo ""

# Optional: Push images
if [ "$PUSH" = "true" ]; then
  echo "Pushing images to registry..."
  docker push ${REGISTRY}/morphml-master:${VERSION}
  docker push ${REGISTRY}/morphml-worker:${VERSION}
  echo "✓ Images pushed to registry"
  echo ""
fi

echo "========================================="
echo "  Build Complete!"
echo "========================================="
echo ""
echo "Master image: ${REGISTRY}/morphml-master:${VERSION}"
echo "Worker image: ${REGISTRY}/morphml-worker:${VERSION}"
echo ""
echo "To push images, run:"
echo "  PUSH=true ./deployment/scripts/build.sh"
echo ""
