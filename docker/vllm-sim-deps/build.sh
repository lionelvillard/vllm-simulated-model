#!/usr/bin/env bash
# Build and push the vllm-sim-deps container image
#
# Usage (from repo root):
#   docker/vllm-sim-deps/build.sh [VERSION]
#
# Examples:
#   docker/vllm-sim-deps/build.sh v0.1.0
#   docker/vllm-sim-deps/build.sh v0.2.0

set -euo pipefail

VERSION="${1:-v0.1.0}"
REGISTRY="ghcr.io/lionelvillard"
IMAGE_NAME="vllm-sim-deps"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "Building ${FULL_IMAGE}..."
docker build -f docker/vllm-sim-deps/Dockerfile -t "${FULL_IMAGE}" .

echo "Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"

echo ""
echo "✓ Image built and pushed successfully!"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "The deployment manifests in models/*/deployments/*/k8s/ are configured to use this image."
