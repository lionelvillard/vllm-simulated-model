# Minimal image for vLLM simulated model plugin
# NIXL is no longer included - use llm-d-cpu base image which has NIXL pre-installed
#
# Build from repo root:
#   docker build -f docker/vllm-sim-deps/Dockerfile -t ghcr.io/lionelvillard/vllm-sim-deps:v0.3.0 .
# Push:
#   docker push ghcr.io/lionelvillard/vllm-sim-deps:v0.3.0

FROM python:3.12-slim AS builder

# Copy local source code
COPY src /build/src
COPY pyproject.toml /build/

# Install plugin from local clone (no bytecode compilation, no dependencies - torch provided by vLLM)
RUN pip install --target=/plugins --no-deps --no-cache-dir --no-compile /build

# Size optimization: remove unnecessary metadata and test files
RUN find /plugins -type d -name "*.dist-info" -exec sh -c 'cd "$1" && ls | grep -v "top_level.txt\|entry_points.txt\|METADATA" | xargs rm -f' _ {} \; && \
    find /plugins -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /plugins -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Create minimal final image with just the plugin
# Use busybox to provide sh and cp for Kubernetes initContainer commands
FROM busybox:1.36
COPY --from=builder /plugins /plugins
