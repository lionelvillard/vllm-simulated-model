# Multi-stage build for vLLM simulated model dependencies
# Combines plugin installation and nixl dependencies into a single image
#
# Build from repo root:
#   docker build -f docker/vllm-sim-deps/Dockerfile -t ghcr.io/lionelvillard/vllm-sim-deps:v0.1.0 .
# Push:
#   docker push ghcr.io/lionelvillard/vllm-sim-deps:v0.1.0

FROM python:3.12-slim as builder

# Install plugin
RUN pip install --target=/plugins --no-deps --no-cache-dir \
    https://github.com/lionelvillard/vllm-simulated-model/archive/refs/tarball/main.tar.gz

# Install nixl
RUN pip install --target=/nixl-deps --no-cache-dir nixl==1.3.2

# Create minimal final image with just the installed packages
FROM scratch
COPY --from=builder /plugins /plugins
COPY --from=builder /nixl-deps /nixl-deps
