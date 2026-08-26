# vllm-sim-deps Container Image

Pre-built container image that bundles the vLLM simulated model plugin to speed
up Kubernetes pod startup.

## Contents

The image packages:
- **vllm-simulated-model plugin** — provides simulated latency models

> [!NOTE]
> NIXL is no longer included in this image. Use `ghcr.io/llm-d/llm-d-cpu` base
> image which has NIXL pre-installed.

## Building and Pushing

**Prerequisites:**
```bash
# Authenticate with GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u lionelvillard --password-stdin
```

**Build and push:**
```bash
# From repo root
./docker/vllm-sim-deps/build.sh v0.3.0
```

The script builds the image and pushes it to `ghcr.io/lionelvillard/vllm-sim-deps:v0.3.0`.

## Version Compatibility

| Image Version | Plugin Version | Base Image | Notes |
|---------------|----------------|------------|-------|
| v0.1.0        | 0.1.0          | N/A        | Legacy: included NIXL 1.3.2 |
| v0.2.0        | 0.1.0+         | ghcr.io/llm-d/llm-d-cpu:v0.9.0 | Plugin only - NIXL in base image |
| v0.3.0        | 0.1.0+         | ghcr.io/llm-d/llm-d-cpu:v0.9.0 | Plugin only - NIXL in base image |

## Updating to a New Version

1. **Build and push** with the new version tag (no Dockerfile changes needed — the
   plugin is always installed from the local source tree at build time):
   ```bash
   ./docker/vllm-sim-deps/build.sh <new-version>
   ```

2. **Update deployment manifests** across the repo:
   ```bash
   find models -name "*-deployment.yaml" -exec \
     sed -i '' 's|vllm-sim-deps:v0.3.0|vllm-sim-deps:<new-version>|g' {} +
   ```

4. **Update the version table above** with the new version entry.

## Development Builds

For testing unreleased changes, build directly from your working tree (no
Dockerfile modification needed — it always installs from the local source):

```bash
# From repo root
docker build -f docker/vllm-sim-deps/Dockerfile -t ghcr.io/lionelvillard/vllm-sim-deps:dev .
docker push ghcr.io/lionelvillard/vllm-sim-deps:dev
```

> [!WARNING]
> Development builds from `main` are not guaranteed to be stable. Always use
> tagged releases for production deployments.

## Versioning Scheme

Image version indicates the plugin packaging:

- **Format:** `v<major>.<minor>.<patch>` (e.g., v0.2.0)
- **Plugin version:** Matches the git tag used in the Dockerfile
- **Base image compatibility:** Documented in compatibility table

### When to Bump Versions

| Change | Action |
|--------|--------|
| Plugin update (features, bug fixes) | Create new git tag, rebuild with new version |
| Base image change (llm-d-cpu version) | Update compatibility table |
| Breaking plugin API changes | Bump major version |
