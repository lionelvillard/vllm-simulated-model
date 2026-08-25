# vllm-sim-deps Container Image

Pre-built container image that bundles vLLM simulated model dependencies to speed
up Kubernetes pod startup.

## Contents

The image packages:
- **vllm-simulated-model plugin** — provides simulated latency models
- **nixl** — KV cache transfer library for prefill/decode disaggregation

## Building and Pushing

**Prerequisites:**
```bash
# Authenticate with GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u lionelvillard --password-stdin
```

**Build and push:**
```bash
# From repo root
./docker/vllm-sim-deps/build.sh v0.1.0
```

The script builds the image and pushes it to `ghcr.io/lionelvillard/vllm-sim-deps:v0.1.0`.

## Version Compatibility

| Image Version | Plugin Version | nixl Version | vLLM Version | Notes |
|---------------|----------------|--------------|--------------|-------|
| v0.1.0        | 0.1.0          | 1.3.2        | ≥0.6.8       | Initial release with NixlConnector support |

## Updating to a New Version

1. **Update the Dockerfile** with new version tags:
   ```dockerfile
   # Change the git tag
   RUN pip install --target=/plugins --no-deps --no-cache-dir \
       https://github.com/lionelvillard/vllm-simulated-model/archive/refs/tags/v0.2.0.tar.gz
   
   # Update nixl if needed
   RUN pip install --target=/nixl-deps --no-cache-dir nixl==1.4.0
   ```

2. **Build and push:**
   ```bash
   ./docker/vllm-sim-deps/build.sh v0.2.0
   ```

3. **Update deployment manifests** across the repo:
   ```bash
   find models -name "*-deployment.yaml" -exec \
     sed -i '' 's|vllm-sim-deps:v0.1.0|vllm-sim-deps:v0.2.0|g' {} +
   ```

4. **Update the version table above** with the new version entry.

## Development Builds

For testing unreleased changes:

```bash
# Edit Dockerfile to use main branch
sed -i '' 's|refs/tags/v[0-9.]*|refs/heads/main|' docker/vllm-sim-deps/Dockerfile

# Build and push with dev tag
cd /path/to/repo
docker build -f docker/vllm-sim-deps/Dockerfile -t ghcr.io/lionelvillard/vllm-sim-deps:dev .
docker push ghcr.io/lionelvillard/vllm-sim-deps:dev

# Revert Dockerfile
git checkout docker/vllm-sim-deps/Dockerfile
```

> [!WARNING]
> Development builds from `main` are not guaranteed to be stable. Always use
> tagged releases for production deployments.

## Versioning Scheme

Image version follows the plugin version:

- **Format:** `v<major>.<minor>.<patch>` (e.g., v0.1.0)
- **Plugin version:** Matches the git tag used in the Dockerfile
- **nixl version:** Specified in Dockerfile, tracked in compatibility table
- **vLLM compatibility:** Documented based on testing

### When to Bump Versions

| Change | Action |
|--------|--------|
| Plugin update (features, bug fixes) | Create new git tag, rebuild with new version |
| nixl update (security, compatibility) | Bump patch version, update Dockerfile |
| Breaking vLLM integration changes | Bump major version |
