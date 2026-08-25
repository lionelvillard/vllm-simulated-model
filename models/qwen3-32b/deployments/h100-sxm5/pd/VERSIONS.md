# Dependency Version Compatibility

This document tracks version compatibility between the init container dependencies
image and vLLM versions for the Qwen3-32B P/D deployment.

## Current Version

**Init container image:** `ghcr.io/lionelvillard/vllm-sim-deps:v0.1.0`

## Compatibility Matrix

| Init Image | Plugin Version | nixl Version | vLLM Version | Notes |
|------------|----------------|--------------|--------------|-------|
| v0.1.0     | 0.1.0          | 1.3.2        | ≥0.6.8       | Initial release with NixlConnector support |

## Version Components

Each init container image version bundles:

1. **vllm-simulated-model plugin** — provides the simulated latency model
2. **nixl** — KV cache transfer library for P/D disaggregation

## Image Build & Release

Build and push the dependencies image:

```bash
cd models/qwen3-32b/deployments/h100-sxm5/pd/k8s

# Authenticate with GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u lionelvillard --password-stdin

# Build and push
./build-deps-image.sh v0.1.0
```

The image is published to `ghcr.io/lionelvillard/vllm-sim-deps`.

## Versioning Scheme

The init container image version follows the plugin version:

- **Format:** `v<major>.<minor>.<patch>` (e.g., v0.1.0)
- **Plugin version:** Matches the git tag used in the Dockerfile
- **nixl version:** Specified in Dockerfile, tracked in compatibility table
- **vLLM compatibility:** Documented based on testing

### When to bump versions

| Change | Action |
|--------|--------|
| Plugin update (new features, bug fixes) | Create new git tag, rebuild image with new version |
| nixl update (security, compatibility) | Bump patch version, update Dockerfile, rebuild |
| Breaking changes to vLLM integration | Bump major version |

## Updating to a New Version

1. **Update the Dockerfile:**
   ```dockerfile
   # Change the git tag
   RUN pip install --target=/plugins --no-deps --no-cache-dir \
       https://github.com/lionelvillard/vllm-simulated-model/archive/refs/tags/v0.2.0.tar.gz
   
   # Update nixl if needed
   RUN pip install --target=/nixl-deps --no-cache-dir nixl==1.4.0
   ```

2. **Build and push the new image:**
   ```bash
   ./build-deps-image.sh v0.2.0
   ```

3. **Update deployment manifests:**
   ```bash
   sed -i '' 's|vllm-sim-deps:v0.1.0|vllm-sim-deps:v0.2.0|g' \
     prefill-deployment.yaml decode-deployment.yaml
   ```

4. **Update this compatibility table** with the new version entry.

5. **Test the deployment** before rolling out to production.

## Using Development Builds

For development and testing, you can build from the `main` branch:

```bash
# Edit Dockerfile.deps to use main branch instead of a tag
sed -i '' 's|refs/tags/v[0-9.]*|refs/heads/main|' Dockerfile.deps

# Build with dev tag
docker build -f Dockerfile.deps -t ghcr.io/lionelvillard/vllm-sim-deps:dev .
docker push ghcr.io/lionelvillard/vllm-sim-deps:dev
```

> [!WARNING]
> Development builds from `main` are not guaranteed to be stable. Always use
> tagged releases for production deployments.
