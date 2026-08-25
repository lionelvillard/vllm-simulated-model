# Dependency Version Compatibility

This document tracks version compatibility between the init container dependencies
image and vLLM versions for the Qwen3-32B P/D deployment.

## Current Version

**Init container image:** `ghcr.io/lionelvillard/vllm-sim-deps:v0.3.0`

## Compatibility Matrix

| Init Image | Plugin Version | Base Image | vLLM Version | Notes |
|------------|----------------|------------|--------------|-------|
| v0.3.0     | 0.1.0+         | ghcr.io/llm-d/llm-d-cpu:v0.9.0 | ≥0.6.8 | Plugin only - NIXL in base image |
| v0.2.0     | 0.1.0+         | ghcr.io/llm-d/llm-d-cpu:v0.9.0 | ≥0.6.8 | Plugin only - NIXL in base image |
| v0.1.0     | 0.1.0          | N/A        | ≥0.6.8       | Legacy: included NIXL 1.3.2 |

## Version Components

Each init container image version bundles:

1. **vllm-simulated-model plugin** — provides the simulated latency model

> [!NOTE]
> NIXL is no longer included in v0.2.0+. The llm-d-cpu base image provides NIXL pre-installed.

## Image Build & Release

Build and push the dependencies image from the repo root:

```bash
# Authenticate with GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u lionelvillard --password-stdin

# Build and push
./docker/vllm-sim-deps/build.sh v0.3.0
```

The image is published to `ghcr.io/lionelvillard/vllm-sim-deps`.

See [docker/vllm-sim-deps/README.md](../../../../../docker/vllm-sim-deps/README.md) for build details.

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

1. **Update deployment manifests:**
   ```bash
   # Update to new version
   find models/qwen3-32b/deployments/h100-sxm5/pd -name "*-deployment.yaml" -exec \
     sed -i '' 's|vllm-sim-deps:v0.3.0|vllm-sim-deps:v0.4.0|g' {} +
   ```

2. **Update this compatibility table** with the new version entry.

3. **Test the deployment** before rolling out to production.

For building new image versions, see [docker/vllm-sim-deps/README.md](../../../../../docker/vllm-sim-deps/README.md).
