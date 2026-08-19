# Kubernetes CPU Deployment Design

**Date:** 2026-08-19  
**Status:** Approved

## Goal

Deploy vLLM with the simulated model plugin on CPU-only Kubernetes nodes (linux/amd64) without building a custom image. The plugin Python package is injected at pod startup via an init container; the model config is injected via a ConfigMap.

## Constraints

- No custom image build
- linux/amd64, CPU-only nodes (no GPU)
- External network access available at pod startup
- vLLM plugin (`vllm_simulated`) must be registered via its `vllm.general_plugins` entry point

## Resources

| Kind | Name | Purpose |
|---|---|---|
| ConfigMap | `vllm-sim-model-config` | holds `config.json` for the simulated model |
| Deployment | `vllm-sim` | single replica, CPU-only, no GPU scheduling |
| Service | `vllm-sim` | ClusterIP on port 8000 |

A single `emptyDir` volume named `plugins` is shared between the init container and the main container.

## Init Container

**Image:** `python:3.12-slim` (small; only pip is needed)

**Command:**
```
pip install --target /plugins --no-deps \
  git+https://github.com/vllm-project/vllm-simulated-model.git@main
```

- `--no-deps` skips pulling `torch` (already in the main container)
- `--target /plugins` writes package files + dist-info into the shared volume
- The dist-info written by pip is what makes the `vllm.general_plugins` entry point discoverable via `importlib.metadata`
- Pin `@main` to a specific commit SHA for reproducibility in production

**Volume mount:** `plugins` emptyDir at `/plugins`

## Main Container

**Image:** `vllm/vllm-openai-cpu:latest-x86_64`  
(CPU-specific image published by the vLLM project on Docker Hub; pin to `v{VERSION}-x86_64` for production)

**Environment variables:**

| Variable | Value | Notes |
|---|---|---|
| `PYTHONPATH` | `/plugins` | makes plugin discoverable to Python's entry-point loader |
| `VLLM_CPU_KVCACHE_SPACE` | `4` | KV cache size in GB; tune to node memory |
| `VLLM_CPU_OMP_THREADS_BIND` | `0-3` | binds OpenMP threads to cores 0–3; match to CPU request |

**Command:**
```
vllm serve /model \
  --load-format dummy \
  --device cpu \
  --skip-tokenizer-init \
  --gpu-memory-utilization 0.5
```

- `/model` is the ConfigMap mount containing `config.json`
- `--skip-tokenizer-init` avoids any HuggingFace download at startup
- `--device cpu` forces CPU execution path
- `--gpu-memory-utilization 0.5` controls CPU memory usage (despite the flag name, this is how vLLM caps memory in CPU mode)

**Volume mounts:**
- `plugins` emptyDir at `/plugins`
- `vllm-sim-model-config` ConfigMap at `/model` (read-only)

**Resources:**
```yaml
requests:
  cpu: "4"
  memory: 8Gi
limits:
  cpu: "4"
  memory: 8Gi
```

**Probes:** liveness and readiness on `GET /health` (port 8000), `initialDelaySeconds: 60` to account for init container pip install time.

## ConfigMap: model config

Contains the `config.json` for the simulated model. The example from `examples/sim-qwen-3.8-27b/config.json`:

```json
{
  "architectures": ["SimulatedForCausalLM"],
  "model_type": "qwen3_5",
  "hidden_size": 5120,
  "num_hidden_layers": 64,
  "num_attention_heads": 24,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "vocab_size": 248320,
  "max_position_embeddings": 262144,
  "eos_token_id": 248044,
  "torch_dtype": "bfloat16",
  "latency": {
    "base_ms": 5.0,
    "prefill_ms_per_token": 0.05,
    "decode_ms_per_seq": 1.2,
    "ctx_ms_per_ktoken": 0.3,
    "deterministic_length": true
  }
}
```

## Service

ClusterIP exposing port 8000 (OpenAI-compatible API endpoint).

## Startup Sequence

1. Init container installs `vllm_simulated` into `/plugins` (shared emptyDir)
2. Main container starts; `PYTHONPATH=/plugins` makes the package and its dist-info visible
3. vLLM discovers the `vllm.general_plugins` entry point and calls `vllm_simulated.register()`
4. `SimulatedForCausalLM` is registered with vLLM's ModelRegistry
5. vLLM loads the model from `/model/config.json` with `--load-format dummy` (no weights to fetch)
6. Server becomes ready on port 8000

## Tuning Notes

- `VLLM_CPU_KVCACHE_SPACE`: increase for higher parallelism; must stay well below node memory
- `VLLM_CPU_OMP_THREADS_BIND`: should match the CPU request/limit range
- `--gpu-memory-utilization`: set to avoid OOM; 0.5 is conservative
- Pin the git ref in the init container command to a commit SHA before production use
- Startup time includes pip install (~10–30s depending on network); factor into readiness probe delay

## Post-Smoke-Test Amendments

The following amendments were made after smoke testing against the actual vLLM v0.27.1 CPU image:

### 1. Init Container Install Method: Git → Tarball URL

**Change:**
```
# BEFORE
git+https://github.com/vllm-project/vllm-simulated-model.git@main

# AFTER
https://github.com/lionelvillard/vllm-simulated-model/archive/refs/heads/main.tar.gz
```

**Rationale:** The git-based install URL requires git to be present in the init container and may trigger credential prompting in OpenShift environments. pip handles tarball downloads natively without requiring git, so there is no functional dependency. This also allows reverting the init container image back to the slimmer base image.

**Additional Flag:** Added `--no-cache-dir` to pip install to reduce the init container's transient layer size and improve startup performance.

**GitHub Org Correction:** Corrected to use the official fork `lionelvillard/vllm-simulated-model` pending contribution back to the main vLLM project organization.

### 2. Init Container Image: Slim Confirmed

**Change:** Confirmed use of `python:3.12-slim` (~50 MB vs. ~350 MB for `python:3.12`)

**Rationale:** Since the install method no longer requires git, the full `python:3.12` image is unnecessary. The slim image includes pip and is sufficient for the tarball install.

### 3. Removed Invalid `--device=cpu` Flag

**Change:**
```
# BEFORE
vllm serve /model --load-format dummy --device cpu --skip-tokenizer-init ...

# AFTER
vllm serve /model --load-format dummy --tokenizer=gpt2 ...
```

**Rationale:** The `--device cpu` flag is not valid in vLLM v0.27.1 CPU image and causes a parse error (`device_ids='cpu'`). The CPU image defaults to CPU mode; the flag is redundant and unsupported.

### 4. Replaced `--skip-tokenizer-init` with `--tokenizer=gpt2`

**Rationale:** `--skip-tokenizer-init` breaks all inference endpoints (`/v1/completions` and `/v1/chat/completions`) because vLLM still requires a tokenizer for message encoding/decoding. The gpt2 tokenizer is lightweight (~0.5 MB) and is downloaded from HuggingFace at startup. This approach balances startup latency with correctness and is operationally simpler than conditional initialization.

### 5. Added OpenShift Non-Root SCC Cache Redirection

**Change:** Added environment variables to the main container:
```yaml
- name: HOME
  value: /tmp
- name: XDG_CACHE_HOME
  value: /tmp/cache
```

**Rationale:** OpenShift's restricted SCC (Security Context Constraint) for non-root pods prevents writes to `/.cache` and `/.triton` in the root filesystem. These environment variables redirect cache paths to `/tmp`, which is writable by non-root processes. This is required for vLLM to function in restricted OpenShift environments.

### 6. Added Shared Memory emptyDir Volume

**Change:** Added `/dev/shm` memory-backed emptyDir:
```yaml
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: 256Mi
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: 256Mi
```

**Rationale:** vLLM v0.27.1 engine IPC requires greater than 160 Mi of shared memory. The Kubernetes default for `/dev/shm` is 64 Mi, which causes engine initialization failures. The 256 Mi allocation provides sufficient headroom for concurrent request handling.
