# Qwen3-32B — H100 SXM5 Prefill/Decode Disaggregation

Prefill/decode disaggregation: a prefill instance processes prompt tokens and
transfers KV caches to a decode instance via `NixlConnector` over UCX, with a
lightweight proxy coordinating the two phases.

The Kubernetes stack under `k8s/` runs the calibrated **physics** simulator —
CPU pods, dummy weights — so you can exercise disaggregated serving and vLLM's
KV-transfer path without any H100s. The `evaluation/` directory contains
latency-model variants and benchmark reports; the `physics` variant is the
calibration source for `k8s/`.

> [!NOTE]
> The simulated P/D stack (CPU + `NixlConnector` + dummy weights) has not yet
> been validated against a live cluster; treat it as a starting point.

## Deployment

### Kubernetes (simulated)

`k8s/` runs the disaggregated topology as the simulated model on CPU nodes — no
GPUs required. 

**One-time setup:**
```bash
export VLLM_SIM_NAMESPACE=default  # or your target namespace
```

Apply the manifests (skip `sim-config.json`; it's the plain-JSON source the 
ConfigMap embeds, not a Kubernetes resource):

```bash
K=models/qwen3-32b/deployments/h100-sxm5/pd/k8s
kubectl apply -n $VLLM_SIM_NAMESPACE \
  -f $K/configmap.yaml \
  -f $K/prefill-deployment.yaml -f $K/prefill-service.yaml \
  -f $K/decode-deployment.yaml  -f $K/decode-service.yaml \
  -f $K/proxy-deployment.yaml   -f $K/proxy-service.yaml
```

This deploys:

| Resource | Role | Notes |
|----------|------|-------|
| `vllm-qwen3-32b-pd-eae748-prefill` | `kv_role: kv_producer` | Sim CPU pod; side-channel on port 5600; `VLLM_NIXL_SIDE_CHANNEL_HOST` = pod IP |
| `vllm-qwen3-32b-pd-eae748-decode` | `kv_role: kv_consumer` | Sim CPU pod; side-channel on port 5601; `VLLM_NIXL_SIDE_CHANNEL_HOST` = pod IP |
| `vllm-qwen3-32b-pd-eae748-proxy` | Request router | Init containers wait for both backends; routing-only, so it needs no GPU |

Both backends run the sim plugin (`vllm/vllm-openai-cpu` image, `--load-format
dummy`) and mount the physics ConfigMap at `/model`. The `6-char` hash `eae748`
is the SHA of the (physics) latency config, matching the ConfigMap and the
`vllm-qwen3-32b-pd-<hash>[-<role>]` naming scheme. Both deployments include an
init container that installs NIXL 1.3.2, which is required for KV cache transfer
via `NixlConnector`.

The proxy still uses the `vllm/vllm-openai` image because it runs vLLM's
`disagg_proxy_demo.py`; it only routes HTTP (no CUDA imports) and declares no GPU
request, so it schedules on a CPU node.

> [!NOTE]
> The proxy deployment includes a patch init container that removes IP address
> validation from `disagg_proxy_demo.py` to support Kubernetes DNS service names
> (the upstream script only accepts `localhost` or IP addresses, not hostnames).

Send all client traffic to `vllm-qwen3-32b-pd-eae748-proxy:8000`. The proxy
forwards the prefill phase (`max_tokens=1`) to `vllm-qwen3-32b-pd-eae748-prefill`,
then sends the full request to `vllm-qwen3-32b-pd-eae748-decode` for streaming
generation.

**HF token:** Both backends read `HF_TOKEN` from a Secret named `hf-token` (key:
`token`) — needed only if the tokenizer (`Qwen/Qwen3-32B`) is gated; the sim
uses dummy weights, so no model download occurs. Create it before applying:

```bash
kubectl create secret generic hf-token -n $VLLM_SIM_NAMESPACE --from-literal=token=hf_...
```

> [!TIP]
> The secret reference is `optional: true`, so pods start without it if the
> tokenizer is already cached or publicly accessible.

**Readiness:** The sim starts quickly — no ~64 GB weight download. Each backend's
`startupProbe` allows up to ~150 s (plugin install + CPU engine init) before
liveness kicks in. The proxy's init containers poll `/health` on each backend
every 5 s and block until both respond.

**Node placement:** The simulated backends request `4` CPU / `8Gi` memory each
and no GPU, so they schedule on ordinary CPU nodes — there is no `nodeSelector`.
Add one (plus `tolerations`) only if you need to pin the sim to specific nodes.

### Local (CPU, simulated)

#### Prerequisites

> [!IMPORTANT]
> **vLLM must be built from source** — the PyPI package does not support
> CPU-only or macOS environments. See the [vLLM CPU build guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
> for platform-specific instructions.

Once vLLM is installed, install this plugin from the `vllm-simulated-model`
repo root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

The plugin registers itself via the `vllm.general_plugins` entry point; no code
changes to vLLM are needed.

**NIXL installation** — the `NixlConnector` requires the NIXL library for KV
cache transfer. **NIXL is Linux-only** (tested on Ubuntu 22.04/24.04 and Fedora).

**Quick start (NVIDIA GPU on Linux):**
```bash
uv pip install nixl==1.3.2
```

**CPU-only Linux:**
NIXL supports multiple transport backends (UCX with RDMA/TCP, LIBFABRIC, etc.).
The PyPI wheel includes UCX. For CPU-only environments or custom builds, you can
build UCX and NIXL from source from the vLLM repo root (sibling `../vllm`):
```bash
python tools/install_nixl_from_source_ubuntu.py
```

If not running as root, manually install system dependencies first:
```bash
sudo apt-get install -y build-essential git cmake ninja-build \
  autotools-dev automake meson libtool libtool-bin pkg-config patchelf
```

For more details, see the [NIXL usage guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html).

> [!WARNING]
> **macOS limitation:** NIXL does not support macOS. The disaggregated
> prefill/decode setup with `NixlConnector` requires a Linux machine (bare
> metal, VM, or container) or deployment to Kubernetes.

#### Run the simulator

Open three terminals from the vLLM repo root.

**Terminal 1 — prefill:**
```bash
VLLM_SIMULATED_PLUGIN_CONFIG=models/qwen3-32b/deployments/h100-sxm5/pd/evaluation/physics/sim-config.json \
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --load-format dummy \
  --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

**Terminal 2 — decode:**
```bash
VLLM_SIMULATED_PLUGIN_CONFIG=models/qwen3-32b/deployments/h100-sxm5/pd/evaluation/physics/sim-config.json \
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --load-format dummy \
  --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

**Terminal 3 — proxy** (after both vLLM servers are ready):
```bash
python3 examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py \
  --model qwen3-32b \
  --prefill localhost:8100 \
  --decode  localhost:8200 \
  --port 8000
```

The commands above use the calibrated `physics` variant. To use a different
latency model, replace `physics` with `flat` or `physics-beta-1.0` in the
`VLLM_SIMULATED_PLUGIN_CONFIG` path. The `--load-format dummy` flag skips
weight loading; the simulator reads latency parameters from the JSON config.
Send requests to `http://localhost:8000`.

## Latency Models

Configs live under `evaluation/`.

| Directory | Description |
|-----------|-------------|
| [evaluation/flat](evaluation/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [evaluation/physics](evaluation/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [evaluation/physics-beta-1.0](evaluation/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory contains:
- `configmap.yaml` — model architecture + latency params as a Kubernetes ConfigMap
- `sim-config.json` — same config as a plain JSON file for local use
- `prefill-deployment.yaml`, `prefill-service.yaml` — prefill backend manifests
- `decode-deployment.yaml`, `decode-service.yaml` — decode backend manifests
- `proxy-deployment.yaml`, `proxy-service.yaml` — proxy manifests

## Eval Results

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [results/flat](results/flat/) | flat | — |
| [results/physics-beta-1.0](results/physics-beta-1.0/) | physics-beta-1.0 | — |

## Eval (real vs sim comparison)

`eval.sh` handles the whole lifecycle (setup → run → teardown) from the variant
dir; the report lands under `evaluation/results/<variant>/` — commit it:

```bash
bash evaluation/eval.sh \
  models/qwen3-32b/deployments/h100-sxm5/pd/evaluation/physics
```

The script uses `$VLLM_SIM_NAMESPACE` from your environment.

Use `eval.sh tune <variant-dir>` to auto-tune a physics variant's `beta`. See
`evaluation/README.md` for the full lifecycle, subcommands, and env knobs.
