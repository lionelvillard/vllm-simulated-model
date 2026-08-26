# Qwen3-32B — H100 SXM5 Prefill/Decode Disaggregation (TP=2)

Prefill/decode disaggregation with tensor-parallel size 2: each role (prefill and
decode) runs across two H100 GPUs, with KV caches transferred via `NixlConnector`
over UCX. The llm-d sidecar coordinates the two phases. This requires four H100s
total (two per role).

The Kubernetes stack under `k8s/` runs the calibrated **physics** simulator —
CPU pods, dummy weights — so you can exercise disaggregated TP=2 serving without
any H100s. The `evaluation/` directory contains latency-model variants and
benchmark reports; the `physics` variant is the calibration source for `k8s/`.

> [!NOTE]
> The beta parameters in the physics variant are inherited from the TP=1 calibration
> and have not yet been calibrated against a real TP=2 cluster. Run the evaluation
> to produce TP=2-specific betas before using this simulator for latency estimation.

## Deployment

### Pre-built Dependencies Image

The deployment uses a pre-built container image (`ghcr.io/lionelvillard/vllm-sim-deps:v0.3.0`)
that bundles the simulated model plugin and NIXL dependencies. This significantly
speeds up pod startup compared to installing dependencies via pip at runtime.

The image is already built and published. For version compatibility and build
instructions, see [docker/vllm-sim-deps/README.md](../../../../../docker/vllm-sim-deps/README.md).

### Kubernetes (simulated)

`k8s/` runs the disaggregated TP=2 topology as the simulated model on CPU nodes —
no GPUs required.

**One-time setup:**
```bash
export VLLM_SIM_NAMESPACE=default  # or your target namespace
```

Apply the manifests (skip `sim-config.json`; it's the plain-JSON source the
ConfigMap embeds, not a Kubernetes resource):

```bash
K=models/qwen3-32b/deployments/h100-sxm5/pd-tp2/k8s
kubectl apply -n $VLLM_SIM_NAMESPACE \
  -f $K/configmap.yaml \
  -f $K/prefill-deployment.yaml -f $K/prefill-service.yaml \
  -f $K/decode-deployment.yaml  -f $K/decode-service.yaml
```

This deploys:

| Resource | Role | Notes |
|----------|------|-------|
| `vllm-qwen3-32b-pd-tp2-fcb0d4-prefill` | `kv_role: kv_producer` | Sim CPU pod; vLLM on port 8000; NIXL side-channel on port 5600 |
| `vllm-qwen3-32b-pd-tp2-fcb0d4-decode` | `kv_role: kv_consumer` + sidecar | Sim CPU pod; **sidecar on port 8000** (client-facing); vLLM on port 8200 (internal); NIXL on port 5601 |

Both backends run the sim plugin (`vllm/vllm-openai-cpu` image, `--load-format
dummy`) and mount the physics ConfigMap at `/model`. The 6-char hash `fcb0d4`
is derived from the TP=2 physics latency config, matching the ConfigMap and
`vllm-qwen3-32b-pd-tp2-<hash>[-<role>]` naming scheme. Both deployments include
an init container (`ghcr.io/lionelvillard/vllm-sim-deps:v0.3.0`) that provides
pre-built dependencies: the simulated model plugin and NIXL for KV cache transfer.

The simulated backends are started with `--tensor-parallel-size 2`, which spawns
two CPU worker processes per pod communicating via gloo. The plugin reads the actual
`tensor_parallel_size` from vLLM's parallel config and uses it directly in the
physics formula — no separate `"tp"` field in the JSON config is needed.

The decode deployment includes the [llm-d](https://github.com/llm-d/llm-d-router)
disaggregation sidecar (`ghcr.io/llm-d/llm-d-router-disagg-sidecar`) running
alongside the vLLM decode worker. The sidecar is configured with
`--kv-connector=nixlv2` to match the vLLM `NixlConnector` used by both backends.

**Port architecture** (following llm-d reference configuration):
- **Sidecar (port 8000)**: Client-facing endpoint, the only port exposed via Service
- **vLLM decode worker (port 8200)**: Internal only, accessed by sidecar via localhost
- **NIXL side-channel (port 5601)**: KV cache transfer from prefill worker

Send all client traffic to `vllm-qwen3-32b-pd-tp2-fcb0d4-decode:8000`. The sidecar
forwards the prefill phase to `vllm-qwen3-32b-pd-tp2-fcb0d4-prefill:8000`, coordinates
the KV cache transfer via NIXL, then sends the decode request to the local vLLM
worker on port 8200 for streaming generation.

**HF token:** Both backends read `HF_TOKEN` from a Secret named `hf-token` (key:
`token`) — needed only if the tokenizer (`Qwen/Qwen3-32B`) is gated; the sim
uses dummy weights, so no model download occurs. Create it before applying:

```bash
kubectl create secret generic hf-token -n $VLLM_SIM_NAMESPACE --from-literal=token=hf_...
```

> [!TIP]
> The secret reference is `optional: true`, so pods start without it if the
> tokenizer is already cached or publicly accessible.

**Readiness:** The sim starts quickly — no ~64 GB weight download. The prefill
backend's `startupProbe` allows up to ~150 s (plugin install + CPU engine init).
The decode pod runs two containers with independent health checks:
- **Sidecar**: `/health` on port 8000 (5s initial delay)
- **vLLM worker**: `/health` on port 8200 (up to ~150s startup probe)

**Node placement:** The simulated backends request `8` CPU / `16Gi` memory each
(two worker processes per pod for TP=2) and no GPU, so they schedule on ordinary
CPU nodes — there is no `nodeSelector`. Add one (plus `tolerations`) only if you
need to pin the sim to specific nodes.

#### Verify deployment

```bash
# Forward decode sidecar port to localhost
kubectl port-forward -n $VLLM_SIM_NAMESPACE \
  svc/vllm-qwen3-32b-pd-tp2-fcb0d4-decode 8000:8000

# In another terminal:
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion with P/D disaggregation
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-prefiller-host-port: vllm-qwen3-32b-pd-tp2-fcb0d4-prefill:8000" \
  -d '{
    "model": "qwen3-32b",
    "messages": [
      {"role": "user", "content": "Tell me a short story about a robot."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

> [!NOTE]
> The `x-prefiller-host-port` header explicitly triggers P/D disaggregation. The
> sidecar routes the request to the prefill worker for prompt processing, coordinates
> KV cache transfer via NIXL, then sends the decode request to the local vLLM worker
> for streaming generation.

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

> [!NOTE]
> This deployment uses `SimulatedNixlConnector`, which does **not** require
> NIXL, UCX, or RDMA hardware. It simulates KV transfer latency using a
> bandwidth model (default 100 Gbps) and runs on any platform, including macOS.
> No actual KV data is transferred.

#### Run the simulator

Open three terminals from the vLLM repo root.

**Terminal 1 — prefill:**
```bash
VLLM_SIMULATED_PLUGIN_CONFIG=models/qwen3-32b/deployments/h100-sxm5/pd-tp2/evaluation/physics/sim-config.json \
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --load-format dummy \
  --tensor-parallel-size 2 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"SimulatedNixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"bandwidth_gbps":100,"handshake_ms":2}}'
```

**Terminal 2 — decode:**
```bash
VLLM_SIMULATED_PLUGIN_CONFIG=models/qwen3-32b/deployments/h100-sxm5/pd-tp2/evaluation/physics/sim-config.json \
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --load-format dummy \
  --tensor-parallel-size 2 \
  --port 8200 \
  --kv-transfer-config '{"kv_connector":"SimulatedNixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"bandwidth_gbps":100,"handshake_ms":2}}'
```

**Terminal 3 — coordination** (after both vLLM servers are ready):

```bash
python3 examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py \
  --model qwen3-32b \
  --prefill localhost:8000 \
  --decode  localhost:8200 \
  --port 8080
```

> [!NOTE]
> The Kubernetes deployment uses the llm-d sidecar instead of vLLM's proxy script.
> For local testing with llm-d, you would need to build and run the `pd-sidecar`
> binary from [llm-d-router](https://github.com/llm-d/llm-d-router).

The commands above use the calibrated `physics` variant. To use a different
latency model, replace `physics` with `flat` or `physics-beta-1.0` in the
`VLLM_SIMULATED_PLUGIN_CONFIG` path. Send requests to `http://localhost:8080`.

## Latency Models

Configs live under `evaluation/`.

| Directory | Description |
|-----------|-------------|
| [evaluation/flat](evaluation/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) — parameters not yet calibrated for TP=2 |
| [evaluation/physics](evaluation/physics/) | Roofline physics model, β = [0.15, 0.0, 126.0], `tp: 2` — betas inherited from TP=1; recalibrate after first eval run |
| [evaluation/physics-beta-1.0](evaluation/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory contains:
- `configmap.yaml` — model architecture + latency params as a Kubernetes ConfigMap
- `sim-config.json` — same config as a plain JSON file for local use
- `real-prefill-deployment.yaml`, `real-prefill-service.yaml` — real TP=2 prefill deployment for eval
- `real-decode-deployment.yaml`, `real-service.yaml` — real TP=2 decode deployment for eval
- `sim-prefill-deployment.yaml`, `sim-prefill-service.yaml` — sim prefill Deployment and Service
- `sim-decode-deployment.yaml`, `sim-service.yaml` — sim decode Deployment and Service
- `benchmark-job.yaml` — in-cluster benchmark Job for real-vs-sim comparison

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
  models/qwen3-32b/deployments/h100-sxm5/pd-tp2/evaluation/physics
```

The script uses `$VLLM_SIM_NAMESPACE` from your environment.

Use `eval.sh tune <variant-dir>` to auto-tune a physics variant's `beta`. See
`evaluation/README.md` for the full lifecycle, subcommands, and env knobs.

> [!IMPORTANT]
> Real deployments require two H100 GPUs per pod (`nvidia.com/gpu: 2`). Each
> prefill and decode pod requests 2 GPUs, so 4 H100s total are needed for a full
> eval run.
