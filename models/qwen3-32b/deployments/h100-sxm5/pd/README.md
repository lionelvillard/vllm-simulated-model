# Qwen3-32B — H100 SXM5 Prefill/Decode Disaggregation

Two separate H100 SXM5 instances connected via `NixlConnector`. The prefill instance processes prompt tokens and transfers KV caches to the decode instance over UCX. A lightweight proxy coordinates the two phases.

**Prerequisites:** 2 × H100 SXM5 GPUs; vLLM installed with Nixl support (`pip install vllm[nixl]`).

## Local (two GPUs on one host)

Open three terminals from the vLLM repo root.

**Terminal 1 — prefill:**
```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
```

**Terminal 2 — decode:**
```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
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

Send requests to `http://localhost:8000`.

## Latency Models

Configs live under `latency/`.

| Directory | Description |
|-----------|-------------|
| [latency/flat](latency/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [latency/physics](latency/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [latency/physics-beta-1.0](latency/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory contains:
- `configmap.yaml` — model architecture + latency params as a Kubernetes ConfigMap
- `sim-config.json` — same config as a plain JSON file for local use

Resource names follow the scheme `vllm-qwen3-32b-pd-<hash>[-<role>]` where `<hash>` is a
6-char SHA-256 of the latency config. This ensures simultaneous deployments of different variants
never conflict.

## Eval Results

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [results/flat](results/flat/) | flat | — |
| [results/physics-beta-1.0](results/physics-beta-1.0/) | physics-beta-1.0 | — |

## Kubernetes

```bash
kubectl apply -n <ns> -f k8s/
```

This deploys:

| Resource | Role | Notes |
|----------|------|-------|
| `vllm-qwen3-32b-pd-525604-prefill` | `kv_role: kv_producer` | Side-channel on port 5600; `VLLM_NIXL_SIDE_CHANNEL_HOST` set to pod IP |
| `vllm-qwen3-32b-pd-525604-decode` | `kv_role: kv_consumer` | Side-channel on port 5601; `VLLM_NIXL_SIDE_CHANNEL_HOST` set to pod IP |
| `vllm-qwen3-32b-pd-525604-proxy` | Request router | Init containers wait for both pods before the proxy starts |

Send all client traffic to `vllm-qwen3-32b-pd-525604-proxy:8000`. The proxy forwards the prefill phase (`max_tokens=1`) to `vllm-qwen3-32b-pd-525604-prefill`, then sends the full request to `vllm-qwen3-32b-pd-525604-decode` for streaming generation.

### Eval (real vs sim comparison)

```bash
# 1. Apply the ConfigMap for the chosen latency variant:
kubectl apply -n <ns> -f latency/physics/configmap.yaml

# 2. Run the benchmark sweep from your machine:
NAMESPACE=<ns> bash evaluation/run_eval.sh

# 3. Commit results under results/<latency-variant>/
```

See `evaluation/README.md` for full details and troubleshooting.

### Node selection

Both `vllm-qwen3-32b-pd-525604-prefill` and `vllm-qwen3-32b-pd-525604-decode` use `nodeSelector: nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3`. Adjust this label (or add `tolerations`) to match your cluster's GPU nodes.

### HF token

Both GPU deployments read `HF_TOKEN` from a Secret named `hf-token` (key: `token`). Create it before applying:

```bash
kubectl create secret generic hf-token -n <ns> --from-literal=token=<your-token>
```

The secret reference is `optional: true`, so pods start without it if the model is already cached or publicly accessible.

### Readiness

Both prefill and decode pods have a 600 s `initialDelaySeconds` to allow weight download (~64 GB for Qwen3-32B). The proxy's init containers poll `/health` on each backend every 5 s and block until both respond.
