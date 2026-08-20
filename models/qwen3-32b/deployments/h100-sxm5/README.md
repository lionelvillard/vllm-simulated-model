# Qwen3-32B — H100 SXM5 TP=1

Single H100 SXM5 (80 GB HBM3), tensor-parallel size 1.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 80 GB |
| Peak TFLOPs (BF16) | 989 |
| HBM bandwidth | 3350 GB/s |
| Tensor parallel | 1 |

## Latency Models

Latency configs live under `standalone/latency/`.

| Directory | Description |
|-----------|-------------|
| [standalone/latency/flat](standalone/latency/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [standalone/latency/physics](standalone/latency/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [standalone/latency/physics-beta-1.0](standalone/latency/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory contains:
- `sim-config.json` — model architecture + latency params, applied to vLLM via `--load-format=dummy`
- `configmap.yaml` — Kubernetes ConfigMap wrapping the above (apply before the sim Deployment)

## Eval Results

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [standalone/results/flat](standalone/results/flat/) | flat | Initial eval run |
| [standalone/results/physics-beta-1.0](standalone/results/physics-beta-1.0/) | physics-beta-1.0 | Pre-calibration baseline |

## Deploying

### Standalone sim (no real model)

```bash
# 1. Pick a latency model and apply its ConfigMap:
kubectl apply -f standalone/latency/physics/configmap.yaml

# 2. Deploy the sim:
kubectl apply -f standalone/k8s/
```

### Eval (real vs sim comparison)

```bash
# 1. Apply sim ConfigMap (choose latency variant):
kubectl apply -n <ns> -f standalone/latency/physics/configmap.yaml

# 2. Deploy real model + sim + services:
kubectl apply -n <ns> -f standalone/k8s/eval/

# 3. Run the benchmark sweep from your machine:
NAMESPACE=<ns> bash evaluation/run_eval.sh

# 4. Commit results under standalone/results/<latency-variant>/
```

See `evaluation/README.md` for full details and troubleshooting.

### P/D disaggregation (prefill/decode split)

Two separate GPU instances connected via `NixlConnector`. The prefill instance processes prompt tokens and transfers KV caches to the decode instance over UCX. A lightweight proxy coordinates the two phases.

**Prerequisites:** 2 × H100 SXM5 GPUs; vLLM installed with Nixl support (`pip install vllm[nixl]`).

#### Local (two GPUs on one host)

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

#### Kubernetes

```bash
kubectl apply -n <ns> -f pd/k8s/
```

This deploys:
- `vllm-prefill` — `kv_role: kv_producer`, side-channel on port 5600
- `vllm-decode`  — `kv_role: kv_consumer`, side-channel on port 5601
- `vllm-pd-proxy` — runs `disagg_proxy_demo.py` from `/vllm-workspace/examples/`; init containers wait for both pods before the proxy starts

Send all client traffic to `vllm-pd-proxy:8000`. The proxy forwards the prefill phase (with `max_tokens=1`) to `vllm-prefill`, then sends the full request to `vllm-decode` for streaming generation.