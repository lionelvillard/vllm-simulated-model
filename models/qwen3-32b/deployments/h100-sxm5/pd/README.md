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

## Kubernetes

```bash
kubectl apply -n <ns> -f k8s/
```

This deploys:

| Resource | Role | Notes |
|----------|------|-------|
| `vllm-prefill` | `kv_role: kv_producer` | Side-channel on port 5600; `VLLM_NIXL_SIDE_CHANNEL_HOST` set to pod IP |
| `vllm-decode` | `kv_role: kv_consumer` | Side-channel on port 5601; `VLLM_NIXL_SIDE_CHANNEL_HOST` set to pod IP |
| `vllm-pd-proxy` | Request router | Init containers wait for both pods before the proxy starts |

Send all client traffic to `vllm-pd-proxy:8000`. The proxy forwards the prefill phase (`max_tokens=1`) to `vllm-prefill`, then sends the full request to `vllm-decode` for streaming generation.

### Node selection

Both `vllm-prefill` and `vllm-decode` use `nodeSelector: nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3`. Adjust this label (or add `tolerations`) to match your cluster's GPU nodes.

### HF token

Both GPU deployments read `HF_TOKEN` from a Secret named `hf-token` (key: `token`). Create it before applying:

```bash
kubectl create secret generic hf-token -n <ns> --from-literal=token=<your-token>
```

The secret reference is `optional: true`, so pods start without it if the model is already cached or publicly accessible.

### Readiness

Both prefill and decode pods have a 600 s `initialDelaySeconds` to allow weight download (~64 GB for Qwen3-32B). The proxy's init containers poll `/health` on each backend every 5 s and block until both respond.
