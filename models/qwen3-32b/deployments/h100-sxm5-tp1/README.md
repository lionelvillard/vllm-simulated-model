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

| Directory | Description |
|-----------|-------------|
| [flat](latency/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [physics](latency/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [physics-beta-1.0](latency/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory contains:
- `sim-config.json` — model architecture + latency params, applied to vLLM via `--load-format=dummy`
- `configmap.yaml` — Kubernetes ConfigMap wrapping the above (apply before the sim Deployment)

## Eval Results

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [results/flat](results/flat/) | flat | Initial eval run |
| [results/physics-beta-1.0](results/physics-beta-1.0/) | physics-beta-1.0 | Pre-calibration baseline |

## Deploying

### Standalone sim (no real model)

```bash
# 1. Pick a latency model and apply its ConfigMap:
kubectl apply -f latency/physics/configmap.yaml

# 2. Deploy the sim:
kubectl apply -f k8s/standalone/
```

### Eval (real vs sim comparison)

```bash
# 1. Apply sim ConfigMap (choose latency variant):
kubectl apply -n <ns> -f latency/physics/configmap.yaml

# 2. Deploy real model + sim + services:
kubectl apply -n <ns> -f k8s/eval/

# 3. Run the benchmark sweep from your machine:
NAMESPACE=<ns> bash evaluation/run_eval.sh

# 4. Commit results under results/<latency-variant>/
```

See `evaluation/README.md` for full details and troubleshooting.
