# Qwen3-32B — H100 SXM5 Standalone

Single-instance deployment: one H100 SXM5 serving the full model.

## Latency Models

Configs live under `latency/`.

| Directory | Description |
|-----------|-------------|
| [latency/flat](latency/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [latency/physics](latency/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [latency/physics-beta-1.0](latency/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory is a self-contained deployable stack:
- `configmap.yaml` — model architecture + latency params as a Kubernetes ConfigMap
- `deployment.yaml` — sim Deployment (references this variant's ConfigMap and the shared PVC)
- `service.yaml` — ClusterIP Service for the sim
- `eval/` — real + sim Deployments/Services and benchmark Job for a real-vs-sim comparison

Resource names follow the scheme `vllm-qwen3-32b-standalone-<hash>[-<role>]` where `<hash>`
is a 6-char SHA-256 of the latency config. This ensures simultaneous deployments of different
variants never conflict.

## Eval Results

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [results/flat](results/flat/) | flat | Initial eval run |
| [results/physics-beta-1.0](results/physics-beta-1.0/) | physics-beta-1.0 | Pre-calibration baseline |

## Deploy

### Shared prerequisite (once per cluster/namespace)

```bash
kubectl apply -f k8s/pvc.yaml
```

### Sim only

```bash
# Apply the full variant stack in one shot:
kubectl apply -f latency/physics/
```

### Eval (real vs sim comparison)

```bash
# 1. Apply the ConfigMap for the chosen latency variant:
kubectl apply -n <ns> -f latency/physics/configmap.yaml

# 2. Deploy real model + sim (Deployments and Services only):
kubectl apply -n <ns> -f latency/physics/eval/sim-deployment.yaml
kubectl apply -n <ns> -f latency/physics/eval/sim-service.yaml
kubectl apply -n <ns> -f latency/physics/eval/real-deployment.yaml
kubectl apply -n <ns> -f latency/physics/eval/real-service.yaml

# 3. Run the benchmark sweep from your machine:
NAMESPACE=<ns> bash evaluation/run_eval.sh

# 4. Commit results under results/<latency-variant>/
```

See `evaluation/README.md` for full details and troubleshooting.
