# Qwen3-32B — H100 SXM5 Standalone

Single-instance deployment: one H100 SXM5 serving the full model.

## Layout

- `k8s/` — shared cluster prerequisites (the `hf-cache` PVC).
- `evaluation/` — everything for evaluating the sim: latency-model variants
  (`latency/`), the benchmark matrix (`sweep.yaml`), and eval reports (`results/`).

## Latency Models

Configs live under `evaluation/latency/`.

| Directory | Description |
|-----------|-------------|
| [evaluation/latency/flat](evaluation/latency/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [evaluation/latency/physics](evaluation/latency/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [evaluation/latency/physics-beta-1.0](evaluation/latency/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory is a self-contained deployable stack:
- `configmap.yaml` — model architecture + latency params as a Kubernetes ConfigMap
- `deployment.yaml` — sim Deployment (references this variant's ConfigMap and the shared PVC)
- `service.yaml` — ClusterIP Service for the sim
- `eval/` — real + sim Deployments/Services and benchmark Job for a real-vs-sim comparison

Resource names follow the scheme `vllm-qwen3-32b-standalone-<hash>[-<role>]` where `<hash>`
is a 6-char SHA-256 of the latency config. This ensures simultaneous deployments of different
variants never conflict.

## Eval Results

Reports live under `evaluation/results/<category>/<variant>/`, mirroring the config tree.

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [evaluation/results/latency/flat](evaluation/results/latency/flat/) | flat | Initial eval run |
| [evaluation/results/latency/physics-beta-1.0](evaluation/results/latency/physics-beta-1.0/) | physics-beta-1.0 | Pre-calibration baseline |

## Deploy

### Shared prerequisite (once per cluster/namespace)

```bash
kubectl apply -f k8s/pvc.yaml
```

### Sim only

```bash
# Apply the full variant stack in one shot:
kubectl apply -f evaluation/latency/physics/
```

### Eval (real vs sim comparison)

```bash
# 1. Apply the ConfigMap for the chosen latency variant:
kubectl apply -n <ns> -f evaluation/latency/physics/configmap.yaml

# 2. Deploy real model + sim (Deployments and Services only):
kubectl apply -n <ns> -f evaluation/latency/physics/eval/sim-deployment.yaml
kubectl apply -n <ns> -f evaluation/latency/physics/eval/sim-service.yaml
kubectl apply -n <ns> -f evaluation/latency/physics/eval/real-deployment.yaml
kubectl apply -n <ns> -f evaluation/latency/physics/eval/real-service.yaml

# 3. Run the benchmark sweep from your machine (pass the variant dir):
NAMESPACE=<ns> bash evaluation/run_eval.sh \
  models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics

# 4. The report is written to evaluation/results/latency/physics/ — commit it.
```

See `evaluation/README.md` for full details and troubleshooting.
