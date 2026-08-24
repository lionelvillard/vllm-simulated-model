# Qwen3-32B — H100 SXM5 Standalone

Single-instance deployment: one H100 SXM5 serving the full model.

## Layout

- `k8s/` — the tuned, deployable simulator. Holds the calibrated **physics**
  stack: `configmap.yaml` (+ its plain-JSON `sim-config.json`), `deployment.yaml`,
  `service.yaml`, and the shared `hf-cache` `pvc.yaml`. This is what you deploy.
- `evaluation/` — everything for evaluating/calibrating the sim: latency-model
  variants (`flat/`, `physics/`, `physics-beta-1.0/`), the benchmark matrix
  (`sweep.yaml`), and eval reports (`results/`). The `physics` variant is the
  calibration source for `k8s/`.

## Latency Models

Configs live under `evaluation/`.

| Directory | Description |
|-----------|-------------|
| [evaluation/flat](evaluation/flat/) | Empirical flat model (`base_ms` + per-token/per-seq terms) tuned by hand |
| [evaluation/physics](evaluation/physics/) | Roofline physics model, calibrated betas β = [0.152, 0.0, 126.0] |
| [evaluation/physics-beta-1.0](evaluation/physics-beta-1.0/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated baseline |

Each latency directory is a self-contained deployable stack:
- `configmap.yaml` — model architecture + latency params as a Kubernetes ConfigMap
- `deployment.yaml` — sim Deployment (references this variant's ConfigMap and the shared PVC)
- `service.yaml` — ClusterIP Service for the sim
- `eval/` — real + sim Deployments/Services and benchmark Job for a real-vs-sim comparison

Resource names follow the scheme `vllm-qwen3-32b-standalone-<hash>[-<role>]` where `<hash>`
is a 6-char SHA-256 of the latency config. This ensures simultaneous deployments of different
variants never conflict.

## Eval Results

Reports live under `evaluation/results/<variant>/`, mirroring the config tree.

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [evaluation/results/flat](evaluation/results/flat/) | flat | Initial eval run |
| [evaluation/results/physics-beta-1.0](evaluation/results/physics-beta-1.0/) | physics-beta-1.0 | Pre-calibration baseline |

## Deploy

### Shared prerequisite (once per cluster/namespace)

```bash
kubectl apply -f k8s/pvc.yaml
```

### Tuned sim

`k8s/` is the calibrated physics deployment. Apply the manifests (skip
`sim-config.json` — it's the plain-JSON source the ConfigMap embeds, not a
Kubernetes resource):

```bash
kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
```

To deploy a different latency variant instead, point at its ConfigMap under
`evaluation/<variant>/` and reuse `k8s/deployment.yaml` / `k8s/service.yaml`.

### Eval (real vs sim comparison)

`eval.sh` handles the whole lifecycle — it deploys the real + sim stack,
runs the sweep, and tears down. Just pass the variant dir:

```bash
NAMESPACE=<ns> bash evaluation/eval.sh \
  models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/physics
```

The report is written to `evaluation/results/physics/` — commit it. The
real Deployment is left running for reuse; run `... teardown <variant-dir>` to
remove it. Use the `setup` / `run` / `teardown` subcommands to iterate without
redeploying, or `eval.sh tune <variant-dir>` to auto-tune a physics variant's
`beta`. See `evaluation/README.md` for the full lifecycle and env knobs.
