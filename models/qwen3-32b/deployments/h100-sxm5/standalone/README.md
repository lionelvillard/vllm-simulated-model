# Qwen3-32B — H100 SXM5 Standalone

Single-instance deployment: one H100 SXM5 serving the full model.

The Kubernetes stack under `k8s/` runs the calibrated **physics** simulator on
CPU nodes — no GPUs required. The `evaluation/` directory contains latency-model
variants and benchmark reports; the `physics` variant is the calibration source
for `k8s/`.

## Deployment

### Kubernetes (simulated)

`k8s/` runs the calibrated physics simulator on CPU nodes — no GPUs required.

**One-time setup:**
```bash
export VLLM_SIM_NAMESPACE=default  # or your target namespace
```

Apply the manifests (skip `sim-config.json` — it's the plain-JSON source the
ConfigMap embeds, not a Kubernetes resource):

```bash
# Shared prerequisite (once per cluster/namespace)
kubectl apply -n $VLLM_SIM_NAMESPACE -f k8s/pvc.yaml

# Tuned simulator
kubectl apply -n $VLLM_SIM_NAMESPACE -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
```

To deploy a different latency variant instead, point at its ConfigMap under
`evaluation/<variant>/` and reuse `k8s/deployment.yaml` / `k8s/service.yaml`.

Resource names follow the scheme `vllm-qwen3-32b-standalone-<hash>` where
`<hash>` is a 6-char SHA-256 of the latency config, ensuring simultaneous
deployments of different variants never conflict.

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

#### Run the simulator

From the vLLM repo root:

```bash
VLLM_SIMULATED_PLUGIN_CONFIG=models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/physics/sim-config.json \
vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --load-format dummy \
  --port 8000
```

The command above uses the calibrated `physics` variant. To use a different
latency model, replace `physics` with `flat` or `physics-beta-1.0` in the
`VLLM_SIMULATED_PLUGIN_CONFIG` path. The `--load-format dummy` flag skips
weight loading; the simulator reads latency parameters from the JSON config.

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
- `deployment.yaml` — sim Deployment (references this variant's ConfigMap and the shared PVC)
- `service.yaml` — ClusterIP Service for the sim
- `eval/` — real + sim Deployments/Services and benchmark Job for a real-vs-sim comparison

## Eval Results

Reports live under `evaluation/results/<variant>/`, mirroring the config tree.

| Directory | Latency model used | Notes |
|-----------|--------------------|-------|
| [evaluation/results/flat](evaluation/results/flat/) | flat | Initial eval run |
| [evaluation/results/physics-beta-1.0](evaluation/results/physics-beta-1.0/) | physics-beta-1.0 | Pre-calibration baseline |

## Eval (real vs sim comparison)

`eval.sh` handles the whole lifecycle (setup → run → teardown) from the variant
dir; the report lands under `evaluation/results/<variant>/` — commit it:

```bash
bash evaluation/eval.sh \
  models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/physics
```

The script uses `$VLLM_SIM_NAMESPACE` from your environment.

Use `eval.sh tune <variant-dir>` to auto-tune a physics variant's `beta`. See
`evaluation/README.md` for the full lifecycle, subcommands, and env knobs.
