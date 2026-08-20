# Qwen3-235B-A22B — H100 PCIe TP=1

Single H100 PCIe (80 GB), tensor-parallel size 1.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 PCIe 80 GB |
| Peak TFLOPs (BF16) | 312 |
| HBM bandwidth | 2000 GB/s |
| Tensor parallel | 1 |

## Latency Models

| Directory | Description |
|-----------|-------------|
| [physics](latency/physics/) | Roofline physics model, unit betas β = [1.0, 1.0, 0.0] — uncalibrated |

## Status

No K8s manifests or eval results yet. To add them:
1. Adapt the manifests from [qwen3-32b/deployments/h100-sxm5-tp1/k8s](../../qwen3-32b/deployments/h100-sxm5-tp1/k8s/).
2. Update `real-deployment.yaml` to use `Qwen/Qwen3-235B-A22B` and request the correct GPU type.
3. Run the eval and commit results under `results/physics/`.
4. Calibrate betas and create a `latency/physics/` variant with the tuned values.
