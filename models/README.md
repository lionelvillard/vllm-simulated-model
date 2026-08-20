# Models

Deployment configurations for simulated vLLM models. Each model directory contains the real HuggingFace config plus one or more deployment variants, each with latency model configurations and evaluation results.

## Layout

```
models/
  <model>/                          # e.g. qwen3-32b, qwen3-a22b
    config.json                     # Real model config from HuggingFace (no latency block)
    README.md                       # Model overview and deployment inventory
    deployments/
      <hardware>-tp<n>/             # e.g. h100-sxm5-tp1, cpu-tp4
        sweep.yaml                  # Benchmark matrix for this deployment
        latency/
          flat/                     # Empirical flat latency model
          physics/                  # Physics roofline model (calibrated)
          physics-beta-1.0/         # Physics model, unit betas — uncalibrated baseline
            sim-config.json         # Model arch + latency params (applied to vLLM)
            configmap.yaml          # Kubernetes ConfigMap wrapping sim-config.json
        k8s/
          standalone/               # Deploy only the sim model
            deployment.yaml
            service.yaml
            pvc.yaml
          eval/                     # Deploy real + sim side by side for comparison
            real-deployment.yaml
            real-service.yaml
            sim-deployment.yaml
            sim-service.yaml
            benchmark-job.yaml
        results/
          <latency>/                # Eval results, named after the latency variant used
            report.md
            report.json
            warmup.json
            c<N>-isl<X>-osl<Y>-{real,sim}.json
```

## Hardware Slugs

| Slug | GPU | Peak TFLOPs (BF16) | HBM bandwidth |
|------|-----|--------------------|---------------|
| `h100-sxm5` | NVIDIA H100 SXM5 80 GB | 989 | 3350 GB/s |
| `h100-pcie` | NVIDIA H100 PCIe 80 GB | 312 | 2000 GB/s |
| `cpu` | x86 CPU (vLLM CPU backend) | — | — |

## Models

| Directory | Model | Parameters | Architecture |
|-----------|-------|------------|--------------|
| [qwen3-32b](qwen3-32b/) | Qwen/Qwen3-32B | 32B | Dense |
| [qwen3-a22b](qwen3-a22b/) | Qwen/Qwen3-235B-A22B | 235B total / 22B active | MoE |

## Adding a New Model

1. Download the real config from HuggingFace:
   ```bash
   huggingface-cli download <HF-id> config.json --local-dir models/<slug>/
   ```
2. Write `models/<slug>/README.md` with model overview and deployment table.
3. Add a deployment following the steps below.

## Adding a New Deployment

1. Create the deployment directory: `models/<model>/deployments/<hardware>-tp<n>/`.
2. Copy `sweep.yaml` from a sibling deployment and adjust concurrency limits for the new hardware.
3. Generate physics latency configs:
   ```bash
   python -m evaluation.gen_sim_config \
     --model models/<model>/config.json \
     --out models/<model>/deployments/<dep>/latency/physics/configmap.yaml \
     --tp <n> --peak-tflops <X> --hbm-gbps <Y>
   ```
   The tool also extracts the embedded JSON to `latency/physics/sim-config.json` if you pass `--sim-config-out`.
4. Copy K8s manifests from a sibling deployment's `k8s/` and update image, model name, and resource requests.
5. Write `k8s/standalone/` manifests for standalone sim deployment (no real model).
6. Run the evaluation: `NAMESPACE=<ns> bash evaluation/run_eval.sh` and commit results under `results/<latency>/`.
