# Models

Deployment configurations for simulated vLLM models. Each model directory contains the real HuggingFace config plus one or more deployment variants, each with latency model configurations and evaluation results.

## Layout

```
models/
  <model>/                          # e.g. qwen3-32b
    config.json                     # Real model config from HuggingFace (no latency block)
    README.md                       # Model overview and deployment inventory
    deployments/
      README.md                     # Deployment inventory for this model
      <hardware>/                   # e.g. h100-sxm5, cpu
        README.md                   # Hardware overview, latency table, deploy instructions
        standalone/                 # Single-instance (non-disaggregated) deployment
          k8s/                      # Tuned, deployable simulator (calibrated physics)
            pvc.yaml                # Shared hf-cache PVC (cluster prerequisite)
            configmap.yaml          # Physics ConfigMap (model arch + latency params)
            sim-config.json         # Plain-JSON copy of the ConfigMap's config.json
            deployment.yaml         # Sim Deployment
            service.yaml            # Sim ClusterIP Service
          evaluation/               # Everything needed to evaluate the sim
            sweep.yaml              # Benchmark matrix for this deployment
            flat/                   # Empirical flat latency model
            physics/                # Physics roofline model (calibrated)
            physics-beta-1.0/       # Physics model, unit betas — uncalibrated baseline
              sim-config.json       # Model arch + latency params (plain JSON)
              configmap.yaml        # Kubernetes ConfigMap wrapping sim-config.json
              real-deployment.yaml  # Real model deployment for eval
              real-service.yaml
              sim-deployment.yaml   # Sim Deployment for this variant
              sim-service.yaml      # Sim ClusterIP Service
              benchmark-job.yaml    # In-cluster benchmark Job
            results/
              <variant>/            # Eval report, named after the latency variant used
                report.md
                report.json
                warmup.json
                c<N>-isl<X>-osl<Y>-{real,sim}.json
        pd/                         # Prefill/decode disaggregated deployment (simulated, CPU)
          k8s/                      # Tuned, deployable simulated P/D (calibrated physics)
            configmap.yaml          # Physics ConfigMap (model arch + latency params)
            sim-config.json         # Plain-JSON copy of the ConfigMap's config.json
            prefill-deployment.yaml # Sim prefill pod (NixlConnector kv_producer)
            prefill-service.yaml
            decode-deployment.yaml  # Sim decode pod (NixlConnector kv_consumer + llm-d sidecar)
            decode-service.yaml
          evaluation/               # Everything needed to evaluate the sim
            sweep.yaml              # Benchmark matrix for this deployment
            flat/                   # Empirical flat latency model
            physics/                # Physics roofline model (calibrated)
            physics-beta-1.0/       # Physics model, unit betas — uncalibrated baseline
              sim-config.json       # Model arch + latency params (plain JSON)
              configmap.yaml        # Kubernetes ConfigMap wrapping sim-config.json
              real-deployment.yaml  # Real model deployment for eval
              real-service.yaml
              sim-deployment.yaml   # Sim Deployment for this variant
              sim-service.yaml      # Sim ClusterIP Service
              benchmark-job.yaml    # In-cluster benchmark Job
            results/
              <variant>/
                report.md
                report.json
                warmup.json
                c<N>-isl<X>-osl<Y>-{real,sim}.json
```

## Hardware Slugs

The deployment directory is named after the hardware only (no TP suffix — tensor-parallel size is documented inside the deployment's `README.md`).

| Slug | GPU | Peak TFLOPs (BF16) | HBM bandwidth |
|------|-----|--------------------|---------------|
| `h100-sxm5` | NVIDIA H100 SXM5 80 GB | 989 | 3350 GB/s |
| `h100-pcie` | NVIDIA H100 PCIe 80 GB | 756 | 2000 GB/s |
| `cpu` | x86 CPU (vLLM CPU backend) | — | — |

## Models

| Directory | Model | Parameters | Architecture |
|-----------|-------|------------|--------------|
| [qwen3-32b](qwen3-32b/) | Qwen/Qwen3-32B | 32B | Dense |

## Adding a New Model

1. Download the real config from HuggingFace:
   ```bash
   huggingface-cli download <HF-id> config.json --local-dir models/<slug>/
   ```
2. Write `models/<slug>/README.md` with model overview and deployment table.
3. Add a deployment following the steps below.

## Adding a New Deployment

1. Create the deployment directory: `models/<model>/deployments/<hardware>/`.
2. Copy `standalone/evaluation/sweep.yaml` from a sibling deployment and adjust concurrency limits for the new hardware.
3. Generate physics latency configs:
   ```bash
   python -m evaluation.gen_sim_config \
     --model models/<model>/config.json \
     --out models/<model>/deployments/<hardware>/standalone/evaluation/physics/configmap.yaml \
     --tp <n> --peak-tflops <X> --hbm-gbps <Y>
   ```
   Then extract the plain-JSON `sim-config.json` sidecar (the auto-tuner reads it — see `evaluation/README.md`):
   ```bash
   .venv/bin/python -c 'import sys,yaml; cm=yaml.safe_load(open(sys.argv[1])); open(sys.argv[2],"w").write(cm["data"]["config.json"])' \
     models/<model>/deployments/<hardware>/standalone/evaluation/physics/configmap.yaml \
     models/<model>/deployments/<hardware>/standalone/evaluation/physics/sim-config.json
   ```
4. Copy K8s manifests from a sibling deployment's `standalone/k8s/` and `standalone/evaluation/<variant>/` and update image, model name, and resource requests.
5. Run the evaluation by passing the variant dir: `NAMESPACE=<ns> bash evaluation/eval.sh models/<model>/deployments/<hardware>/standalone/evaluation/physics`. The report lands under `standalone/evaluation/results/physics/` — commit it. (Use `eval.sh tune <variant-dir>` to auto-tune a physics variant's `beta`.)
6. Optionally add a `pd/` subdirectory with prefill/decode disaggregated manifests (see `h100-sxm5/pd/` for reference).
