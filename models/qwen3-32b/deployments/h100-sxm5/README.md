# Qwen3-32B — H100 SXM5

NVIDIA H100 SXM5 (80 GB HBM3) deployments.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 80 GB |
| Peak TFLOPs (BF16) | 989 |
| HBM bandwidth | 3350 GB/s |

## Configurations

| Directory | Description | GPUs |
|-----------|-------------|------|
| [standalone](standalone/README.md) | Single GPU serving — latency models, eval, and deployment | 1 |
| [pd](pd/README.md) | Prefill/decode disaggregation, TP=1 per role (2 GPUs total) | 2 |
| [pd-tp2](pd-tp2/README.md) | Prefill/decode disaggregation, TP=2 per role (4 GPUs total) | 4 |
