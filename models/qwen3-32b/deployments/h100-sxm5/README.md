# Qwen3-32B — H100 SXM5 TP=1

Single H100 SXM5 (80 GB HBM3), tensor-parallel size 1.

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 80 GB |
| Peak TFLOPs (BF16) | 989 |
| HBM bandwidth | 3350 GB/s |
| Tensor parallel | 1 |

## Configurations

| Directory | Description |
|-----------|-------------|
| [standalone](standalone/README.md) | Single GPU serving — latency models, eval, and deployment |
| [pd](pd/README.md) | Prefill/decode disaggregation across two H100s |
