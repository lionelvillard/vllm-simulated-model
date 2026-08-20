# Qwen3-235B-A22B

Mixture-of-Experts model, 235 B total parameters / 22 B active per token.

- **HuggingFace**: `Qwen/Qwen3-235B-A22B`
- **Architecture**: `Qwen3MoeForCausalLM`
- **Total parameters**: 235 B
- **Active parameters per token**: 22 B (128 experts, 8 active per token)
- **Max context**: 128 K tokens
- **dtype**: bfloat16

## Deployments

| Directory | Hardware | TP | Latency models | Eval results |
|-----------|----------|----|----------------|--------------|
| [h100-pcie-tp1](deployments/h100-pcie-tp1/) | H100 PCIe 80 GB | 1 | physics (uncalibrated) | none yet |
