# Qwen3-32B

Dense transformer, 32 billion parameters.

- **HuggingFace**: `Qwen/Qwen3-32B`
- **Architecture**: `Qwen3ForCausalLM`
- **Parameters**: 32 B
- **Max context**: 32 K tokens (40 960 max position embeddings)
- **dtype**: bfloat16

## Deployments

| Directory | Hardware | TP | Configurations | Latency models | Eval results |
|-----------|----------|----|----------------|----------------|--------------|
| [h100-sxm5](deployments/h100-sxm5/README.md) | H100 SXM5 80 GB | 1 | standalone, pd | flat, physics (β=[0.15, 0.0, 126.0]), physics-beta-1.0 | flat, physics-beta-1.0 |
