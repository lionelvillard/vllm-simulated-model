# vllm-simulated-model

An out-of-tree vLLM plugin that runs the **full vLLM serving stack** on CPU
(incl. macOS) without a real model, real weights, or a GPU. It returns random
tokens and sleeps to reproduce a target model's latency profile, so you can
benchmark and load-test vLLM's scheduler, batching, API server, and streaming
on a laptop.

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Latency models](#latency-models)
  - [Linear model](#linear-model-type-linear)
  - [Physics model](#physics-model-type-physics)
- [Evaluation](#evaluation)
- [Comparison with related tools](#comparison-with-related-tools)
- [Limitations](#limitations)

## Quick Start

Production-calibrated deployments (Kubernetes and local CPU, including macOS)
are available for the models listed below. Each deployment directory includes:
- **k8s/** — production manifests (ConfigMap, Deployment, Service)
- **evaluation/** — latency model variants (`physics`, `flat`, `physics-beta-1.0`)
- **README.md** — full deployment instructions with prerequisites

See [Supported Models](#supported-models) for the complete list.

## Supported Models

| Model | Hardware | Deployment Modes | Directory |
|-------|----------|------------------|-----------|
| Qwen3-32B | H100 SXM5 | Standalone | [models/qwen3-32b/deployments/h100-sxm5/standalone](models/qwen3-32b/deployments/h100-sxm5/standalone/) |
| Qwen3-32B | H100 SXM5 | Prefill/Decode Disaggregation | [models/qwen3-32b/deployments/h100-sxm5/pd](models/qwen3-32b/deployments/h100-sxm5/pd/) |

## Latency models

Latency configuration lives in the model's `config.json` under the `latency`
key. The `type` field selects which model to use; it defaults to `"linear"`.

Any `latency` block can be overridden at launch with `--hf-overrides`. The
flag **replaces** the entire `latency` mapping, so include every key you want
set:

```bash
VLLM_SIMULATED_PLUGIN_CONFIG=/path/to/sim-config.json \
vllm serve <model-id> --load-format dummy \
  --hf-overrides '{"latency": {"type": "linear", "base_ms": 5.0, ...}}'
```

Both models support `deterministic_length` (bool, default `true`): when set,
the EOS token is masked so requests always run to `max_tokens` — convenient for
fixed-length ITL benchmarking.

### Linear model (`type: "linear"`)

A simple affine formula — one coefficient per batch dimension:

```
step_time_ms = base_ms
             + prefill_ms_per_token * num_prefill_tokens
             + decode_ms_per_seq    * num_decode_seqs
             + ctx_ms_per_ktoken    * (sum_context_len / 1000)
```

Use this when you have empirical timing data and want to fit coefficients
directly, or when you want a fast, interpretable model with no hardware spec.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_ms` | float | `0.0` | Fixed overhead per step (ms). |
| `prefill_ms_per_token` | float | `0.0` | Marginal cost per prefill token (ms). |
| `decode_ms_per_seq` | float | `0.0` | Marginal cost per decode sequence (ms). |
| `ctx_ms_per_ktoken` | float | `0.0` | Marginal cost per 1 k tokens of total context (ms). |

All coefficients must be ≥ 0.

```json
"latency": {
  "base_ms": 5.0,
  "prefill_ms_per_token": 0.05,
  "decode_ms_per_seq": 1.2,
  "ctx_ms_per_ktoken": 0.3,
  "deterministic_length": true
}
```

### Physics model (`type: "physics"`)

A hardware-aware roofline model adapted from
[BLIS](https://github.com/inference-sim/inference-sim). It derives step time from first
principles — FLOPs, HBM bandwidth, and model architecture — rather than fitted
coefficients:

```
T_prefill = beta_pf * max(T_compute_pf, T_kv_write_pf)
T_decode  = beta_dc * max(T_compute_dc, T_weight_load + T_kv_read_dc)
step_time_ms = T_prefill + T_decode + beta_base
```

Compute and memory terms are derived from the architecture fields already
present in `config.json` (layer count, hidden size, attention heads, FFN
dimensions, MoE topology). Both dense and MoE architectures are supported,
including interleaved MoE layers and shared-expert FFNs.

Use this when you have hardware specs (peak TFLOPs, HBM bandwidth) and want
predictions that automatically reflect model architecture and TP degree without
manual coefficient fitting.

**`hardware` block (required):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `peak_tflops` | float | required | Peak BF16/FP16 Tensor Core throughput of **one GPU**, without sparsity (TFLOPs). See reference table below. |
| `hbm_gbps` | float | required | HBM memory bandwidth of **one GPU** (GB/s). See reference table below. |
| `weight_dtype` | string | `"bfloat16"` | Byte-width used for weight loads. Use `bfloat16` or `float16` for standard serving; `float8`/`int8` for quantized models. |

**GPU reference values** — use dense (non-sparsity) figures, which match real
LLM inference:

| GPU | `peak_tflops` | `hbm_gbps` |
|-----|--------------|------------|
| H100 SXM5 | `989` | `3350` |
| H100 PCIe | `756` | `2000` |
| A100 SXM4 80 GB | `312` | `2000` |
| A100 PCIe 80 GB | `312` | `1935` |
| L40S | `362` | `864` |
| A10G | `31` | `600` |

**Top-level keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `beta` | [float, float, float] | `[1.0, 1.0, 0.0]` | Scaling factors `[beta_pf, beta_dc, beta_base]` (all ≥ 0). |
| `tp` | int | `1` | Tensor-parallel degree. Set this to match your actual deployment — the model divides weight and projection FLOPs across `tp` GPUs. |

`beta` absorbs software overhead that the roofline does not model (kernel launch
latency, NCCL collectives, framework overhead). Start with `[1.0, 1.0, 0.0]`
and tune against a short real benchmark run:

- **`beta_pf`** scales total prefill time. Raise it if measured TTFT exceeds
  prediction (typical range 1.0–1.3 for memory-bound prefill).
- **`beta_dc`** scales total decode time. Raise it if measured ITL exceeds
  prediction (typical range 1.0–1.5 for decode-bound workloads).
- **`beta_base`** is a fixed additive term (ms). Use it to capture constant
  per-step overhead such as scheduling or sampler time (typically 0–5 ms).

**Example — Qwen3-235B-A22B on A100 SXM4 80 GB:**

```json
"latency": {
  "type": "physics",
  "hardware": {
    "peak_tflops": 312.0,
    "hbm_gbps": 2000.0,
    "weight_dtype": "bfloat16"
  },
  "beta": [1.0, 1.0, 0.0],
  "tp": 8,
  "deterministic_length": true
}
```

## Evaluation

The `evaluation/` directory provides an automated tool to measure how accurately the simulator reproduces the latency and throughput behavior of a real vLLM server running on an H100 GPU. The tool runs identical benchmark workloads against both a real model deployment and the simulated model, then reports per-metric error (MAPE) and signed error to guide physics model tuning.

See [evaluation/README.md](evaluation/README.md) for the full operator runbook, including deployment steps, how to run the evaluation, and how to interpret and act on the results.

## Comparison with related tools

| | **vllm-simulated-model** | **[BLIS](https://github.com/inference-sim/inference-sim)** | **[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim)** |
|---|---|---|---|
| **What runs** | Real vLLM serving stack | Discrete-event cluster simulator (Go) | Fake HTTP server (no vLLM) |
| **Model forward pass** | Replaced with sleep | Replaced with roofline math | Replaced with sleep + jitter |
| **Scheduler / batching** | Real vLLM code | Simulated | Not simulated |
| **API surface** | Real vLLM OpenAI API | None (JSON metrics output) | Mimicked OpenAI API |
| **Interface** | `vllm serve` CLI | `blis run` CLI + YAML specs | CLI / Docker / Helm |
| **Inputs** | Model config.json + hardware spec | HF model ID + hardware + workload YAML | Model name + latency config YAML |
| **Outputs** | Live benchmark metrics (TTFT, ITL, throughput) | Capacity planning metrics (p99 TTFT/ITL, saturation) | Streaming responses + Prometheus metrics |
| **Fidelity to real vLLM** | High — real code path | Low — reimplemented model | Low — API shape only |
| **Primary use case** | High-fidelity behavior without a GPU | Cluster capacity planning and policy research | llm-d control-plane / infra development |

### The real question: how much fidelity do you need?

All three tools remove the GPU. They differ in how much *real vLLM behavior* they preserve, and that fidelity is not free — it is the axis to decide on.

- **vllm-simulated-model** runs the actual vLLM scheduler, batching, streaming, and API server; only the model forward pass is replaced with a sleep. So the emergent behavior you observe — batch composition, preemption, queueing, prefix caching, chunked prefill — is vLLM's real behavior, not a model of it. Use it when the *thing under test is vLLM itself* (or something whose correctness depends on vLLM's exact behavior).
- **BLIS** reimplements the cluster (KV cache, preemption, autoscaling) as a discrete-event model. It is faster and can explore hardware you don't own, but it cannot reproduce a vLLM-specific behavior or regression, because no vLLM code runs.
- **llm-d-inference-sim** reproduces vLLM's API surface and metrics but no scheduling logic. It is ideal as a cheap pod stand-in.

It all comes down to *which metrics have to be simulated accurately* for the experiment to be meaningful. If the metrics you care about only need realistic latency and load behavior, a lower-fidelity stand-in like llm-d-inference-sim, or a capacity model like BLIS, is usually the better fit. Reach for vllm-simulated-model when those metrics depend on vLLM's real scheduling and batching, not just on plausible-looking latency.

### Maintenance

Because this is an out-of-tree vLLM plugin rather than a reimplementation, it inherits vLLM's behavior for free and stays correct as vLLM evolves. When vLLM changes its scheduler, batching, or API, the simulation reflects that change automatically with no update here — the plugin only needs attention when the narrow contract it hooks into (the model-runner / forward-pass interface) changes. BLIS and llm-d-inference-sim, being separate implementations, must be actively tracked against vLLM to stay representative.

vllm-simulated-model's physics latency model is adapted from BLIS's roofline math, so predictions from the two tools are comparable when given the same hardware spec.

## Limitations

- Generated text is random; only timing and stack behavior are meaningful.
- Physics model `beta` parameters require manual tuning or auto-calibration via `eval.sh tune` (see [evaluation/README.md](evaluation/README.md)).
- Timing uses `time.sleep` (~1 ms granularity); best for ITL ≳ a few ms.
