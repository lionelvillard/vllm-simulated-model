# vllm-simulated-model

An out-of-tree vLLM plugin that runs the **full vLLM serving stack** on CPU
(incl. macOS) without a real model, real weights, or a GPU. It returns random
tokens and sleeps to reproduce a target model's latency profile, so you can
benchmark and load-test vLLM's scheduler, batching, API server, and streaming
on a laptop.

## Table of Contents

- [Get Started](#get-started)
  - [Local](#local)
  - [Kubernetes](#kubernetes)
- [Latency models](#latency-models)
  - [Linear model](#linear-model-type-linear)
  - [Physics model](#physics-model-type-physics)
- [Comparison with related tools](#comparison-with-related-tools)
- [Limitations](#limitations)

## Get Started

### Local

> **Note:** vLLM must be compiled from source — `pip install vllm` does not
> support CPU-only / macOS environments. See the
> [vLLM build guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
> for platform-specific instructions.

```bash
uv venv --python 3.12
source .venv/bin/activate
# Build and install vLLM from source first (see note above)
uv pip install -e .
```

The plugin registers itself via the `vllm.general_plugins` entry point; no code
changes to vLLM are needed.

**Run the simulator:**

```bash
vllm serve ./examples/sim-qwen-3.8-27b \
  --load-format dummy \
  --gpu-memory-utilization 0.2 \
  --tokenizer Qwen/Qwen3-8B \
  --served-model-name sim-qwen-3.8-27b
```

Then benchmark it like any vLLM server (e.g. `vllm bench serve ...`). Measured
TTFT/ITL reflect the configured latency model, while every other component is
the real vLLM code path.

**Send a test request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sim-qwen-3.8-27b",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 32
  }'
```

### Kubernetes

The `deploy/` directory contains ready-to-use manifests for CPU-only linux/amd64 nodes.
No custom image build is required — the plugin is injected at pod startup via an init
container.

**Prerequisites:** a cluster with at least one CPU node (4 CPU / 8 Gi) and internet
egress so the init container can fetch the package from GitHub.

If you need a private HuggingFace tokenizer, create the Secret before deploying
(the deployment has `optional: true` so it starts fine without it):

```bash
kubectl create secret generic hf-token \
  --from-literal=token=<your-hf-token> \
  -n <namespace>
```

**Run the simulator:**

```bash
kubectl apply -n <namespace> -f deploy/
```

This creates:

| Resource | Name | Purpose |
|---|---|---|
| ConfigMap | `vllm-sim-model-config` | Qwen 3.5 model config and latency coefficients |
| Deployment | `vllm-sim` | Single-replica CPU vLLM server |
| Service | `vllm-sim` | ClusterIP on port 8000 |

The pod takes roughly 60–120 s to become ready: the init container installs the plugin
(~10–30 s depending on network), then vLLM initializes the CPU engine.

**Send a test request:**

```bash
kubectl port-forward svc/vllm-sim 8000:8000
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sim-qwen-3.8-27b",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 16
  }'
```

#### Tuning

| What to change | Where | Guidance |
|---|---|---|
| KV cache size | `VLLM_CPU_KVCACHE_SPACE` env | Increase until ~80% of node memory used (default: 4 GB) |
| Thread binding | `VLLM_CPU_OMP_THREADS_BIND` env | Match the CPU limit range, e.g. `0-7` for 8 CPUs (default: `0-3`) |
| Memory cap | `--gpu-memory-utilization` in command | Increase toward `0.9` once stable (default: `0.5`) |
| Model config | `deploy/configmap.yaml` | Edit the `config.json` data to change latency coefficients |
| Tokenizer | `--tokenizer` in command | Use any HuggingFace tokenizer; set `hf-token` Secret if private |
| Model name | `--served-model-name` in command | Change the name clients use in the `model` field |
| Plugin version | tarball URL in init container command | Replace `/heads/main` with a release tag or commit SHA for production |

#### OpenShift / non-root clusters

The manifests already include the adjustments required for clusters that enforce non-root
security policies:

- `HOME=/tmp` and `XDG_CACHE_HOME=/tmp/cache` — redirect vLLM's Triton and model-info
  caches away from read-only system paths
- A `Memory`-backed emptyDir at `/dev/shm` (256 Mi) — vLLM's engine IPC requires more
  than the 64 Mi Kubernetes default

## Latency models

Latency configuration lives in the model's `config.json` under the `latency`
key. The `type` field selects which model to use; it defaults to `"linear"`.

Any `latency` block can be overridden at launch with `--hf-overrides`. The
flag **replaces** the entire `latency` mapping, so include every key you want
set:

```bash
vllm serve ./examples/sim-qwen-3.8-27b --load-format dummy \
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
- Coefficients are user-supplied (no auto-calibration yet).
- Timing uses `time.sleep` (~1 ms granularity); best for ITL ≳ a few ms.
