# Simulated Model Plugin for vLLM — Design

**Date:** 2026-08-18
**Status:** Design approved (pending spec review)
**Type:** Out-of-tree vLLM plugin (standalone package)

## 1. Purpose & Scope

Provide a **simulated model** that lets vLLM run its full serving stack — API
server, tokenizer, scheduler, KV-cache manager, continuous batching, sampler,
streaming — **without a real model, real weights, or a GPU**. The model returns
random tokens and sleeps to reproduce a target model's latency profile.

**Primary use case:** benchmark and load-test the vLLM serving stack on a
laptop (CPU, incl. macOS / Apple Silicon), validating throughput and
concurrency behavior where the *compute* is faked but every other component is
the real vLLM code path.

**Explicitly out of scope (MVP):**
- Real model quality / correctness of generated text (tokens are random).
- Automatic latency calibration from a real run (coefficients are user-supplied).
- Trace-driven replay, named presets, distribution sampling (possible future work).
- GPU-specific timing fidelity (device-agnostic, but only CPU is targeted/tested).

## 2. Delivery: Out-of-Tree Plugin

The feature ships as a **standalone Python package** (`vllm-simulated-model`),
installed alongside vLLM. It does **not** modify the vLLM source tree.

- Package registers itself through the `vllm.general_plugins` entry-point group
  (discovered by `vllm/plugins/__init__.py:load_general_plugins`).
- Its `register()` function calls
  `ModelRegistry.register_model("SimulatedForCausalLM", "vllm_simulated.model:SimulatedForCausalLM")`.
  The **lazy `"module:class"` string form** is required so importing the plugin
  does not initialize CUDA in a forked subprocess.
- Documented pattern: `docs/contributing/model/registration.md`,
  `docs/design/plugin_system.md` in the vLLM repo.

Package location: new sibling directory outside the vLLM checkout
(`../vllm-simulated-model/`), initialized as its own git repository.

Proposed layout:

```
vllm-simulated-model/
  pyproject.toml                    # entry point: vllm.general_plugins -> vllm_simulated:register
  src/vllm_simulated/
    __init__.py                     # register()
    model.py                        # SimulatedForCausalLM
    latency.py                      # SimulatedLatencyModel + config parsing/validation
  examples/
    sim-llama3-8b/config.json       # HF config (llama-3-8b shape + latency block)
  tests/
    test_latency.py                 # unit: formula
    test_end_to_end.py              # integration: LLM(...) on CPU
  docs/
    2026-08-18-simulated-model-plugin-design.md
  README.md
```

## 3. Architecture

The plugin adds one model architecture that satisfies vLLM's V1
`VllmModelForTextGeneration` contract (template: `vllm/model_executor/models/gpt2.py`).
It runs on the **stock CPU model runner** (`CPUModelRunner`, which subclasses
`GPUModelRunner`); no runner, executor, scheduler, or platform changes.

### 3.1 `SimulatedForCausalLM` (`model.py`)

Minimal V1 interface: `__init__(self, *, vllm_config, prefix="")`,
`embed_input_ids`, `forward`, `compute_logits`, `load_weights`.

**`__init__`**
- Read shape from the parsed HF config (available independently of weights, via
  `ModelConfig` getters that map to HF fields): `hidden_size`, `vocab_size`,
  `num_hidden_layers`, `num_kv_heads`, `head_size`.
- Construct `num_hidden_layers` real `vllm.model_executor.layers.attention.Attention`
  modules, sized with `model_config.get_num_kv_heads()` / `get_head_size()`.
  **Rationale:** the V1 runner derives the KV-cache spec by *type-discovering*
  `AttentionLayerBase` modules (`gpu_model_runner.py:get_kv_cache_spec`) and
  calling `get_kv_cache_spec()` on each. Registering correctly-sized attention
  modules yields a non-empty, correctly-sized `FullAttentionSpec`, so the
  scheduler and KV-cache manager allocate/free blocks exactly as for a real
  transformer of that shape. The modules' attention math is **never executed**.
- Build the `SimulatedLatencyModel` from the config's `latency` block.
- No `lm_head` / `LogitsProcessor` and no real parameters: `compute_logits`
  synthesizes logits directly (see 3.2). A dummy `lm_head` Linear would perform
  a real (random-weight) matmul on CPU, adding unintended latency and undercutting
  the point of faked compute — so it is deliberately omitted.

**`forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None)`**
- Read batch composition from the forward context
  (`vllm.forward_context.get_forward_context().attn_metadata`).
- Derive `(num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)`
  via `split_decodes_and_prefills(common_attn_metadata)` and context lengths
  from `seq_lens`.
- Compute `step_time_ms = SimulatedLatencyModel(...)` and **sleep** for it.
- Return a dummy `hidden_states` tensor of shape `[num_tokens, hidden_size]`
  (zeros or small random), so downstream `hidden_states[logits_indices]` and
  `compute_logits` operate on correctly-shaped input.

**Why the sleep is faithful:** on CPU the engine step is fully synchronous —
`UniProcExecutor.collective_rpc(non_block=True)` runs inline, async scheduling
is force-disabled by `CpuPlatform` (`vllm/platforms/cpu.py`), CUDA graphs are
off, and device sync is a no-op. A `time.sleep()` in `forward` therefore
directly delays that engine step. One decode step serves the entire current
batch after a single sleep — mirroring real continuous-batching dynamics
(larger batch → one step, priced by the latency model).

**`compute_logits(hidden_states) -> Tensor`**
- Return `torch.randn(num_tokens, vocab_size)` so the **real sampler** applies
  each request's sampling params (temperature/top-p/…) and picks tokens. This
  exercises the sampler and yields random output tokens.
- Mask the EOS token id (set its logit to `-inf`) when
  `deterministic_length` is enabled (default on), so decode never
  early-terminates and output length equals the request's `max_tokens` — the
  behavior benchmarks want. Toggleable via config.

**`load_weights(weights)`** — accept and ignore the iterator. Used with
`--load-format dummy`, which downloads nothing and fills no weights.

### 3.2 `SimulatedLatencyModel` (`latency.py`)

Pure, batch-aware, linear model:

```
step_time_ms = base_ms
             + prefill_ms_per_token * num_prefill_tokens
             + decode_ms_per_seq    * num_decode_seqs
             + ctx_ms_per_ktoken    * (sum_context_len / 1000)
```

- **TTFT** emerges from prefill-step timing; **ITL** from decode-step timing —
  both fall out of one formula rather than being separately modeled.
- Inputs come from the forward-context batch metadata (see 3.1).
- Coefficients are user-supplied (see §4). Validated non-negative at init;
  computed `step_time` floored at 0.
- Precision: `time.sleep()` (~1ms granularity) is adequate for typical ITL
  (10–100 ms). An optional busy-wait mode for sub-ms precision is noted as a
  refinement, not MVP.

## 4. Configuration

No new CLI flags. Coefficients live in a `latency` block in the model's
`config.json`, read from `hf_config`; overridable at launch via the existing
`--hf-overrides`.

Example `examples/sim-llama3-8b/config.json` (abridged):

```json
{
  "architectures": ["SimulatedForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "latency": {
    "base_ms": 5.0,
    "prefill_ms_per_token": 0.05,
    "decode_ms_per_seq": 1.2,
    "ctx_ms_per_ktoken": 0.3,
    "deterministic_length": true
  }
}
```

Launch:

```bash
vllm serve ./examples/sim-llama3-8b \
  --load-format dummy \
  --tokenizer meta-llama/Meta-Llama-3-8B    # borrow a real tokenizer, or --skip-tokenizer-init
```

Override coefficients without editing the file:

```bash
vllm serve ./examples/sim-llama3-8b --load-format dummy \
  --hf-overrides '{"latency": {"decode_ms_per_seq": 2.0}}'
```

## 5. Error Handling

- Missing/invalid `latency` block → clear error at model init naming the
  offending field.
- Negative coefficients → rejected at init.
- Computed `step_time` floored at 0 to guard against pathological configs.
- Missing shape fields in `config.json` → surfaced by vLLM's normal config
  parsing.

## 6. Testing

- **Unit (`test_latency.py`)**: `SimulatedLatencyModel` formula. Given synthetic
  `(num_prefill_tokens, num_decode_seqs, sum_context_len)`, assert the exact
  `step_time_ms`. Pure function; no engine, no sleep.
- **Integration (`test_end_to_end.py`)**: after `pip install -e .`, run
  `LLM(model="examples/sim-llama3-8b", load_format="dummy",
  skip_tokenizer_init=...)` on CPU. Assert: (a) output length equals requested
  `max_tokens` (EOS masking works), (b) measured mean ITL over N decode steps ≈
  `decode_ms_per_seq`-derived expectation within tolerance, (c) runs with no GPU
  and no model download. Kept small (short outputs) for CI.
- Latency-timing assertions use generous tolerances to avoid flakiness on shared
  CI runners.

## 7. Component Contracts (isolation check)

- `SimulatedLatencyModel` — **does**: maps batch composition → step time.
  **Uses**: called with primitive counts; no vLLM deps → trivially unit-testable.
  **Depends on**: nothing but its coefficients.
- `SimulatedForCausalLM` — **does**: satisfies the V1 model contract, injects
  sleep + random logits. **Uses**: instantiated by vLLM's loader from a
  `config.json`. **Depends on**: vLLM model interfaces, `Attention`,
  forward-context, `SimulatedLatencyModel`.
- `register()` — **does**: registers the arch. **Uses**: called by vLLM plugin
  discovery. **Depends on**: `ModelRegistry`.

## 8. Key vLLM Integration Points (verified)

- Registration API: `ModelRegistry.register_model` (`vllm/model_executor/models/registry.py:1083`), lazy `"module:class"` form.
- Plugin discovery: `vllm.general_plugins` entry point (`vllm/plugins/__init__.py`).
- Model contract & template: `interfaces_base.py` (`VllmModelForTextGeneration`), `vllm/model_executor/models/gpt2.py`.
- Forward hook is the model's own `forward` (called at `gpu_model_runner.py:_model_forward` → `self.model(...)`), synchronous on CPU.
- Batch metadata: `split_decodes_and_prefills` (`vllm/v1/attention/backends/utils.py`), `CommonAttentionMetadata.seq_lens`.
- KV-cache spec: type-discovered `Attention` modules → `FullAttentionSpec` (`gpu_model_runner.py:get_kv_cache_spec`, `attention.py`).
- CPU memory sizing needs no forward pass (`vllm/v1/worker/cpu_worker.py:determine_available_memory`).
- `--load-format dummy`: `DummyModelLoader` (`vllm/model_executor/model_loader/dummy_loader.py`).
- macOS: `CpuPlatform` handles Darwin/Apple Silicon dtype + gloo backend (`vllm/platforms/cpu.py`).

## 9. Future Work (not MVP)

- Calibration tool that fits coefficients from a real `vllm bench` run.
- Named presets for common model/hardware combos.
- Per-request latency variance (distribution sampling).
- Busy-wait precision mode for sub-ms ITL.
