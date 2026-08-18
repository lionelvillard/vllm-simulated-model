# vllm-simulated-model

An out-of-tree vLLM plugin that runs the **full vLLM serving stack** on CPU
(incl. macOS) without a real model, real weights, or a GPU. It returns random
tokens and sleeps to reproduce a target model's latency profile, so you can
benchmark and load-test vLLM's scheduler, batching, API server, and streaming
on a laptop.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm
uv pip install -e .
```

The plugin registers itself via the `vllm.general_plugins` entry point; no code
changes to vLLM are needed.

## Run

```bash
vllm serve ./examples/sim-llama3-8b \
  --load-format dummy \
  --tokenizer meta-llama/Meta-Llama-3-8B   # or --skip-tokenizer-init
```

Then benchmark it like any vLLM server (e.g. `vllm bench serve ...`). Measured
TTFT/ITL reflect the configured latency model, while every other component is
the real vLLM code path.

## Latency model

Per engine step:

```
step_time_ms = base_ms
             + prefill_ms_per_token * num_prefill_tokens
             + decode_ms_per_seq    * num_decode_seqs
             + ctx_ms_per_ktoken    * (sum_context_len / 1000)
```

Coefficients live in the model's `config.json` under `latency` and can be
overridden at launch:

```bash
vllm serve ./examples/sim-llama3-8b --load-format dummy \
  --hf-overrides '{"latency": {"decode_ms_per_seq": 2.0}}'
```

`deterministic_length: true` masks the EOS token so requests run to `max_tokens`
— convenient for fixed-length ITL benchmarking.

## Limitations

- Generated text is random; only timing and stack behavior are meaningful.
- Coefficients are user-supplied (no auto-calibration yet).
- Timing uses `time.sleep` (~1 ms granularity); best for ITL ≳ a few ms.
