# Simulated Model Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an out-of-tree vLLM plugin that registers a `SimulatedForCausalLM` model which runs the full vLLM serving stack on CPU while faking compute and reproducing a target model's latency profile.

**Architecture:** A standalone package registers one model architecture via the `vllm.general_plugins` entry point. The model declares correctly-sized `Attention` modules so the real scheduler and KV-cache manager behave like a real transformer, but its `forward` runs no attention math — it reads batch composition from the forward context, sleeps for a batch-aware computed step time, and returns dummy hidden states. `compute_logits` returns random logits so the real sampler picks (random) tokens.

**Tech Stack:** Python 3.12, PyTorch (CPU), vLLM (installed separately), pytest, `uv` for env management.

**Spec:** `docs/2026-08-18-simulated-model-plugin-design.md` (in this package repo)

## Global Constraints

- **Out-of-tree only.** Do NOT edit the vLLM source tree. The plugin is a separate package installed alongside vLLM.
- **Python env via `uv`/`.venv`.** Never use system `python3` or bare `pip`. Use `uv venv --python 3.12` and `.venv/bin/python`.
- **Line length 88** for Python.
- **CPU / macOS target.** No CUDA required at runtime. Tests must pass with no GPU and no model download.
- **Lazy registration.** `register_model` MUST use the `"module:class"` string form, never import the model class eagerly (avoids CUDA-fork errors).
- **No real parameters.** No `lm_head`, no `LogitsProcessor`, no real weights — `load_weights` is a no-op and logits are synthesized directly.
- **Package name:** distribution `vllm-simulated-model`, import package `vllm_simulated`. Repo root is the sibling directory `vllm-simulated-model/` (already contains `docs/` + git repo).

---

### Task 1: Package scaffold + latency core

**Files:**
- Create: `pyproject.toml`
- Create: `src/vllm_simulated/__init__.py` (empty placeholder for now)
- Create: `src/vllm_simulated/latency.py`
- Test: `tests/test_latency.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LatencyConfig` dataclass (frozen) with fields `base_ms: float`, `prefill_ms_per_token: float`, `decode_ms_per_seq: float`, `ctx_ms_per_ktoken: float`, `deterministic_length: bool` (default `True`); classmethod `from_dict(d: dict) -> LatencyConfig`; method `validate() -> None`.
  - `BatchShape` dataclass (frozen): `num_prefill_tokens: int`, `num_decode_seqs: int`, `sum_context_len: int`.
  - `SimulatedLatencyModel(config: LatencyConfig)` with `step_time_ms(shape: BatchShape) -> float`.
  - `batch_shape_from_attn_metadata(md) -> BatchShape` — reads `md.query_start_loc` (1-D tensor of per-request cumulative query offsets) and `md.seq_lens` (1-D tensor of per-request context lengths).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vllm-simulated-model"
version = "0.1.0"
description = "A simulated vLLM model that fakes compute and reproduces latency profiles."
requires-python = ">=3.12"
dependencies = ["torch"]

[project.optional-dependencies]
test = ["pytest", "vllm"]

[project.entry-points."vllm.general_plugins"]
simulated_model = "vllm_simulated:register"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
```

- [ ] **Step 2: Create empty `src/vllm_simulated/__init__.py`**

```python
# register() is added in Task 2.
```

- [ ] **Step 3: Write the failing tests** in `tests/test_latency.py`

```python
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_simulated.latency import (
    BatchShape,
    LatencyConfig,
    SimulatedLatencyModel,
    batch_shape_from_attn_metadata,
)


def test_step_time_is_linear_combination():
    cfg = LatencyConfig(
        base_ms=5.0,
        prefill_ms_per_token=0.05,
        decode_ms_per_seq=1.2,
        ctx_ms_per_ktoken=0.3,
    )
    model = SimulatedLatencyModel(cfg)
    shape = BatchShape(num_prefill_tokens=100, num_decode_seqs=10, sum_context_len=2000)
    # 5 + 0.05*100 + 1.2*10 + 0.3*(2000/1000) = 5 + 5 + 12 + 0.6 = 22.6
    assert math.isclose(model.step_time_ms(shape), 22.6, rel_tol=1e-9)


def test_step_time_never_negative():
    model = SimulatedLatencyModel(LatencyConfig())
    assert model.step_time_ms(BatchShape(0, 0, 0)) == 0.0


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown latency config keys"):
        LatencyConfig.from_dict({"base_ms": 1.0, "bogus": 2.0})


def test_from_dict_rejects_negative_coefficients():
    with pytest.raises(ValueError, match="must be >= 0"):
        LatencyConfig.from_dict({"decode_ms_per_seq": -1.0})


def test_batch_shape_from_attn_metadata():
    # 3 requests with query lengths 1, 1, 3 -> 2 decodes, 1 prefill of 3 tokens.
    md = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2, 5]),
        seq_lens=torch.tensor([10, 12, 7]),
    )
    shape = batch_shape_from_attn_metadata(md)
    assert shape == BatchShape(
        num_prefill_tokens=3, num_decode_seqs=2, sum_context_len=29
    )
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vllm_simulated.latency'`

- [ ] **Step 5: Implement `src/vllm_simulated/latency.py`**

```python
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LatencyConfig:
    base_ms: float = 0.0
    prefill_ms_per_token: float = 0.0
    decode_ms_per_seq: float = 0.0
    ctx_ms_per_ktoken: float = 0.0
    deterministic_length: bool = True

    _COEFFICIENTS = (
        "base_ms",
        "prefill_ms_per_token",
        "decode_ms_per_seq",
        "ctx_ms_per_ktoken",
    )

    @classmethod
    def from_dict(cls, d: dict) -> "LatencyConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown latency config keys: {sorted(unknown)}")
        config = cls(**d)
        config.validate()
        return config

    def validate(self) -> None:
        for name in self._COEFFICIENTS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"latency.{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class BatchShape:
    num_prefill_tokens: int
    num_decode_seqs: int
    sum_context_len: int


class SimulatedLatencyModel:
    def __init__(self, config: LatencyConfig) -> None:
        self.config = config

    def step_time_ms(self, shape: BatchShape) -> float:
        c = self.config
        total = (
            c.base_ms
            + c.prefill_ms_per_token * shape.num_prefill_tokens
            + c.decode_ms_per_seq * shape.num_decode_seqs
            + c.ctx_ms_per_ktoken * (shape.sum_context_len / 1000.0)
        )
        return max(0.0, total)


def batch_shape_from_attn_metadata(md) -> BatchShape:
    query_start_loc = md.query_start_loc
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_decode = query_lens <= 1
    num_decode_seqs = int(is_decode.sum().item())
    num_prefill_tokens = int(query_lens[~is_decode].sum().item())
    sum_context_len = int(md.seq_lens.sum().item())
    return BatchShape(
        num_prefill_tokens=num_prefill_tokens,
        num_decode_seqs=num_decode_seqs,
        sum_context_len=sum_context_len,
    )
```

Note: `_COEFFICIENTS` is a class attribute, not a dataclass field (it has no type annotation), so it is excluded from `fields()` and the generated `__init__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_latency.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/vllm_simulated/__init__.py src/vllm_simulated/latency.py tests/test_latency.py
git commit -s -m "Add latency core and package scaffold"
```

---

### Task 2: Model class, registration, and end-to-end load

**Files:**
- Create: `src/vllm_simulated/model.py`
- Modify: `src/vllm_simulated/__init__.py` (add `register()`)
- Create: `tests/fixtures/sim-tiny/config.json`
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `LatencyConfig`, `SimulatedLatencyModel`, `BatchShape`, `batch_shape_from_attn_metadata` from Task 1.
- Produces:
  - `SimulatedForCausalLM(*, vllm_config, prefix="")` — vLLM V1 text-generation model with `embed_input_ids`, `forward`, `compute_logits`, `load_weights`.
  - `register() -> None` in `vllm_simulated/__init__.py`.

- [ ] **Step 1: Create the tiny fixture config** `tests/fixtures/sim-tiny/config.json`

```json
{
  "architectures": ["SimulatedForCausalLM"],
  "model_type": "llama",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "vocab_size": 1000,
  "max_position_embeddings": 4096,
  "eos_token_id": 0,
  "torch_dtype": "float32",
  "latency": {
    "base_ms": 1.0,
    "decode_ms_per_seq": 2.0,
    "deterministic_length": true
  }
}
```

- [ ] **Step 2: Write the failing end-to-end test** in `tests/test_end_to_end.py`

```python
import pytest

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

import vllm_simulated  # noqa: E402


@pytest.fixture(scope="module")
def sim_llm(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("VLLM_CPU_KVCACHE_SPACE", "1")
    vllm_simulated.register()
    return LLM(
        model="tests/fixtures/sim-tiny",
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
    )


def test_output_length_honors_max_tokens(sim_llm):
    outputs = sim_llm.generate(
        TokensPrompt(prompt_token_ids=[1, 2, 3, 4]),
        SamplingParams(max_tokens=8),
    )
    # deterministic_length masks EOS, so the request runs to max_tokens.
    assert len(outputs[0].outputs[0].token_ids) == 8
```

Add a module-scoped monkeypatch fixture in `tests/conftest.py`:

```python
import pytest


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_end_to_end.py -v`
Expected: FAIL — architecture `SimulatedForCausalLM` not registered / model class missing.

- [ ] **Step 4: Implement `src/vllm_simulated/model.py`**

```python
import time

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention

from vllm_simulated.latency import (
    LatencyConfig,
    SimulatedLatencyModel,
    batch_shape_from_attn_metadata,
)


class SimulatedForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        hf_config = model_config.hf_config

        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()
        self.dtype = model_config.dtype
        num_layers = model_config.get_total_num_hidden_layers()
        num_heads = model_config.get_num_attention_heads(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()

        # Real Attention modules make get_kv_cache_spec() non-empty so the
        # scheduler/KV-cache manager allocate blocks like a real transformer.
        # Their attention math is never executed in forward().
        self.attn_layers = nn.ModuleList(
            [
                Attention(
                    num_heads=num_heads,
                    head_size=head_size,
                    scale=head_size**-0.5,
                    num_kv_heads=num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{i}.attn",
                )
                for i in range(num_layers)
            ]
        )

        latency_config = LatencyConfig.from_dict(
            getattr(hf_config, "latency", None) or {}
        )
        self.latency = SimulatedLatencyModel(latency_config)
        self.deterministic_length = latency_config.deterministic_length
        self.eos_token_id = getattr(hf_config, "eos_token_id", None)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            input_ids.shape[0], self.hidden_size, dtype=self.dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            num_tokens = inputs_embeds.shape[0]
        else:
            num_tokens = input_ids.shape[0]

        md = self._current_attn_metadata()
        if md is not None:
            shape = batch_shape_from_attn_metadata(md)
            sleep_s = self.latency.step_time_ms(shape) / 1000.0
            if sleep_s > 0:
                time.sleep(sleep_s)

        return torch.zeros(num_tokens, self.hidden_size, dtype=self.dtype)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        logits = torch.randn(num_tokens, self.vocab_size)
        if self.deterministic_length and self.eos_token_id is not None:
            logits[:, self.eos_token_id] = float("-inf")
        return logits

    def load_weights(self, weights) -> set[str]:
        # No real parameters; consume the (empty) iterator and report nothing.
        for _ in weights:
            pass
        return set()

    @staticmethod
    def _current_attn_metadata():
        attn_metadata = get_forward_context().attn_metadata
        if not attn_metadata:
            return None
        if isinstance(attn_metadata, list):  # dual-batch overlap microbatches
            attn_metadata = attn_metadata[0]
        return next(iter(attn_metadata.values()))
```

- [ ] **Step 5: Add `register()` to `src/vllm_simulated/__init__.py`**

```python
def register() -> None:
    """Entry point for the `vllm.general_plugins` group."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "SimulatedForCausalLM",
        "vllm_simulated.model:SimulatedForCausalLM",
    )
```

- [ ] **Step 6: Install the package into the venv and run the test**

Run:
```bash
.venv/bin/python -m pip --version  # sanity: use the venv, not system pip
uv pip install -e .
.venv/bin/python -m pytest tests/test_end_to_end.py -v
```
Expected: PASS. If instantiation fails on CPU memory sizing, set `VLLM_CPU_KVCACHE_SPACE=1` (already set in the fixture) or pass a smaller value; if it complains about compilation, confirm `enforce_eager=True` is set.

- [ ] **Step 7: Commit**

```bash
git add src/vllm_simulated/model.py src/vllm_simulated/__init__.py tests/fixtures tests/conftest.py tests/test_end_to_end.py
git commit -s -m "Add SimulatedForCausalLM model and plugin registration"
```

---

### Task 3: Latency-fidelity test, demo config, and README

**Files:**
- Test: `tests/test_end_to_end.py` (add one test)
- Create: `examples/sim-llama3-8b/config.json`
- Create: `README.md`

**Interfaces:**
- Consumes: `sim_llm` fixture and `register()` from Task 2.
- Produces: no new code interfaces (test + docs + example only).

- [ ] **Step 1: Write the failing latency lower-bound test** (append to `tests/test_end_to_end.py`)

```python
import time as _time


def test_decode_latency_lower_bound(sim_llm):
    # sim-tiny: base_ms=1, decode_ms_per_seq=2 -> each decode step sleeps ~3ms
    # for a single sequence. With max_tokens=10 there are ~10 steps.
    start = _time.perf_counter()
    sim_llm.generate(
        TokensPrompt(prompt_token_ids=[1, 2, 3, 4]),
        SamplingParams(max_tokens=10),
    )
    elapsed_ms = (_time.perf_counter() - start) * 1000.0
    # time.sleep guarantees AT LEAST the requested duration, so a lower bound
    # is non-flaky. 10 decode steps * 3ms = 30ms; allow slack for the prefill
    # step and scheduling. Assert we spent at least half the modeled decode time.
    assert elapsed_ms >= 10 * 3.0 * 0.5
```

Rationale: only a **lower** bound is asserted. `time.sleep` cannot return early, so this cannot flake on slow CI; it only proves the latency injection actually delays generation.

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `.venv/bin/python -m pytest tests/test_end_to_end.py::test_decode_latency_lower_bound -v`
Expected: If Task 2 is complete, this PASSES immediately (the behavior already exists). If it fails, the sleep injection in `forward` is not firing — debug `_current_attn_metadata` returning `None`.

Note: this test asserts existing behavior rather than driving new code; that is acceptable here because it guards a distinct, regression-prone property (timing) that the length test does not cover.

- [ ] **Step 3: Create the demo config** `examples/sim-llama3-8b/config.json`

```json
{
  "architectures": ["SimulatedForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "max_position_embeddings": 8192,
  "eos_token_id": 128001,
  "torch_dtype": "bfloat16",
  "latency": {
    "base_ms": 5.0,
    "prefill_ms_per_token": 0.05,
    "decode_ms_per_seq": 1.2,
    "ctx_ms_per_ktoken": 0.3,
    "deterministic_length": true
  }
}
```

- [ ] **Step 4: Write `README.md`**

````markdown
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
````

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS (all tests from Tasks 1–3).

- [ ] **Step 6: Commit**

```bash
git add tests/test_end_to_end.py examples/sim-llama3-8b/config.json README.md
git commit -s -m "Add latency-fidelity test, demo config, and README"
```

---

## Self-Review

**Spec coverage:**
- §2 out-of-tree plugin + `vllm.general_plugins` entry point + lazy registration → Task 1 (pyproject entry point), Task 2 (`register()`). ✓
- §3.1 model contract (`__init__`, `embed_input_ids`, `forward`, `compute_logits`, `load_weights`), Attention modules for KV-cache realism, sleep in forward, random logits, EOS masking, no lm_head → Task 2. ✓
- §3.2 batch-aware linear latency model + batch-metadata extraction → Task 1 (`SimulatedLatencyModel`, `batch_shape_from_attn_metadata`). ✓
- §4 config via `latency` block in `config.json` + `--hf-overrides` → Task 2 (`LatencyConfig.from_dict(hf_config.latency)`), Task 3 (README documents `--hf-overrides`). ✓
- §5 error handling (invalid/negative coefficients, floor at 0) → Task 1 (`validate`, `max(0.0, …)`). ✓
- §6 unit + integration tests → Task 1 (unit), Task 2 (e2e length), Task 3 (timing). ✓
- §2 layout (examples, README) → Task 2 (tiny fixture), Task 3 (demo config + README). ✓

**Placeholder scan:** No TBD/TODO; every code step contains full content. ✓

**Type consistency:** `LatencyConfig`, `BatchShape`, `SimulatedLatencyModel.step_time_ms`, `batch_shape_from_attn_metadata` names/signatures match between Task 1 definitions and Task 2 usage. `register()` string `"vllm_simulated.model:SimulatedForCausalLM"` matches the class in `model.py` and the entry point `vllm_simulated:register` matches `__init__.py`. ✓

**Known integration risks (for the executor):**
- If `get_num_attention_heads`/`get_num_kv_heads` signatures differ in the installed vLLM version, adjust the argument (some versions take no `parallel_config`). Verify against the installed `vllm.config.ModelConfig`.
- If CPU engine init needs additional flags (e.g. explicit `dtype="float32"` on macOS without BF16), pass them in the test `LLM(...)` call; the tiny fixture already pins `torch_dtype: float32`.
- If `attn_metadata` values are keyed/wrapped differently, confirm `next(iter(...))` yields an object exposing `query_start_loc` and `seq_lens` (CPU backend: `CPUAttentionMetadata`).
