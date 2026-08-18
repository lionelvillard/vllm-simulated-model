# Plugable Latency Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `LatencyModel` Protocol, registry, and factory so alternative latency formula shapes can be added without restructuring the codebase.

**Architecture:** Add a `LatencyModel` structural Protocol and a `build_latency_model(d: dict) -> LatencyModel` factory to `latency.py`. The factory dispatches on an optional `"type"` key (default `"linear"`) to a `_REGISTRY` dict. `SimulatedLatencyModel` gains a `from_dict` classmethod and is registered as `"linear"`. `model.py` replaces direct construction with the factory; no runtime behavior changes.

**Tech Stack:** Python 3.12, PyTorch (CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-plugable-latency-model-design.md`

## Global Constraints

- Out-of-tree only. Do NOT edit the vLLM source tree.
- Python env via `uv`/`.venv`. Use `.venv/bin/python -m pytest`, not system `pytest`.
- Line length 88.
- No new files — only `latency.py`, `model.py`, and their tests are touched.
- All existing tests must continue to pass after every task.

---

### Task 1: Protocol, registry, and factory in `latency.py`

**Files:**
- Modify: `src/vllm_simulated/latency.py`
- Test: `tests/test_latency.py`

**Interfaces:**
- Consumes: `BatchShape`, `LatencyConfig`, `SimulatedLatencyModel` already in `latency.py`.
- Produces:
  - `LatencyModel` — `typing.Protocol` with one method: `step_time_ms(self, shape: BatchShape) -> float`
  - `SimulatedLatencyModel.from_dict(cls, d: dict) -> SimulatedLatencyModel` classmethod
  - `_REGISTRY: dict[str, type]` — module-level dict `{"linear": SimulatedLatencyModel}`
  - `build_latency_model(d: dict) -> LatencyModel` — factory function

- [ ] **Step 1: Add failing tests to `tests/test_latency.py`**

Update the import block at the top of the file (add `build_latency_model`):

```python
from vllm_simulated.latency import (
    BatchShape,
    LatencyConfig,
    SimulatedLatencyModel,
    batch_shape_from_attn_metadata,
    build_latency_model,
)
```

Append these four tests at the end of the file:

```python
def test_build_latency_model_linear_explicit():
    model = build_latency_model({"type": "linear", "base_ms": 1.0})
    assert isinstance(model, SimulatedLatencyModel)
    shape = BatchShape(num_prefill_tokens=0, num_decode_seqs=0, sum_context_len=0)
    assert model.step_time_ms(shape) == 1.0


def test_build_latency_model_default_type():
    model = build_latency_model({"base_ms": 2.0})
    assert isinstance(model, SimulatedLatencyModel)
    shape = BatchShape(num_prefill_tokens=0, num_decode_seqs=0, sum_context_len=0)
    assert model.step_time_ms(shape) == 2.0


def test_build_latency_model_unknown_type():
    with pytest.raises(ValueError, match="Unknown latency model type"):
        build_latency_model({"type": "bogus"})


def test_build_latency_model_does_not_mutate_input():
    d = {"type": "linear", "base_ms": 1.0}
    original = dict(d)
    build_latency_model(d)
    assert d == original
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_latency.py -v
```

Expected: 4 new tests FAIL with `ImportError: cannot import name 'build_latency_model'`. Existing 5 tests still PASS.

- [ ] **Step 3: Implement the changes in `src/vllm_simulated/latency.py`**

Add `Protocol` to the typing import and insert the Protocol class, `from_dict` classmethod, registry, and factory. The complete updated file:

```python
from dataclasses import dataclass, fields
from typing import ClassVar, Protocol


class LatencyModel(Protocol):
    def step_time_ms(self, shape: "BatchShape") -> float: ...


@dataclass(frozen=True)
class LatencyConfig:
    base_ms: float = 0.0
    prefill_ms_per_token: float = 0.0
    decode_ms_per_seq: float = 0.0
    ctx_ms_per_ktoken: float = 0.0
    deterministic_length: bool = True

    _COEFFICIENTS: ClassVar[tuple[str, ...]] = (
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
            raise ValueError(
                f"Unknown latency config keys: {sorted(unknown)}"
            )
        config = cls(**d)
        config.validate()
        return config

    def validate(self) -> None:
        for name in self._COEFFICIENTS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(
                    f"latency.{name} must be >= 0, got {value}"
                )


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

    @classmethod
    def from_dict(cls, d: dict) -> "SimulatedLatencyModel":
        return cls(LatencyConfig.from_dict(d))


_REGISTRY: dict[str, type] = {
    "linear": SimulatedLatencyModel,
}


def build_latency_model(d: dict) -> LatencyModel:
    d = dict(d)  # don't mutate caller's dict
    model_type = d.pop("type", "linear")
    cls = _REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown latency model type: {model_type!r}. "
            f"Known types: {sorted(_REGISTRY)}"
        )
    return cls.from_dict(d)


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

- [ ] **Step 4: Run tests to verify all pass**

```bash
.venv/bin/python -m pytest tests/test_latency.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_simulated/latency.py tests/test_latency.py
git commit -s -m "Add LatencyModel protocol, registry, and build_latency_model factory"
```

---

### Task 2: Wire `model.py` to use `build_latency_model`

**Files:**
- Modify: `src/vllm_simulated/model.py`
- Modify: `tests/test_model.py` (rename import `_load_latency_config` → `_build_latency_model`)

**Interfaces:**
- Consumes from Task 1: `LatencyModel` (Protocol), `build_latency_model(d: dict) -> LatencyModel`
- Produces: `_build_latency_model(hf_config) -> LatencyModel` (replaces `_load_latency_config`)

- [ ] **Step 1: Update the import in `tests/test_model.py`**

Change the import at the top of `tests/test_model.py` from:

```python
from vllm_simulated.model import (
    SimulatedForCausalLM,
    _load_latency_config,
)
```

To:

```python
from vllm_simulated.model import (
    SimulatedForCausalLM,
    _build_latency_model,
)
```

Update the test body from:

```python
def test_missing_latency_block_raises():
    with pytest.raises(ValueError, match="latency"):
        _load_latency_config(types.SimpleNamespace())
    with pytest.raises(ValueError, match="latency"):
        _load_latency_config(types.SimpleNamespace(latency={}))
```

To:

```python
def test_missing_latency_block_raises():
    with pytest.raises(ValueError, match="latency"):
        _build_latency_model(types.SimpleNamespace())
    with pytest.raises(ValueError, match="latency"):
        _build_latency_model(types.SimpleNamespace(latency={}))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_model.py::test_missing_latency_block_raises -v
```

Expected: FAIL with `ImportError: cannot import name '_build_latency_model'`.

- [ ] **Step 3: Update `src/vllm_simulated/model.py`**

The complete updated file:

```python
import time

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention

from vllm_simulated.latency import (
    LatencyModel,
    batch_shape_from_attn_metadata,
    build_latency_model,
)


def _build_latency_model(hf_config) -> LatencyModel:
    """Load and build a LatencyModel from hf_config, raising on missing block."""
    latency = getattr(hf_config, "latency", None)
    if not latency:
        raise ValueError(
            "SimulatedForCausalLM requires a non-empty 'latency' block in "
            "the model config; none was found. See the plugin README."
        )
    return build_latency_model(latency)


class SimulatedForCausalLM(nn.Module):
    """Simulated causal LM that emits random tokens and sleeps per step."""

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
        # scheduler/KV-cache manager allocates blocks like a real transformer.
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

        self.latency = _build_latency_model(hf_config)
        self.deterministic_length = getattr(hf_config, "latency", {}).get(
            "deterministic_length", True
        )
        self.eos_token_id = getattr(hf_config, "eos_token_id", None)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
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
        for _ in weights:
            pass
        return set()

    @staticmethod
    def _current_attn_metadata():
        attn_metadata = get_forward_context().attn_metadata
        if not attn_metadata:
            return None
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0]
        return next(iter(attn_metadata.values()), None)
```

- [ ] **Step 4: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS (including `test_latency.py`, `test_model.py`, `test_end_to_end.py`). The e2e tests require vLLM installed — if skipped due to missing vLLM, that is acceptable; verify `test_latency.py` and `test_model.py` both pass.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_simulated/model.py tests/test_model.py
git commit -s -m "Wire model.py to use build_latency_model factory"
```

---

## Self-Review

**Spec coverage:**
- §3.1 `LatencyModel` Protocol → Task 1 Step 3. ✓
- §3.2 `SimulatedLatencyModel.from_dict` classmethod → Task 1 Step 3. ✓
- §3.3 `_REGISTRY` + `build_latency_model` factory (pops `"type"`, error on unknown) → Task 1 Step 3. ✓
- §4.1 `_build_latency_model` replacing `_load_latency_config` → Task 2 Step 3. ✓
- §4.2 `__init__` wiring + `deterministic_length` from raw dict → Task 2 Step 3. ✓
- §4.3 Import changes (drop `LatencyConfig`/`SimulatedLatencyModel`, add `LatencyModel`/`build_latency_model`) → Task 2 Step 3. ✓
- §5 `"type"` key backward compatibility (default `"linear"`, not passed to `LatencyConfig.from_dict`) → Task 1 Step 3 (`d.pop("type", "linear")`). ✓
- §6 Four new unit tests → Task 1 Steps 1–4. ✓
- `test_model.py` import update (`_load_latency_config` → `_build_latency_model`) → Task 2 Steps 1–2. ✓

**Placeholder scan:** No TBD/TODO. All code blocks are complete. ✓

**Type consistency:**
- `build_latency_model` defined in Task 1, imported in Task 2. ✓
- `LatencyModel` Protocol defined in Task 1, used as return type annotation in Task 2. ✓
- `_build_latency_model` defined and exported in Task 2; `test_model.py` imports match. ✓
- `from_dict` classmethod defined on `SimulatedLatencyModel` in Task 1; called by factory in same task. ✓
