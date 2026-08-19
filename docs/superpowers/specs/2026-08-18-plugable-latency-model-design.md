# Plugable Latency Model — Design

**Date:** 2026-08-18
**Status:** Design approved
**Type:** Architectural refactor (built-ins only; no third-party entry points)

## 1. Purpose & Scope

Introduce an abstraction boundary between "latency model" and "linear implementation of it" so that alternative formula shapes can be added later without restructuring the codebase.

**In scope:**
- `LatencyModel` Protocol defining the single method contract
- Explicit `_REGISTRY` dict mapping type names to classes
- `build_latency_model(d: dict) -> LatencyModel` factory replacing direct construction
- Backward-compatible `"type"` key in the `latency` config block (default `"linear"`)
- `SimulatedLatencyModel.from_dict` classmethod as the registry construction convention

**Explicitly out of scope:**
- Third-party / entry-point plugin loading (Approach B from the brainstorm)
- Any new latency formula shapes beyond the existing `"linear"` model
- Changes to `LatencyConfig`, `BatchShape`, or the linear formula itself
- Changes to `config.json` example files (they remain valid as-is)

## 2. Architecture

Only two files change: `latency.py` and `model.py`. No new files.

### Extension pattern

To add a new formula shape:
1. Write a class with `step_time_ms(self, shape: BatchShape) -> float` and `from_dict(cls, d: dict) -> Self`
2. Add one entry to `_REGISTRY` in `latency.py`
3. Users select it via `"type": "your-name"` in their `latency` config block

## 3. `latency.py` changes

### 3.1 `LatencyModel` Protocol

```python
from typing import Protocol

class LatencyModel(Protocol):
    def step_time_ms(self, shape: BatchShape) -> float: ...
```

Structural — existing `SimulatedLatencyModel` satisfies it without any inheritance change.

### 3.2 `SimulatedLatencyModel.from_dict` classmethod

```python
@classmethod
def from_dict(cls, d: dict) -> "SimulatedLatencyModel":
    return cls(LatencyConfig.from_dict(d))
```

This is the registry construction convention: every class in `_REGISTRY` must implement `from_dict(cls, d: dict)`. Not enforced by the Protocol (it's a construction-time contract, not a runtime one); documented here.

### 3.3 Registry and factory

```python
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
```

`"type"` is popped before `cls.from_dict` so it never reaches `LatencyConfig.from_dict`'s unknown-key validation.

## 4. `model.py` changes

### 4.1 Replace `_load_latency_config`

The helper function is renamed and its return type changes from `LatencyConfig` to `LatencyModel`:

```python
def _build_latency_model(hf_config) -> LatencyModel:
    latency = getattr(hf_config, "latency", None)
    if not latency:
        raise ValueError(
            "SimulatedForCausalLM requires a non-empty 'latency' block in "
            "the model config; none was found. See the plugin README."
        )
    return build_latency_model(latency)
```

### 4.2 `SimulatedForCausalLM.__init__` wiring

```python
# Before:
latency_config = _load_latency_config(hf_config)
self.latency = SimulatedLatencyModel(latency_config)
self.deterministic_length = latency_config.deterministic_length

# After:
self.latency = _build_latency_model(hf_config)
self.deterministic_length = getattr(hf_config, "latency", {}).get(
    "deterministic_length", True
)
```

`deterministic_length` is read directly from the raw dict because it controls EOS masking in `compute_logits` — a model-level concern independent of the latency formula.

### 4.3 Import changes

- Drop: `LatencyConfig`, `SimulatedLatencyModel`
- Add: `LatencyModel`, `build_latency_model`

## 5. Config backward compatibility

The `"type"` key inside the `latency` block is optional and defaults to `"linear"`. All existing `config.json` files are valid unchanged. No migration needed.

```json
"latency": {
  "type": "linear",
  "base_ms": 5.0,
  "prefill_ms_per_token": 0.05,
  "decode_ms_per_seq": 1.2,
  "ctx_ms_per_ktoken": 0.3,
  "deterministic_length": true
}
```

## 6. Testing

Existing unit tests (`test_latency.py`) and integration tests (`test_end_to_end.py`) continue to pass unchanged — the linear model's behavior is identical.

New unit tests to add to `test_latency.py`:

- `test_build_latency_model_linear_explicit` — `{"type": "linear", "base_ms": 1.0}` → `SimulatedLatencyModel` instance, correct `step_time_ms`
- `test_build_latency_model_default_type` — `{"base_ms": 1.0}` (no `"type"` key) → same result as explicit `"linear"`
- `test_build_latency_model_unknown_type` — `{"type": "bogus"}` → `ValueError` naming the unknown type
- `test_build_latency_model_does_not_mutate_input` — input dict is unchanged after call (the factory takes a copy)

## 7. Component contracts

- `LatencyModel` Protocol — **does**: defines the `step_time_ms` runtime contract. **Depends on**: `BatchShape`.
- `_REGISTRY` — **does**: maps type names to constructible classes. **Convention**: each class implements `from_dict(cls, d: dict)`.
- `build_latency_model` — **does**: dispatches config dict to the right class. **Depends on**: `_REGISTRY`. **Does not**: validate formula-specific fields (delegated to each class's `from_dict`).
- `SimulatedLatencyModel` — unchanged runtime behavior; gains `from_dict` as its registry entry point.
- `SimulatedForCausalLM` — unchanged runtime behavior; wired through factory instead of direct construction.
