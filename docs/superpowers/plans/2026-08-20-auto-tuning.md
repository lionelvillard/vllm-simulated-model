# Auto-Tuning Physics Beta Parameters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a coordinate-search tuner that automatically fits `beta = [beta_pf, beta_dc, beta_base]` for the physics latency model by running the sim benchmark multiple times via an in-place config update API endpoint.

**Architecture:** A new `POST /sim/config` endpoint plugin (vLLM `endpoint_plugins`) writes new beta values to a file; `SimulatedForCausalLM.forward()` detects the change via `os.stat` mtime and hot-reloads the latency model without restarting the pod. `evaluation/tune.py` runs the real benchmark once, then uses `scipy.optimize.minimize_scalar` (Brent's method, bounded) in three sequential 1-D phases — one per beta coefficient — to minimize per-metric MAPE.

**Tech Stack:** Python 3.12, FastAPI (via vLLM), `scipy.optimize.minimize_scalar`, `starlette.testclient` (tests), existing `evaluation.run_eval` helpers.

**Spec:** `docs/superpowers/specs/2026-08-20-auto-tuning-design.md`

## Global Constraints

- Python ≥ 3.12 (worktree `.venv` via `uv venv --python 3.12`)
- Activate venv before every `pytest` or `python -m` invocation: `source .venv/bin/activate`
- Never use the system `pytest`
- `VLLM_SIM_TUNER=1` env var must be set for the endpoint to register routes
- `VLLM_SIM_CONFIG_PATH` must point to a writable file path (e.g. `/tmp/vllm_sim_config.json`)
- `scipy` is an `eval` extra, not a core dependency — add it to `[project.optional-dependencies] eval`
- The `POST /sim/config` endpoint is guarded: if `VLLM_SIM_TUNER` is not set, `attach_router` registers no routes
- All commits must include `Signed-off-by: Lionel Villard <villard@us.ibm.com>`; no Co-Authored-By or AI-attribution lines

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `src/vllm_simulated/model.py` | **modify** | Add `_config_path`, `_config_mtime`, `_hf_config`; mtime check in `forward()`; bootstrap config file on init |
| `src/vllm_simulated/tuner_api.py` | **create** | `SimTunerEndpointPlugin` — `POST /sim/config` route |
| `pyproject.toml` | **modify** | Add `vllm.endpoint_plugins` entry point; add `scipy` to `eval` extras |
| `evaluation/tune.py` | **create** | Coordinate-search tuner driver |
| `tests/test_model_reload.py` | **create** | Unit tests for `_load_latency_from_file` and mtime check |
| `tests/test_tuner_api.py` | **create** | Unit tests for `POST /sim/config` endpoint |
| `tests/test_tune.py` | **create** | Unit tests for `bench_sim` wrapper and coordinate search |

---

## Task 1: Model hot-reload

**Files:**
- Modify: `src/vllm_simulated/model.py`
- Create: `tests/test_model_reload.py`

**Interfaces:**
- Produces:
  - `_load_latency_from_file(config_path: str, hf_config) -> LatencyModel` — module-level function; reads `config_path` as JSON, returns `build_latency_model(data["latency"], hf_config=hf_config)`
  - `_bootstrap_config_file(config_path: str, hf_config) -> None` — writes `{"latency": dict(hf_config.latency)}` to `config_path` only if the file does not already exist
  - `SimulatedForCausalLM` now stores `_hf_config`, `_config_path`, `_config_mtime` and calls `_maybe_reload_config()` at the top of `forward()`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_reload.py
import json
import os
import types
from pathlib import Path

import pytest

from vllm_simulated.latency import build_latency_model
from vllm_simulated.model import _bootstrap_config_file, _load_latency_from_file


def _make_hf_config(beta):
    """Minimal hf_config stand-in with a latency block."""
    return types.SimpleNamespace(
        latency={
            "type": "physics",
            "hardware": {
                "peak_tflops": 989.0,
                "hbm_gbps": 3350.0,
                "weight_dtype": "bfloat16",
            },
            "beta": beta,
            "tp": 1,
            "deterministic_length": True,
        },
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        vocab_size=256,
    )


@pytest.fixture()
def config_path(tmp_path):
    return str(tmp_path / "sim-config.json")


def test_load_latency_from_file_returns_model_with_correct_beta(config_path):
    hf = _make_hf_config([0.5, 0.8, 10.0])
    cfg = {"latency": dict(hf.latency)}
    Path(config_path).write_text(json.dumps(cfg))

    model = _load_latency_from_file(config_path, hf)

    # PhysicsLatencyModel stores beta on its config attribute
    assert tuple(model.config.beta) == (0.5, 0.8, 10.0)


def test_bootstrap_config_file_creates_file_if_absent(tmp_path, config_path):
    hf = _make_hf_config([1.0, 1.0, 0.0])
    _bootstrap_config_file(config_path, hf)

    data = json.loads(Path(config_path).read_text())
    assert data["latency"]["beta"] == [1.0, 1.0, 0.0]


def test_bootstrap_config_file_does_not_overwrite_existing(config_path):
    Path(config_path).write_text(json.dumps({"latency": {"beta": [9, 9, 9]}}))
    hf = _make_hf_config([1.0, 1.0, 0.0])
    _bootstrap_config_file(config_path, hf)

    data = json.loads(Path(config_path).read_text())
    assert data["latency"]["beta"] == [9, 9, 9]  # unchanged


def test_load_latency_from_file_picks_up_updated_beta(config_path):
    hf = _make_hf_config([1.0, 1.0, 0.0])
    Path(config_path).write_text(json.dumps({"latency": dict(hf.latency)}))

    # Update the file with a new beta
    new_cfg = {"latency": {**dict(hf.latency), "beta": [2.0, 0.5, 7.0]}}
    Path(config_path).write_text(json.dumps(new_cfg))

    model = _load_latency_from_file(config_path, hf)
    assert tuple(model.config.beta) == (2.0, 0.5, 7.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_model_reload.py -v
```

Expected: `ImportError` — `_bootstrap_config_file` and `_load_latency_from_file` don't exist yet.

- [ ] **Step 3: Add the two module-level functions and modify `__init__` / `forward`**

At the top of `src/vllm_simulated/model.py`, add:

```python
import json
import os
from pathlib import Path
```

After the existing imports and before `_build_latency_model`, add:

```python
def _load_latency_from_file(config_path: str, hf_config) -> "LatencyModel":
    with open(config_path) as f:
        data = json.load(f)
    return build_latency_model(data["latency"], hf_config=hf_config)


def _bootstrap_config_file(config_path: str, hf_config) -> None:
    p = Path(config_path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    latency = getattr(hf_config, "latency", {})
    p.write_text(json.dumps({"latency": dict(latency)}))
```

In `SimulatedForCausalLM.__init__`, after `self.latency = _build_latency_model(hf_config)`, add:

```python
self._hf_config = hf_config
self._config_path: str | None = os.environ.get("VLLM_SIM_CONFIG_PATH")
self._config_mtime: int = 0
if self._config_path:
    _bootstrap_config_file(self._config_path, hf_config)
```

In `SimulatedForCausalLM.forward()`, at the very top of the method body (before the existing `if inputs_embeds` block), add:

```python
if self._config_path:
    mtime = os.stat(self._config_path).st_mtime_ns
    if mtime != self._config_mtime:
        self.latency = _load_latency_from_file(
            self._config_path, self._hf_config
        )
        self._config_mtime = mtime
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_model_reload.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
source .venv/bin/activate
pytest tests/ -v --ignore=tests/test_end_to_end.py
```

Expected: all existing tests continue to pass.

- [ ] **Step 6: Commit**

```bash
git add src/vllm_simulated/model.py tests/test_model_reload.py
git commit -m "$(cat <<'EOF'
feat: hot-reload latency config via mtime check in forward()

Adds _load_latency_from_file and _bootstrap_config_file helpers plus
a per-call os.stat mtime check in SimulatedForCausalLM.forward() so
the physics latency model can be updated in-place without restarting
the pod. Activated by VLLM_SIM_CONFIG_PATH env var.

Signed-off-by: Lionel Villard <villard@us.ibm.com>
EOF
)"
```

---

## Task 2: `POST /sim/config` endpoint plugin

**Files:**
- Create: `src/vllm_simulated/tuner_api.py`
- Create: `tests/test_tuner_api.py`

**Interfaces:**
- Consumes: `VLLM_SIM_TUNER` and `VLLM_SIM_CONFIG_PATH` env vars; `_load_latency_from_file` (Task 1) — the endpoint only does file I/O, not model state; `VLLM_SIM_CONFIG_PATH` file must already exist (bootstrapped by the model on startup)
- Produces:
  - `SimTunerEndpointPlugin` — class with `name = "sim_tuner"`, `required_tasks = None`, `attach_router(app: FastAPI)`, `async init_state(engine_client, state, args)`
  - `create_plugin() -> SimTunerEndpointPlugin` — zero-arg factory registered as the entry point
  - `POST /sim/config` accepts `{"beta": [float, float, float]}`, validates, writes atomically, returns `{"status": "ok", "beta": [...]}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tuner_api.py
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

INITIAL_CONFIG = {
    "latency": {
        "type": "physics",
        "hardware": {
            "peak_tflops": 989.0,
            "hbm_gbps": 3350.0,
            "weight_dtype": "bfloat16",
        },
        "beta": [1.0, 1.0, 0.0],
        "tp": 1,
        "deterministic_length": True,
    }
}


@pytest.fixture()
def config_file(tmp_path):
    p = tmp_path / "sim-config.json"
    p.write_text(json.dumps(INITIAL_CONFIG))
    return p


@pytest.fixture()
def client(config_file, monkeypatch):
    monkeypatch.setenv("VLLM_SIM_TUNER", "1")
    monkeypatch.setenv("VLLM_SIM_CONFIG_PATH", str(config_file))
    # Import after env vars are set so attach_router sees them
    from vllm_simulated.tuner_api import SimTunerEndpointPlugin
    app = FastAPI()
    plugin = SimTunerEndpointPlugin()
    plugin.attach_router(app)
    return TestClient(app)


def test_post_sim_config_updates_beta(client, config_file):
    resp = client.post("/sim/config", json={"beta": [0.15, 0.9, 5.0]})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "beta": [0.15, 0.9, 5.0]}
    data = json.loads(config_file.read_text())
    assert data["latency"]["beta"] == [0.15, 0.9, 5.0]


def test_post_sim_config_rejects_wrong_length(client):
    resp = client.post("/sim/config", json={"beta": [1.0, 2.0]})
    assert resp.status_code == 422


def test_post_sim_config_rejects_negative_values(client):
    resp = client.post("/sim/config", json={"beta": [1.0, -0.1, 0.0]})
    assert resp.status_code == 422


def test_no_route_when_tuner_disabled(config_file, monkeypatch):
    monkeypatch.delenv("VLLM_SIM_TUNER", raising=False)
    monkeypatch.setenv("VLLM_SIM_CONFIG_PATH", str(config_file))
    from vllm_simulated.tuner_api import SimTunerEndpointPlugin
    app = FastAPI()
    plugin = SimTunerEndpointPlugin()
    plugin.attach_router(app)
    c = TestClient(app)
    resp = c.post("/sim/config", json={"beta": [1.0, 1.0, 0.0]})
    assert resp.status_code == 404


def test_atomic_write_does_not_corrupt_file(client, config_file):
    resp = client.post("/sim/config", json={"beta": [0.3, 0.7, 12.5]})
    assert resp.status_code == 200
    # File must be valid JSON after the write
    data = json.loads(config_file.read_text())
    assert isinstance(data["latency"]["beta"], list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_tuner_api.py -v
```

Expected: `ImportError` — `vllm_simulated.tuner_api` doesn't exist yet.

- [ ] **Step 3: Implement `tuner_api.py`**

```python
# src/vllm_simulated/tuner_api.py
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


class BetaUpdate(BaseModel):
    beta: Annotated[list[float], Field(min_length=3, max_length=3)]

    @field_validator("beta")
    @classmethod
    def non_negative(cls, v: list[float]) -> list[float]:
        if any(b < 0 for b in v):
            raise ValueError("beta values must be >= 0")
        return v


class SimTunerEndpointPlugin:
    name = "sim_tuner"
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        if not os.environ.get("VLLM_SIM_TUNER"):
            return

        router = APIRouter()

        @router.post("/sim/config")
        def update_config(update: BetaUpdate) -> dict:
            config_path = os.environ.get("VLLM_SIM_CONFIG_PATH")
            if not config_path:
                raise HTTPException(
                    status_code=503,
                    detail="VLLM_SIM_CONFIG_PATH is not set",
                )
            p = Path(config_path)
            if not p.exists():
                raise HTTPException(
                    status_code=503,
                    detail="Config file not yet available; sim may still be starting",
                )
            data = json.loads(p.read_text())
            data["latency"]["beta"] = update.beta
            # Atomic write: write to tmp in same dir, then os.replace
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=p.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(data, tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, config_path)
            return {"status": "ok", "beta": update.beta}

        app.include_router(router)

    async def init_state(self, engine_client, state, args) -> None:
        pass  # no engine interaction needed


def create_plugin() -> SimTunerEndpointPlugin:
    return SimTunerEndpointPlugin()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_tuner_api.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
source .venv/bin/activate
pytest tests/ -v --ignore=tests/test_end_to_end.py
```

Expected: all existing tests continue to pass.

- [ ] **Step 6: Commit**

```bash
git add src/vllm_simulated/tuner_api.py tests/test_tuner_api.py
git commit -m "$(cat <<'EOF'
feat: add POST /sim/config endpoint plugin for in-place beta updates

SimTunerEndpointPlugin registers a /sim/config route on the vLLM
FastAPI app when VLLM_SIM_TUNER=1. Writes new beta values atomically
to VLLM_SIM_CONFIG_PATH (os.replace). No-op when VLLM_SIM_TUNER is
unset, so production deployments are unaffected.

Signed-off-by: Lionel Villard <villard@us.ibm.com>
EOF
)"
```

---

## Task 3: Package wiring (`pyproject.toml`)

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `create_plugin` from Task 2
- Produces: `vllm.endpoint_plugins` entry point `sim_tuner`; `scipy` available when `eval` extras are installed

- [ ] **Step 1: Update `pyproject.toml`**

In `pyproject.toml`, add a new entry-points table after `[project.entry-points."vllm.general_plugins"]`:

```toml
[project.entry-points."vllm.endpoint_plugins"]
sim_tuner = "vllm_simulated.tuner_api:create_plugin"
```

Change the `eval` extras line from:

```toml
eval = ["pyyaml"]
```

to:

```toml
eval = ["pyyaml", "scipy"]
```

- [ ] **Step 2: Re-install the package so the entry point is registered**

```bash
source .venv/bin/activate
uv pip install -e ".[test]"
```

- [ ] **Step 3: Verify the entry point is discoverable**

```bash
source .venv/bin/activate
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.endpoint_plugins')
names = [ep.name for ep in eps]
assert 'sim_tuner' in names, f'sim_tuner not found; got {names}'
print('OK:', names)
"
```

Expected: prints `OK: ['sim_tuner']` (possibly alongside other plugins).

- [ ] **Step 4: Run the full test suite**

```bash
source .venv/bin/activate
pytest tests/ -v --ignore=tests/test_end_to_end.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
chore: register sim_tuner endpoint plugin and add scipy to eval extras

Signed-off-by: Lionel Villard <villard@us.ibm.com>
EOF
)"
```

---

## Task 4: Tuner driver (`evaluation/tune.py`)

**Files:**
- Create: `evaluation/tune.py`
- Create: `tests/test_tune.py`

**Interfaces:**
- Consumes:
  - `evaluation.run_eval.bench_argv(point, *, base_url, model, tokenizer, result_dir, result_filename, seed) -> list[str]`
  - `evaluation.run_eval._detect_model(base_url: str) -> str`
  - `evaluation.run_eval._run_bench(argv: list[str]) -> None`
  - `evaluation.run_eval.SweepPoint(isl, osl, concurrency, num_prompts)`
  - `evaluation.compare.load_result(path) -> dict`
  - `evaluation.compare.compare_point(real: dict, sim: dict) -> list[MetricComparison]`
  - `evaluation.compare.aggregate(points) -> dict` — returns `{"TTFT mean": float, ..., "overall": float}` where values are median APE as fraction
  - `evaluation.gen_sim_config.build_sim_config(real_config, *, hardware, tp, beta, deterministic_length) -> dict`
  - `POST sim_url/sim/config {"beta": [...]}` via `urllib.request`

- Produces:
  - `tune(*, real_url, sim_url, model_config, tokenizer, out_dir, seed=0) -> dict` — runs full coordinate search, returns `{"beta": [best_pf, best_dc, best_base]}`
  - `main(argv=None) -> None` — CLI entry point

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tune.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.compare import MetricComparison, PointResult


def _make_comparison(ttft_ape, itl_ape):
    """Build a minimal list of MetricComparison objects for mocking."""
    return [
        MetricComparison(
            name="TTFT mean", unit="ms", real=100.0,
            sim=100.0 * (1 + ttft_ape), ape=abs(ttft_ape),
            signed_pct=ttft_ape * 100, lower_is_better=True,
        ),
        MetricComparison(
            name="ITL mean", unit="ms", real=20.0,
            sim=20.0 * (1 + itl_ape), ape=abs(itl_ape),
            signed_pct=itl_ape * 100, lower_is_better=True,
        ),
    ]


def test_extract_ttft_mape():
    from evaluation.tune import _ttft_mape, _itl_mape

    comps = _make_comparison(ttft_ape=0.12, itl_ape=0.05)
    assert _ttft_mape(comps) == pytest.approx(0.12)
    assert _itl_mape(comps) == pytest.approx(0.05)


def test_coordinate_search_finds_minimum(tmp_path):
    """Coordinate search must find beta values that minimize each phase's objective."""
    from evaluation.tune import _coordinate_search

    calls = []

    def fake_bench_sim(beta):
        calls.append(list(beta))
        b_pf, b_dc, b_base = beta
        # True minimum: beta_pf=0.3, beta_dc=0.7, beta_base=8.0
        ttft_ape = abs(b_pf - 0.3)
        itl_ape = abs(b_dc - 0.7)
        overall = (abs(b_pf - 0.3) + abs(b_dc - 0.7) + abs(b_base - 8.0)) / 3
        return _make_comparison(ttft_ape, itl_ape)

    def fake_overall(beta):
        b_pf, b_dc, b_base = beta
        return abs(b_pf - 0.3) + abs(b_dc - 0.7) + abs(b_base - 8.0)

    result = _coordinate_search(
        bench_sim=fake_bench_sim,
        bench_sim_overall=fake_overall,
    )

    assert result["beta"][0] == pytest.approx(0.3, abs=0.02)
    assert result["beta"][1] == pytest.approx(0.7, abs=0.02)
    assert result["beta"][2] == pytest.approx(8.0, abs=0.5)


def test_write_tuned_config(tmp_path):
    from evaluation.tune import _write_tuned_config

    original_cfg = {
        "architectures": ["SimulatedForCausalLM"],
        "latency": {
            "type": "physics",
            "hardware": {"peak_tflops": 989.0, "hbm_gbps": 3350.0, "weight_dtype": "bfloat16"},
            "beta": [1.0, 1.0, 0.0],
            "tp": 1,
            "deterministic_length": True,
        },
    }
    model_config_path = tmp_path / "sim-config.json"
    model_config_path.write_text(json.dumps(original_cfg))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_tuned_config(
        model_config=str(model_config_path),
        beta=[0.15, 0.92, 6.3],
        out_dir=str(out_dir),
    )

    result = json.loads((out_dir / "tuned-sim-config.json").read_text())
    assert result["latency"]["beta"] == [0.15, 0.92, 6.3]
    # Architecture block must be preserved
    assert result["architectures"] == ["SimulatedForCausalLM"]
    assert result["latency"]["tp"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_tune.py -v
```

Expected: `ImportError` — `evaluation.tune` doesn't exist yet.

- [ ] **Step 3: Implement `evaluation/tune.py`**

```python
# evaluation/tune.py
import argparse
import json
import time
import urllib.request
from pathlib import Path

from scipy.optimize import minimize_scalar

from evaluation.compare import (
    MetricComparison,
    aggregate,
    compare_point,
    load_result,
    render_json,
    render_markdown,
    PointResult,
)
from evaluation.run_eval import (
    SweepPoint,
    _detect_model,
    _run_bench,
    bench_argv,
)

_TUNING_POINT = SweepPoint(
    isl=1024, osl=128, concurrency=1, num_prompts=32
)

_PF_BOUNDS = (0.05, 5.0)
_DC_BOUNDS = (0.05, 5.0)
_BASE_BOUNDS = (0.0, 200.0)


def _ttft_mape(comparisons: list[MetricComparison]) -> float:
    return next(c.ape for c in comparisons if c.name == "TTFT mean")


def _itl_mape(comparisons: list[MetricComparison]) -> float:
    return next(c.ape for c in comparisons if c.name == "ITL mean")


def _post_beta(sim_url: str, beta: list[float]) -> None:
    body = json.dumps({"beta": beta}).encode()
    req = urllib.request.Request(
        f"{sim_url}/sim/config",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"POST /sim/config returned {resp.status}: {resp.read()}"
            )


def _make_bench_sim(
    *,
    sim_url: str,
    sim_model: str,
    tokenizer: str,
    real_result: dict,
    out_dir: Path,
    seed: int,
):
    counter = [0]

    def bench_sim(beta: list[float]) -> list[MetricComparison]:
        counter[0] += 1
        n = counter[0]
        _post_beta(sim_url, beta)
        time.sleep(0.5)  # barrier: let any in-flight requests drain
        fname = f"sim-tune-{n}.json"
        argv = bench_argv(
            _TUNING_POINT,
            base_url=sim_url,
            model=sim_model,
            tokenizer=tokenizer,
            result_dir=str(out_dir),
            result_filename=fname,
            seed=seed,
        )
        _run_bench(argv)
        sim_result = load_result(out_dir / fname)
        return compare_point(real_result, sim_result)

    return bench_sim


def _coordinate_search(
    *,
    bench_sim,
    bench_sim_overall=None,  # override for testing
) -> dict:
    history = {"phase1": [], "phase2": [], "phase3": []}

    # Phase 1: tune beta_pf — minimize TTFT mean APE
    def f_pf(b):
        comps = bench_sim([b, 1.0, 0.0])
        mape = _ttft_mape(comps)
        history["phase1"].append({"beta": [b, 1.0, 0.0], "mape": mape})
        return mape

    res_pf = minimize_scalar(f_pf, bounds=_PF_BOUNDS, method="bounded")
    best_pf = res_pf.x

    # Phase 2: tune beta_dc — minimize ITL mean APE
    def f_dc(b):
        comps = bench_sim([best_pf, b, 0.0])
        mape = _itl_mape(comps)
        history["phase2"].append({"beta": [best_pf, b, 0.0], "mape": mape})
        return mape

    res_dc = minimize_scalar(f_dc, bounds=_DC_BOUNDS, method="bounded")
    best_dc = res_dc.x

    # Phase 3: tune beta_base — minimize overall median MAPE
    def f_base(b):
        if bench_sim_overall is not None:
            mape = bench_sim_overall([best_pf, best_dc, b])
        else:
            comps = bench_sim([best_pf, best_dc, b])
            pt = PointResult(
                label="tune",
                params={},
                comparisons=comps,
            )
            mape = aggregate([pt])["overall"]
        history["phase3"].append(
            {"beta": [best_pf, best_dc, b], "mape": mape}
        )
        return mape

    res_base = minimize_scalar(f_base, bounds=_BASE_BOUNDS, method="bounded")
    best_base = res_base.x

    return {
        "beta": [best_pf, best_dc, best_base],
        "history": history,
    }


def _write_tuned_config(
    *, model_config: str, beta: list[float], out_dir: str
) -> None:
    with open(model_config) as f:
        cfg = json.load(f)
    cfg["latency"]["beta"] = beta
    out = Path(out_dir) / "tuned-sim-config.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")


def _write_tuning_report(
    *,
    history: dict,
    best_beta: list[float],
    real_result: dict,
    final_sim_result: dict,
    out_dir: Path,
) -> None:
    comps = compare_point(real_result, final_sim_result)
    pt = PointResult(
        label=f"ISL=1024 OSL=128 c=1 (tuned beta={best_beta})",
        params={"isl": 1024, "osl": 128, "concurrency": 1},
        comparisons=comps,
    )
    agg = aggregate([pt])

    lines = ["# Auto-Tuning Report", ""]
    lines.append(f"**Tuned beta:** `{best_beta}`")
    lines.append("")
    for phase, name in [("phase1", "beta_pf"), ("phase2", "beta_dc"), ("phase3", "beta_base")]:
        lines.append(f"## Phase: {name}")
        lines.append("")
        lines.append("| beta | MAPE |")
        lines.append("|---:|---:|")
        for step in history[phase]:
            lines.append(
                f"| {step['beta']} | {step['mape'] * 100:.2f}% |"
            )
        lines.append("")

    lines.append("## Final comparison at tuned beta")
    lines.append("")
    lines += render_markdown([pt], agg).splitlines()

    (out_dir / "tuning-report.md").write_text("\n".join(lines))
    report_json = {
        "best_beta": best_beta,
        "history": history,
        "final_comparison": render_json([pt], agg),
    }
    (out_dir / "tuning-report.json").write_text(
        json.dumps(report_json, indent=2)
    )
    print(f"wrote {out_dir / 'tuning-report.md'} and {out_dir / 'tuning-report.json'}")


def tune(
    *,
    real_url: str,
    sim_url: str,
    model_config: str,
    tokenizer: str,
    out_dir: str,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    real_model = _detect_model(real_url)
    sim_model = _detect_model(sim_url)
    print(f"real: {real_url}  model={real_model}")
    print(f"sim:  {sim_url}  model={sim_model}")

    # Run real benchmark once
    real_fname = "real.json"
    _run_bench(
        bench_argv(
            _TUNING_POINT,
            base_url=real_url,
            model=real_model,
            tokenizer=tokenizer,
            result_dir=str(out),
            result_filename=real_fname,
            seed=seed,
        )
    )
    real_result = load_result(out / real_fname)
    print("real benchmark done")

    bench_sim = _make_bench_sim(
        sim_url=sim_url,
        sim_model=sim_model,
        tokenizer=tokenizer,
        real_result=real_result,
        out_dir=out,
        seed=seed,
    )

    result = _coordinate_search(bench_sim=bench_sim)
    best_beta = [round(b, 6) for b in result["beta"]]
    print(f"tuned beta: {best_beta}")

    # Run one final benchmark at the tuned beta to record the final comparison
    _post_beta(sim_url, best_beta)
    time.sleep(0.5)
    final_fname = "sim-final.json"
    _run_bench(
        bench_argv(
            _TUNING_POINT,
            base_url=sim_url,
            model=sim_model,
            tokenizer=tokenizer,
            result_dir=str(out),
            result_filename=final_fname,
            seed=seed,
        )
    )
    final_sim_result = load_result(out / final_fname)

    _write_tuned_config(model_config=model_config, beta=best_beta, out_dir=str(out))
    _write_tuning_report(
        history=result["history"],
        best_beta=best_beta,
        real_result=real_result,
        final_sim_result=final_sim_result,
        out_dir=out,
    )
    return {"beta": best_beta}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Auto-tune physics beta parameters via coordinate search."
    )
    ap.add_argument("--real-url", required=True)
    ap.add_argument("--sim-url", required=True)
    ap.add_argument(
        "--model-config", required=True,
        help="Path to the sim-config.json whose beta will be tuned.",
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", default="tune-out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tune(
        real_url=args.real_url,
        sim_url=args.sim_url,
        model_config=args.model_config,
        tokenizer=args.tokenizer,
        out_dir=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/test_tune.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
source .venv/bin/activate
pytest tests/ -v --ignore=tests/test_end_to_end.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/tune.py tests/test_tune.py
git commit -m "$(cat <<'EOF'
feat: add auto-tuning driver for physics beta parameters

evaluation/tune.py: coordinate search over beta_pf, beta_dc, beta_base
using scipy.optimize.minimize_scalar (Brent bounded). Runs real
benchmark once; updates sim config via POST /sim/config for each
evaluation. Writes tuned-sim-config.json and tuning-report.md.

Signed-off-by: Lionel Villard <villard@us.ibm.com>
EOF
)"
```
