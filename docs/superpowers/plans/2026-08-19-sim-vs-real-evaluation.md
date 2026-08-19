# Sim-vs-Real Accuracy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an automated tool that benchmarks the simulated model against a real model on a real GPU (Qwen3-32B on 1× H100, TP=1) and reports per-metric MAPE.

**Architecture:** Pure-Python comparison core (`compare.py`) parses two `vllm bench serve` result JSONs and computes error; a config generator (`gen_sim_config.py`) builds the sim's `config.json` from the real model's architecture; a driver (`run_eval.py`) runs the benchmark sweep against both endpoints; a bash wrapper (`run_eval.sh`) port-forwards the two in-cluster Services for the default laptop-driven run. Kubernetes manifests under `deploy/eval/` bring up the real (GPU) and sim (CPU) servers plus an optional in-cluster benchmark Job. The cluster-touching pieces are validated by a documented manual smoke test; the pure logic is unit-tested in CI.

**Tech Stack:** Python 3.12, PyYAML, pytest, `uv`/`.venv`, `vllm bench serve` CLI, Kubernetes/OpenShift (`oc`).

**Spec:** `docs/2026-08-19-sim-vs-real-evaluation-design.md`

## Global Constraints

- **Out-of-tree only.** Do NOT edit the vLLM source tree.
- **Python env via `uv`/`.venv`.** Never use system `python3` or bare `pip`. Use `uv venv --python 3.12` and `.venv/bin/pytest`. See `CLAUDE.md`.
- **Line length 88** for Python (ruff configured in `pyproject.toml`).
- **Evaluation tests must NOT import `vllm` or `torch`.** The `evaluation/` code is pure orchestration/analysis; keeping it vllm-free lets its tests run fast and without a GPU. `vllm` is invoked only as an external CLI subprocess at runtime, never imported.
- **No network in tests.** `gen_sim_config.py` must accept an already-loaded config dict so tests use a fixture; HF fetching is a thin CLI-only wrapper.
- **First target is fixed:** model `Qwen/Qwen3-32B`, served-model-name `qwen3-32b`, tokenizer `Qwen/Qwen3-32B`, H100 SXM5 (`peak_tflops: 989.0`, `hbm_gbps: 3350.0`, `weight_dtype: "bfloat16"`), `tp: 1`, `beta: [1.0, 1.0, 0.0]`, `deterministic_length: true`.
- **Fairness invariants (spec §5), preserved everywhere:** identical tokenizer both sides; `--ignore-eos` on both sides; identical `--seed`; identical served-model-name; warmup discarded.
- **Commits:** sign off (`Signed-off-by: Lionel Villard <villard@us.ibm.com>`); no co-author / AI-attribution lines (repo `commit` skill).

### `vllm bench serve` result-JSON contract (verified against vLLM source)

Top-level keys used: `duration`, `completed`, `total_input_tokens`, `total_output_tokens`, `request_throughput`, `output_throughput`, `total_token_throughput`.

Per-metric keys exist only for metrics named in `--percentile-metrics` and percentiles named in `--metric-percentiles`. The driver MUST pass
`--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,99`, which yields, for each `m ∈ {ttft, tpot, itl, e2el}`:
`mean_<m>_ms`, `median_<m>_ms`, `std_<m>_ms`, `p90_<m>_ms`, `p99_<m>_ms`.

(Defaults are `--percentile-metrics ttft,tpot,itl` and `--metric-percentiles 99`, i.e. no `e2el` and no `p90` — hence the explicit flags.)

### New dependency

Add `pyyaml` to the `test` extra and introduce an `eval` extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
test = ["pytest", "vllm", "pyyaml"]
eval = ["pyyaml"]
```

---

## File Structure

| File | Responsibility |
|---|---|
| `evaluation/metrics.py` | The list of metrics to compare (name → JSON key, unit, whether lower-is-better) and the `MetricComparison` dataclass. Single source of truth shared by compare + report. |
| `evaluation/compare.py` | Pure logic: load result JSON, compute per-metric APE + signed error, aggregate across points, render markdown + JSON report. |
| `evaluation/gen_sim_config.py` | Build the sim `config.json` from a real config dict; render a ConfigMap YAML; thin CLI that optionally fetches the real config from HF. |
| `evaluation/run_eval.py` | Driver: load sweep, build `vllm bench serve` argv per (point, endpoint), run subprocesses with warmup, hand results to `compare.py`, write reports. |
| `evaluation/sweep.yaml` | The benchmark matrix (data, not code). |
| `evaluation/run_eval.sh` | Wrapper: port-forward both Services, wait for health, run `run_eval.py`, tear down. |
| `evaluation/README.md` | Operator runbook (deploy, run, read report, customize, troubleshoot). |
| `deploy/eval/real-deployment.yaml`, `real-service.yaml` | Real Qwen3-32B on H100 (GPU). |
| `deploy/eval/sim-deployment.yaml`, `sim-service.yaml`, `sim-configmap.yaml` | Sim server on CPU (reuses `deploy/` pattern) + generated config. |
| `deploy/eval/benchmark-job.yaml` | Optional in-cluster driver Job. |
| `tests/test_eval_compare.py` | Unit tests for `metrics.py` + `compare.py`. |
| `tests/test_eval_gen_sim_config.py` | Unit tests for `gen_sim_config.py`. |
| `tests/test_eval_run_eval.py` | Unit tests for `run_eval.py` argv construction + sweep loading. |
| `tests/test_eval_manifests.py` | YAML-parse + key-field assertions for `deploy/eval/` manifests. |
| `tests/fixtures/eval/real-*.json`, `sim-*.json` | Sample `vllm bench serve` result JSONs. |
| `tests/fixtures/eval/qwen3-32b-config.json` | Sample real model `config.json`. |

---

## Task 1: Metrics registry + `MetricComparison`

**Files:**
- Create: `evaluation/__init__.py` (empty)
- Create: `evaluation/metrics.py`
- Test: `tests/test_eval_compare.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) Metric(name: str, key: str, unit: str, lower_is_better: bool)`
  - `METRICS: list[Metric]` — the ordered comparison set.
  - `@dataclass(frozen=True) MetricComparison(name: str, unit: str, real: float, sim: float, ape: float, signed_pct: float, lower_is_better: bool)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_compare.py
from evaluation.metrics import METRICS, Metric, MetricComparison


def test_metrics_registry_covers_spec():
    keys = {m.key for m in METRICS}
    # spec §7 report rows
    assert {
        "mean_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
        "mean_itl_ms", "mean_tpot_ms",
        "mean_e2el_ms", "p99_e2el_ms",
        "output_throughput", "request_throughput",
    } <= keys
    # every metric is well-formed
    for m in METRICS:
        assert isinstance(m, Metric)
        assert m.unit in {"ms", "tok/s", "req/s"}


def test_metric_comparison_is_frozen():
    c = MetricComparison(
        name="TTFT mean", unit="ms", real=100.0, sim=110.0,
        ape=0.1, signed_pct=10.0, lower_is_better=True,
    )
    assert c.real == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/__init__.py
```

```python
# evaluation/metrics.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str          # human label, e.g. "TTFT mean"
    key: str           # vllm bench serve JSON key, e.g. "mean_ttft_ms"
    unit: str          # "ms" | "tok/s" | "req/s"
    lower_is_better: bool


@dataclass(frozen=True)
class MetricComparison:
    name: str
    unit: str
    real: float
    sim: float
    ape: float         # |sim - real| / real, as a fraction (0.1 == 10%)
    signed_pct: float  # (sim - real) / real * 100, sign preserved
    lower_is_better: bool


METRICS: list[Metric] = [
    Metric("TTFT mean", "mean_ttft_ms", "ms", True),
    Metric("TTFT p90", "p90_ttft_ms", "ms", True),
    Metric("TTFT p99", "p99_ttft_ms", "ms", True),
    Metric("TPOT mean", "mean_tpot_ms", "ms", True),
    Metric("ITL mean", "mean_itl_ms", "ms", True),
    Metric("ITL p99", "p99_itl_ms", "ms", True),
    Metric("E2E mean", "mean_e2el_ms", "ms", True),
    Metric("E2E p99", "p99_e2el_ms", "ms", True),
    Metric("Output throughput", "output_throughput", "tok/s", False),
    Metric("Request throughput", "request_throughput", "req/s", False),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/__init__.py evaluation/metrics.py tests/test_eval_compare.py
git commit -s -m "feat(eval): add metrics registry and MetricComparison"
```

---

## Task 2: `compare.py` — load + per-point comparison

**Files:**
- Create: `evaluation/compare.py`
- Create: `tests/fixtures/eval/real-c16.json`, `tests/fixtures/eval/sim-c16.json`
- Test: `tests/test_eval_compare.py` (extend)

**Interfaces:**
- Consumes: `evaluation.metrics.METRICS`, `Metric`, `MetricComparison`.
- Produces:
  - `load_result(path: str | Path) -> dict` — read a `vllm bench serve` JSON.
  - `compare_point(real: dict, sim: dict) -> list[MetricComparison]` — one entry per `METRICS`, raising `KeyError` (loud) if a required key is absent from either dict.

- [ ] **Step 1: Write the failing test**

Create fixtures first (minimal but realistic; values chosen so APEs are easy to verify):

```json
// tests/fixtures/eval/real-c16.json
{
  "duration": 30.0, "completed": 256,
  "total_input_tokens": 262144, "total_output_tokens": 32768,
  "request_throughput": 8.0, "output_throughput": 1000.0,
  "total_token_throughput": 9000.0,
  "mean_ttft_ms": 100.0, "median_ttft_ms": 95.0, "std_ttft_ms": 10.0,
  "p90_ttft_ms": 150.0, "p99_ttft_ms": 200.0,
  "mean_tpot_ms": 20.0, "median_tpot_ms": 19.0, "std_tpot_ms": 2.0,
  "p90_tpot_ms": 24.0, "p99_tpot_ms": 30.0,
  "mean_itl_ms": 20.0, "median_itl_ms": 19.0, "std_itl_ms": 2.0,
  "p90_itl_ms": 24.0, "p99_itl_ms": 30.0,
  "mean_e2el_ms": 2000.0, "median_e2el_ms": 1950.0, "std_e2el_ms": 100.0,
  "p90_e2el_ms": 2400.0, "p99_e2el_ms": 3000.0
}
```

```json
// tests/fixtures/eval/sim-c16.json
{
  "duration": 30.0, "completed": 256,
  "total_input_tokens": 262144, "total_output_tokens": 32768,
  "request_throughput": 8.4, "output_throughput": 1100.0,
  "total_token_throughput": 9500.0,
  "mean_ttft_ms": 110.0, "median_ttft_ms": 100.0, "std_ttft_ms": 12.0,
  "p90_ttft_ms": 165.0, "p99_ttft_ms": 240.0,
  "mean_tpot_ms": 19.0, "median_tpot_ms": 18.0, "std_tpot_ms": 2.0,
  "p90_tpot_ms": 22.0, "p99_tpot_ms": 27.0,
  "mean_itl_ms": 19.0, "median_itl_ms": 18.0, "std_itl_ms": 2.0,
  "p90_itl_ms": 22.0, "p99_itl_ms": 27.0,
  "mean_e2el_ms": 1900.0, "median_e2el_ms": 1850.0, "std_e2el_ms": 95.0,
  "p90_e2el_ms": 2280.0, "p99_e2el_ms": 2850.0
}
```

```python
# tests/test_eval_compare.py (append)
from pathlib import Path
import pytest
from evaluation.compare import load_result, compare_point

FIX = Path(__file__).parent / "fixtures" / "eval"


def test_compare_point_computes_ape_and_sign():
    real = load_result(FIX / "real-c16.json")
    sim = load_result(FIX / "sim-c16.json")
    comps = {c.name: c for c in compare_point(real, sim)}
    # TTFT mean: |110-100|/100 = 0.10 ; signed +10%
    assert comps["TTFT mean"].ape == pytest.approx(0.10)
    assert comps["TTFT mean"].signed_pct == pytest.approx(10.0)
    # ITL mean: |19-20|/20 = 0.05 ; signed -5%
    assert comps["ITL mean"].ape == pytest.approx(0.05)
    assert comps["ITL mean"].signed_pct == pytest.approx(-5.0)
    # throughput present
    assert comps["Output throughput"].ape == pytest.approx(0.10)


def test_compare_point_missing_key_raises():
    real = load_result(FIX / "real-c16.json")
    sim = load_result(FIX / "sim-c16.json")
    del sim["mean_ttft_ms"]
    with pytest.raises(KeyError):
        compare_point(real, sim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.compare'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/compare.py
import json
from pathlib import Path

from evaluation.metrics import METRICS, MetricComparison


def load_result(path):
    with open(Path(path)) as f:
        return json.load(f)


def compare_point(real: dict, sim: dict) -> list[MetricComparison]:
    comps: list[MetricComparison] = []
    for m in METRICS:
        if m.key not in real:
            raise KeyError(f"metric {m.key!r} missing from real result")
        if m.key not in sim:
            raise KeyError(f"metric {m.key!r} missing from sim result")
        r = float(real[m.key])
        s = float(sim[m.key])
        ape = abs(s - r) / r if r != 0 else float("inf")
        signed = (s - r) / r * 100.0 if r != 0 else float("inf")
        comps.append(
            MetricComparison(
                name=m.name, unit=m.unit, real=r, sim=s,
                ape=ape, signed_pct=signed, lower_is_better=m.lower_is_better,
            )
        )
    return comps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/compare.py tests/fixtures/eval/real-c16.json tests/fixtures/eval/sim-c16.json tests/test_eval_compare.py
git commit -s -m "feat(eval): parse bench results and compute per-metric APE"
```

---

## Task 3: `compare.py` — aggregation + report rendering

**Files:**
- Modify: `evaluation/compare.py`
- Test: `tests/test_eval_compare.py` (extend)

**Interfaces:**
- Consumes: `compare_point`, `MetricComparison`.
- Produces:
  - `@dataclass(frozen=True) PointResult(label: str, params: dict, comparisons: list[MetricComparison])`
  - `aggregate(points: list[PointResult]) -> dict[str, float]` — median APE per metric name, plus `"overall"` = median of all per-metric medians.
  - `render_json(points: list[PointResult], agg: dict) -> dict`
  - `render_markdown(points: list[PointResult], agg: dict) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_compare.py (append)
from evaluation.compare import PointResult, aggregate, render_markdown, render_json


def _point(label):
    real = load_result(FIX / "real-c16.json")
    sim = load_result(FIX / "sim-c16.json")
    return PointResult(label=label, params={"isl": 1024, "osl": 128, "c": 16},
                       comparisons=compare_point(real, sim))


def test_aggregate_medians():
    pts = [_point("a"), _point("b")]
    agg = aggregate(pts)
    # identical points -> median equals the single-point APE
    assert agg["TTFT mean"] == pytest.approx(0.10)
    assert "overall" in agg
    assert agg["overall"] >= 0.0


def test_render_markdown_has_tables_and_signed_hint():
    pts = [_point("ISL=1024 OSL=128 c=16")]
    md = render_markdown(pts, aggregate(pts))
    assert "ISL=1024 OSL=128 c=16" in md
    assert "| TTFT mean" in md
    assert "%" in md
    assert "Aggregate" in md
    # signed hint mentions direction for TTFT/ITL
    assert "sim slower" in md or "sim faster" in md


def test_render_json_roundtrips():
    pts = [_point("p")]
    out = render_json(pts, aggregate(pts))
    assert out["aggregate"]["overall"] >= 0.0
    assert out["points"][0]["label"] == "p"
    assert out["points"][0]["metrics"][0]["ape"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_compare.py -v`
Expected: FAIL with `ImportError: cannot import name 'PointResult'`

- [ ] **Step 3: Write minimal implementation**

Append to `evaluation/compare.py`:

```python
from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True)
class PointResult:
    label: str
    params: dict
    comparisons: list  # list[MetricComparison]


def aggregate(points) -> dict:
    per_metric: dict[str, list[float]] = {}
    for p in points:
        for c in p.comparisons:
            per_metric.setdefault(c.name, []).append(c.ape)
    agg = {name: median(vals) for name, vals in per_metric.items()}
    agg["overall"] = median(agg.values()) if agg else 0.0
    return agg


def _signed_hint(c) -> str:
    if c.unit != "ms":
        direction = "sim higher" if c.signed_pct > 0 else "sim lower"
        return f"{direction}"
    if c.signed_pct > 0:
        return "sim slower"
    return "sim faster"


def render_markdown(points, agg) -> str:
    lines: list[str] = ["# Sim-vs-Real Evaluation Report", ""]
    for p in points:
        lines.append(f"### {p.label}")
        lines.append("")
        lines.append("| Metric | Real | Sim | APE | Signed |")
        lines.append("|---|--:|--:|--:|:--|")
        for c in p.comparisons:
            lines.append(
                f"| {c.name} ({c.unit}) | {c.real:.2f} | {c.sim:.2f} "
                f"| {c.ape * 100:.1f}% | {c.signed_pct:+.1f}% {_signed_hint(c)} |"
            )
        lines.append("")
    lines.append("## Aggregate (median APE across points)")
    lines.append("")
    lines.append("| Metric | Median APE |")
    lines.append("|---|--:|")
    for name, val in agg.items():
        if name == "overall":
            continue
        lines.append(f"| {name} | {val * 100:.1f}% |")
    lines.append("")
    lines.append(f"**Overall median MAPE: {agg['overall'] * 100:.1f}%**")
    lines.append("")
    return "\n".join(lines)


def render_json(points, agg) -> dict:
    return {
        "aggregate": agg,
        "points": [
            {
                "label": p.label,
                "params": p.params,
                "metrics": [
                    {
                        "name": c.name, "unit": c.unit, "real": c.real,
                        "sim": c.sim, "ape": c.ape, "signed_pct": c.signed_pct,
                    }
                    for c in p.comparisons
                ],
            }
            for p in points
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_compare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/compare.py tests/test_eval_compare.py
git commit -s -m "feat(eval): aggregate MAPE and render markdown/json reports"
```

---

## Task 4: `gen_sim_config.py` — sim config from real config

**Files:**
- Create: `evaluation/gen_sim_config.py`
- Create: `tests/fixtures/eval/qwen3-32b-config.json`
- Test: `tests/test_eval_gen_sim_config.py`

**Interfaces:**
- Consumes: nothing (pure dict transforms + optional CLI HF fetch).
- Produces:
  - `H100_SXM5: dict` = `{"peak_tflops": 989.0, "hbm_gbps": 3350.0, "weight_dtype": "bfloat16"}`
  - `build_sim_config(real_config: dict, *, hardware: dict = H100_SXM5, tp: int = 1, beta=(1.0, 1.0, 0.0), deterministic_length: bool = True) -> dict`
  - `render_configmap(sim_config: dict, name: str = "vllm-sim-model-config") -> str` — YAML text embedding `config.json`.

Behavior of `build_sim_config`: copy all architecture fields verbatim, force `architectures = ["SimulatedForCausalLM"]`, drop any pre-existing `latency` key, and add a `latency` block of `type: "physics"` with the given hardware/tp/beta/deterministic_length. Preserve `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings`, `eos_token_id`, `torch_dtype`, and any MoE fields present.

- [ ] **Step 1: Write the failing test**

Create a representative real config fixture (values are illustrative Qwen3-32B-shape; the test asserts they are copied verbatim, not their specific magnitudes):

```json
// tests/fixtures/eval/qwen3-32b-config.json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 5120,
  "num_hidden_layers": 64,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 25600,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "eos_token_id": 151645,
  "torch_dtype": "bfloat16"
}
```

```python
# tests/test_eval_gen_sim_config.py
import json
from pathlib import Path

from evaluation.gen_sim_config import build_sim_config, render_configmap, H100_SXM5

FIX = Path(__file__).parent / "fixtures" / "eval"


def _real():
    with open(FIX / "qwen3-32b-config.json") as f:
        return json.load(f)


def test_build_preserves_architecture_and_injects_physics():
    sim = build_sim_config(_real())
    # architecture copied verbatim
    assert sim["hidden_size"] == 5120
    assert sim["num_hidden_layers"] == 64
    assert sim["num_key_value_heads"] == 8
    assert sim["intermediate_size"] == 25600
    # simulated architecture registered
    assert sim["architectures"] == ["SimulatedForCausalLM"]
    # physics latency injected with H100 SXM5 defaults
    lat = sim["latency"]
    assert lat["type"] == "physics"
    assert lat["hardware"] == H100_SXM5
    assert lat["tp"] == 1
    assert lat["beta"] == [1.0, 1.0, 0.0]
    assert lat["deterministic_length"] is True


def test_build_drops_stale_latency():
    real = _real()
    real["latency"] = {"type": "linear", "base_ms": 999.0}
    sim = build_sim_config(real)
    assert sim["latency"]["type"] == "physics"


def test_render_configmap_is_valid_yaml_with_embedded_json():
    import yaml
    sim = build_sim_config(_real())
    text = render_configmap(sim)
    doc = yaml.safe_load(text)
    assert doc["kind"] == "ConfigMap"
    assert doc["metadata"]["name"] == "vllm-sim-model-config"
    embedded = json.loads(doc["data"]["config.json"])
    assert embedded["architectures"] == ["SimulatedForCausalLM"]
    assert embedded["latency"]["type"] == "physics"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_gen_sim_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.gen_sim_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/gen_sim_config.py
import argparse
import copy
import json

import yaml

H100_SXM5 = {"peak_tflops": 989.0, "hbm_gbps": 3350.0, "weight_dtype": "bfloat16"}


def build_sim_config(
    real_config: dict,
    *,
    hardware: dict = H100_SXM5,
    tp: int = 1,
    beta=(1.0, 1.0, 0.0),
    deterministic_length: bool = True,
) -> dict:
    sim = copy.deepcopy(real_config)
    sim.pop("latency", None)
    sim["architectures"] = ["SimulatedForCausalLM"]
    sim["latency"] = {
        "type": "physics",
        "hardware": dict(hardware),
        "beta": [float(b) for b in beta],
        "tp": int(tp),
        "deterministic_length": bool(deterministic_length),
    }
    return sim


def render_configmap(sim_config: dict, name: str = "vllm-sim-model-config") -> str:
    doc = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name},
        "data": {"config.json": json.dumps(sim_config, indent=2)},
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _load_real_config(model: str) -> dict:
    """CLI-only: fetch config.json from HF or read a local path."""
    import os

    if os.path.isfile(model):
        with open(model) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model, filename="config.json")
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate sim ConfigMap from a real model config.")
    ap.add_argument("--model", required=True, help="HF model id or path to config.json")
    ap.add_argument("--out", required=True, help="output ConfigMap YAML path")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--peak-tflops", type=float, default=H100_SXM5["peak_tflops"])
    ap.add_argument("--hbm-gbps", type=float, default=H100_SXM5["hbm_gbps"])
    ap.add_argument("--weight-dtype", default=H100_SXM5["weight_dtype"])
    args = ap.parse_args(argv)

    real = _load_real_config(args.model)
    hardware = {
        "peak_tflops": args.peak_tflops,
        "hbm_gbps": args.hbm_gbps,
        "weight_dtype": args.weight_dtype,
    }
    sim = build_sim_config(real, hardware=hardware, tp=args.tp)
    with open(args.out, "w") as f:
        f.write(render_configmap(sim))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_gen_sim_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/gen_sim_config.py tests/fixtures/eval/qwen3-32b-config.json tests/test_eval_gen_sim_config.py
git commit -s -m "feat(eval): generate sim ConfigMap from real model config"
```

---

## Task 5: `run_eval.py` — sweep loading + benchmark argv

**Files:**
- Create: `evaluation/run_eval.py`
- Create: `evaluation/sweep.yaml`
- Test: `tests/test_eval_run_eval.py`

**Interfaces:**
- Consumes: `evaluation.compare` (`load_result`, `compare_point`, `PointResult`, `aggregate`, `render_markdown`, `render_json`).
- Produces:
  - `@dataclass(frozen=True) SweepPoint(isl: int, osl: int, concurrency: int, num_prompts: int)` with `.label` property `"ISL={isl} OSL={osl} c={concurrency}"`.
  - `load_sweep(path) -> list[SweepPoint]`
  - `bench_argv(point, *, base_url, model, tokenizer, result_dir, result_filename, seed=0, backend="openai", endpoint="/v1/completions") -> list[str]` — the exact `vllm bench serve` argv, including `--ignore-eos` and the percentile flags.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_run_eval.py
from pathlib import Path

from evaluation.run_eval import SweepPoint, load_sweep, bench_argv


def test_load_sweep(tmp_path):
    p = tmp_path / "sweep.yaml"
    p.write_text(
        "points:\n"
        "  - {isl: 1024, osl: 128, concurrency: 16, num_prompts: 256}\n"
        "  - {isl: 256, osl: 256, concurrency: 32, num_prompts: 256}\n"
    )
    pts = load_sweep(p)
    assert len(pts) == 2
    assert pts[0] == SweepPoint(isl=1024, osl=128, concurrency=16, num_prompts=256)
    assert pts[0].label == "ISL=1024 OSL=128 c=16"


def test_bench_argv_enforces_fairness_flags():
    pt = SweepPoint(isl=1024, osl=128, concurrency=16, num_prompts=256)
    argv = bench_argv(
        pt, base_url="http://localhost:9001", model="qwen3-32b",
        tokenizer="Qwen/Qwen3-32B", result_dir="/tmp/out",
        result_filename="c16-sim.json",
    )
    joined = " ".join(argv)
    assert argv[:3] == ["vllm", "bench", "serve"]
    assert "--ignore-eos" in argv
    assert "--seed 0" in joined
    assert "--random-input-len 1024" in joined
    assert "--random-output-len 128" in joined
    assert "--max-concurrency 16" in joined
    assert "--num-prompts 256" in joined
    assert "--model qwen3-32b" in joined
    assert "--tokenizer Qwen/Qwen3-32B" in joined
    assert "--percentile-metrics ttft,tpot,itl,e2el" in joined
    assert "--metric-percentiles 90,99" in joined
    assert "--save-result" in argv
    assert "--result-filename c16-sim.json" in joined
    assert "--base-url http://localhost:9001" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.run_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluation/run_eval.py
import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from evaluation.compare import (
    PointResult, aggregate, compare_point, load_result,
    render_json, render_markdown,
)


@dataclass(frozen=True)
class SweepPoint:
    isl: int
    osl: int
    concurrency: int
    num_prompts: int

    @property
    def label(self) -> str:
        return f"ISL={self.isl} OSL={self.osl} c={self.concurrency}"


def load_sweep(path) -> list[SweepPoint]:
    with open(Path(path)) as f:
        doc = yaml.safe_load(f)
    return [
        SweepPoint(
            isl=p["isl"], osl=p["osl"],
            concurrency=p["concurrency"], num_prompts=p["num_prompts"],
        )
        for p in doc["points"]
    ]


def bench_argv(
    point: SweepPoint,
    *,
    base_url: str,
    model: str,
    tokenizer: str,
    result_dir: str,
    result_filename: str,
    seed: int = 0,
    backend: str = "openai",
    endpoint: str = "/v1/completions",
) -> list[str]:
    return [
        "vllm", "bench", "serve",
        "--backend", backend,
        "--base-url", base_url,
        "--endpoint", endpoint,
        "--model", model,
        "--tokenizer", tokenizer,
        "--dataset-name", "random",
        "--random-input-len", str(point.isl),
        "--random-output-len", str(point.osl),
        "--num-prompts", str(point.num_prompts),
        "--max-concurrency", str(point.concurrency),
        "--ignore-eos",
        "--seed", str(seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "90,99",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
    ]


def _run_bench(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def _warmup_argv(point, *, base_url, model, tokenizer, result_dir, seed):
    warm = SweepPoint(isl=point.isl, osl=point.osl, concurrency=point.concurrency,
                      num_prompts=max(1, point.concurrency))
    return bench_argv(
        warm, base_url=base_url, model=model, tokenizer=tokenizer,
        result_dir=result_dir, result_filename="warmup.json", seed=seed,
    )


def run(
    *, real_url, sim_url, model, tokenizer, sweep_path, out_dir,
    seed=0, warmup=True,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = load_sweep(sweep_path)
    results: list[PointResult] = []
    for pt in points:
        for label, url in (("real", real_url), ("sim", sim_url)):
            if warmup:
                _run_bench(_warmup_argv(
                    pt, base_url=url, model=model, tokenizer=tokenizer,
                    result_dir=str(out), seed=seed))
            fname = f"c{pt.concurrency}-isl{pt.isl}-osl{pt.osl}-{label}.json"
            _run_bench(bench_argv(
                pt, base_url=url, model=model, tokenizer=tokenizer,
                result_dir=str(out), result_filename=fname, seed=seed))
        real = load_result(out / f"c{pt.concurrency}-isl{pt.isl}-osl{pt.osl}-real.json")
        sim = load_result(out / f"c{pt.concurrency}-isl{pt.isl}-osl{pt.osl}-sim.json")
        results.append(PointResult(
            label=pt.label,
            params={"isl": pt.isl, "osl": pt.osl, "concurrency": pt.concurrency},
            comparisons=compare_point(real, sim)))
    agg = aggregate(results)
    (out / "report.md").write_text(render_markdown(results, agg))
    (out / "report.json").write_text(json.dumps(render_json(results, agg), indent=2))
    print(f"wrote {out / 'report.md'} and {out / 'report.json'}")
    print(f"overall median MAPE: {agg['overall'] * 100:.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run sim-vs-real evaluation sweep.")
    ap.add_argument("--real-url", required=True)
    ap.add_argument("--sim-url", required=True)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    ap.add_argument("--sweep", default=str(Path(__file__).parent / "sweep.yaml"))
    ap.add_argument("--out", default="eval-out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)
    run(real_url=args.real_url, sim_url=args.sim_url, model=args.model,
        tokenizer=args.tokenizer, sweep_path=args.sweep, out_dir=args.out,
        seed=args.seed, warmup=not args.no_warmup)


if __name__ == "__main__":
    main()
```

```yaml
# evaluation/sweep.yaml
# Benchmark matrix (spec §6). Edit to extend; keep points small — each runs on the real H100.
points:
  - {isl: 1024, osl: 128, concurrency: 1,  num_prompts: 32}
  - {isl: 1024, osl: 128, concurrency: 16, num_prompts: 256}
  - {isl: 1024, osl: 128, concurrency: 64, num_prompts: 512}
  - {isl: 256,  osl: 256, concurrency: 32, num_prompts: 256}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_run_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/run_eval.py evaluation/sweep.yaml tests/test_eval_run_eval.py
git commit -s -m "feat(eval): add sweep-driving benchmark runner"
```

---

## Task 6: Kubernetes manifests (`deploy/eval/`)

**Files:**
- Create: `deploy/eval/real-deployment.yaml`, `deploy/eval/real-service.yaml`
- Create: `deploy/eval/sim-deployment.yaml`, `deploy/eval/sim-service.yaml`, `deploy/eval/sim-configmap.yaml`
- Create: `deploy/eval/benchmark-job.yaml`
- Test: `tests/test_eval_manifests.py`

**Interfaces:**
- Consumes: the sim pattern from `deploy/deployment.yaml`, `deploy/configmap.yaml`, `deploy/service.yaml` (init-container plugin injection, non-root SCC cache redirection, `/dev/shm` sizing).
- Produces: manifests that a test parses to assert key fields.

Key requirements baked into the manifests:
- **`real-deployment.yaml`:** image a GPU vLLM image (e.g. `vllm/vllm-openai:latest`); command `vllm serve Qwen/Qwen3-32B --served-model-name qwen3-32b --tokenizer Qwen/Qwen3-32B --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 8192`; `resources.limits["nvidia.com/gpu"] = 1`; `HF_HOME`/`HOME`/`XDG_CACHE_HOME` set to a writable path; an `emptyDir` (sizeLimit ~80Gi) mounted at the HF cache for the ~64 GB weight download; `/health` probes with a generous `initialDelaySeconds` (e.g. 600) for download+load. `labels.app: vllm-real`.
- **`real-service.yaml`:** `metadata.name: vllm-real`, ClusterIP, port 8000 → 8000, selector `app: vllm-real`.
- **`sim-*.yaml`:** copy of the proven `deploy/` manifests but renamed `app: vllm-sim`, `--served-model-name qwen3-32b`, `--tokenizer Qwen/Qwen3-32B`, mounting `sim-configmap.yaml`. `sim-configmap.yaml` is a placeholder to be regenerated by `gen_sim_config.py`; include a header comment saying so.
- **`benchmark-job.yaml`:** a Job using a python image that `pip install`s the `eval` extra (or mounts the repo), runs `python -m evaluation.run_eval --real-url http://vllm-real:8000 --sim-url http://vllm-sim:8000 --out /out`, with `/out` an `emptyDir` (or PVC) and results echoed to logs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_manifests.py
from pathlib import Path

import yaml

DEPLOY = Path(__file__).parent.parent / "deploy" / "eval"


def _load(name):
    with open(DEPLOY / name) as f:
        return list(yaml.safe_load_all(f))[0]


def test_all_manifests_parse():
    for f in DEPLOY.glob("*.yaml"):
        with open(f) as fh:
            docs = list(yaml.safe_load_all(fh))
        assert docs and all(d.get("kind") for d in docs), f


def test_real_deployment_requests_gpu_and_correct_model():
    d = _load("real-deployment.yaml")
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    cmd = " ".join(c["command"] + c.get("args", []))
    assert "Qwen/Qwen3-32B" in cmd
    assert "--served-model-name qwen3-32b" in cmd
    assert "--tensor-parallel-size 1" in cmd


def test_sim_and_real_share_served_model_name():
    real = _load("real-deployment.yaml")
    sim = _load("sim-deployment.yaml")
    rc = " ".join(real["spec"]["template"]["spec"]["containers"][0]["command"]
                  + real["spec"]["template"]["spec"]["containers"][0].get("args", []))
    sc = " ".join(sim["spec"]["template"]["spec"]["containers"][0]["command"]
                  + sim["spec"]["template"]["spec"]["containers"][0].get("args", []))
    assert "--served-model-name qwen3-32b" in rc
    assert "--served-model-name qwen3-32b" in sc


def test_services_target_their_apps():
    assert _load("real-service.yaml")["spec"]["selector"]["app"] == "vllm-real"
    assert _load("sim-service.yaml")["spec"]["selector"]["app"] == "vllm-sim"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_manifests.py -v`
Expected: FAIL (files do not exist / `FileNotFoundError`)

- [ ] **Step 3: Write the manifests**

Create the six manifests per the "Key requirements" above. Base `sim-*.yaml` on the existing `deploy/*.yaml` (change labels to `vllm-sim`, add `--served-model-name qwen3-32b` and `--tokenizer Qwen/Qwen3-32B`). Base `real-deployment.yaml` on a standard GPU vLLM Deployment; ensure the four asserted fields are present. Generate the initial `sim-configmap.yaml` with:

```bash
.venv/bin/python -m evaluation.gen_sim_config \
  --model tests/fixtures/eval/qwen3-32b-config.json \
  --out deploy/eval/sim-configmap.yaml
```

(For the real run the operator regenerates it from the true `Qwen/Qwen3-32B` config — documented in the README. Using the fixture here keeps the committed manifest self-consistent and test-parseable.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_manifests.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deploy/eval tests/test_eval_manifests.py
git commit -s -m "feat(eval): add k8s manifests for real+sim servers and benchmark job"
```

---

## Task 7: `run_eval.sh` wrapper

**Files:**
- Create: `evaluation/run_eval.sh`
- Test: validate with `bash -n` and, if available, `shellcheck` (no unit test — it drives a live cluster).

**Interfaces:**
- Consumes: `oc` / `kubectl`, `evaluation/run_eval.py`.
- Produces: a runnable script; the report is written by `run_eval.py`.

Script behavior: accept optional `NAMESPACE`, `REAL_SVC=vllm-real`, `SIM_SVC=vllm-sim`, `REAL_PORT=9001`, `SIM_PORT=9002`, `OUT=eval-out` via env; `oc port-forward svc/$REAL_SVC $REAL_PORT:8000 &` and same for sim; poll `GET /health` on both until ready or timeout; run `python -m evaluation.run_eval --real-url http://localhost:$REAL_PORT --sim-url http://localhost:$SIM_PORT --out "$OUT"`; always kill the port-forward PIDs on exit (`trap`).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Port-forward the in-cluster real+sim vLLM services and run the evaluation sweep.
set -euo pipefail

NAMESPACE="${NAMESPACE:-}"
REAL_SVC="${REAL_SVC:-vllm-real}"
SIM_SVC="${SIM_SVC:-vllm-sim}"
REAL_PORT="${REAL_PORT:-9001}"
SIM_PORT="${SIM_PORT:-9002}"
OUT="${OUT:-eval-out}"
KUBECTL="${KUBECTL:-oc}"

ns_flag=()
[ -n "$NAMESPACE" ] && ns_flag=(-n "$NAMESPACE")

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

"$KUBECTL" "${ns_flag[@]}" port-forward "svc/$REAL_SVC" "$REAL_PORT:8000" >/dev/null 2>&1 &
pids+=($!)
"$KUBECTL" "${ns_flag[@]}" port-forward "svc/$SIM_SVC" "$SIM_PORT:8000" >/dev/null 2>&1 &
pids+=($!)

wait_health() {
  local port="$1" name="$2" i
  for i in $(seq 1 60); do
    if curl -fsS "http://localhost:$port/health" >/dev/null 2>&1; then
      echo "$name ready on :$port"; return 0
    fi
    sleep 2
  done
  echo "ERROR: $name not ready on :$port" >&2; return 1
}

wait_health "$REAL_PORT" real
wait_health "$SIM_PORT" sim

python -m evaluation.run_eval \
  --real-url "http://localhost:$REAL_PORT" \
  --sim-url "http://localhost:$SIM_PORT" \
  --out "$OUT"
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n evaluation/run_eval.sh && chmod +x evaluation/run_eval.sh`
Expected: no output (valid), file executable. If `shellcheck` is installed, run `shellcheck evaluation/run_eval.sh` and address warnings.

- [ ] **Step 3: Commit**

```bash
git add evaluation/run_eval.sh
git commit -s -m "feat(eval): add port-forward wrapper for the evaluation sweep"
```

---

## Task 8: `evaluation/README.md` runbook + deps + top-level link

**Files:**
- Create: `evaluation/README.md`
- Modify: `pyproject.toml` (add `pyyaml` to `test`, add `eval` extra)
- Modify: `README.md` (add a short "Evaluation" pointer to `evaluation/README.md`)

**Interfaces:**
- Consumes: everything above.
- Produces: docs; the `eval` extra.

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml`:

```toml
[project.optional-dependencies]
test = ["pytest", "vllm", "pyyaml"]
eval = ["pyyaml"]
```

- [ ] **Step 2: Write `evaluation/README.md`** covering spec §8:

Sections (write actual prose + commands, no placeholders):
1. **What this measures** — sim-vs-real accuracy as MAPE; link `../docs/2026-08-19-sim-vs-real-evaluation-design.md`.
2. **Prerequisites** — `oc login` to the H100 cluster; `.venv` with the `eval` extra (`uv pip install -e ".[eval]"`); `vllm` CLI on the driver host; target namespace.
3. **Generate the sim config** —
   ```bash
   .venv/bin/python -m evaluation.gen_sim_config --model Qwen/Qwen3-32B --out deploy/eval/sim-configmap.yaml
   ```
4. **Deploy** — `oc apply -f deploy/eval/`; note the real server's ~64 GB weight download and long readiness delay; `oc get pods -w`.
5. **Run (default, local)** — `evaluation/run_eval.sh` (env knobs: `NAMESPACE`, ports, `OUT`); report lands in `eval-out/report.md`.
6. **Run (in-cluster)** — `oc apply -f deploy/eval/benchmark-job.yaml`; `oc logs job/vllm-eval`.
7. **Read the report** — explain APE, signed error, overall median MAPE; how to act on signed TTFT/ITL error to tune `beta` (link the main README physics section); note `beta=[1,1,0]` baseline is expected to show error.
8. **Customize** — edit `evaluation/sweep.yaml`; change model/GPU via `gen_sim_config.py` flags (`--tp`, `--peak-tflops`, `--hbm-gbps`); the fairness invariants (spec §5) to keep.
9. **Troubleshooting** — weight-download time; `/dev/shm` sizing; OpenShift non-root SCC cache env; `vllm bench serve` field/version mismatch (compare fails loudly naming the key).

- [ ] **Step 3: Add a pointer in top-level `README.md`** (after "Run" / before "Comparison"), e.g. a short "Evaluation" subsection linking `evaluation/README.md` and stating it compares the sim against real vLLM on an H100.

- [ ] **Step 4: Verify the full suite still passes**

Run: `.venv/bin/pytest tests/ -v`
Expected: all pass (existing + new eval tests).

- [ ] **Step 5: Commit**

```bash
git add evaluation/README.md pyproject.toml README.md
git commit -s -m "docs(eval): add evaluation runbook, eval extra, and README pointer"
```

---

## Task 9: Manual smoke test (documented, on the H100 cluster)

**Files:**
- Modify: `docs/2026-08-19-sim-vs-real-evaluation-design.md` (append a "Post-Smoke-Test Amendments" section, mirroring the k8s deployment design's pattern).

This task is not unit-testable — it validates the manifests and wrapper against the real OpenShift H100 cluster and records what had to change.

- [ ] **Step 1:** Regenerate `deploy/eval/sim-configmap.yaml` from the real config, `oc apply -f deploy/eval/`, wait for both pods ready.
- [ ] **Step 2:** Run `evaluation/run_eval.sh`; confirm `eval-out/report.md` is produced with non-empty metrics and an overall median MAPE.
- [ ] **Step 3:** Record any required changes (image tags, resource sizes, probe delays, `vllm bench serve` flag/field drift) in the design doc's amendments section.
- [ ] **Step 4: Commit**

```bash
git add docs/2026-08-19-sim-vs-real-evaluation-design.md deploy/eval
git commit -s -m "docs(eval): record post-smoke-test amendments"
```

---

## Self-Review

**Spec coverage:**
- §2 layout → Tasks 1–8 create every listed file (`compare.py`, `gen_sim_config.py`, `run_eval.py`, `sweep.yaml`, `run_eval.sh`, `README.md`, `deploy/eval/*`, tests, fixtures). ✅
- §3 data flow (generate config → deploy → sweep → compare → report) → Tasks 4, 6, 5, 2–3. ✅
- §4 components → Task 4 (gen), Task 6 (manifests), Task 5 (driver), Tasks 2–3 (compare), Task 7 (wrapper). ✅
- §5 fairness invariants → enforced in `bench_argv` (Task 5: `--ignore-eos`, seed, tokenizer, model) and tested; served-model-name parity tested in Task 6; warmup in `run()` (Task 5). ✅
- §6 sweep → `sweep.yaml` (Task 5) with the exact 4 points. ✅
- §7 report → `render_markdown`/`render_json` (Task 3), signed hints included. ✅
- §8 documentation → `evaluation/README.md` (Task 8). ✅
- §9 testing → unit tests Tasks 1–6; manual smoke Task 9; venv/pytest per CLAUDE.md. ✅
- §10 risks → weight download, `/dev/shm`, SCC, field drift all surface in manifests (Task 6) + README troubleshooting (Task 8) + defensive `KeyError` (Task 2). ✅

**Placeholder scan:** all code steps contain full implementations; `sim-configmap.yaml` is intentionally generated (documented command given), not a placeholder. No TBD/TODO. ✅

**Type consistency:** `MetricComparison`/`Metric` (Task 1) used unchanged in Tasks 2–3; `PointResult`/`aggregate`/`render_*` signatures defined in Task 3 and imported verbatim in Task 5; `SweepPoint`/`bench_argv` defined in Task 5 and matched by its test; `build_sim_config`/`render_configmap` names consistent across Task 4 and Task 6's generation command. ✅
