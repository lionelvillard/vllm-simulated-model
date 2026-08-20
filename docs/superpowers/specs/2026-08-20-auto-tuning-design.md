# Auto-Tuning of Physics Beta Parameters — Design

**Date:** 2026-08-20
**Status:** Approved
**Type:** New subsystem — auto-tuner for `vllm-simulated-model` physics coefficients

---

## 1. Purpose & Scope

Provide a tool that **automatically computes the `beta` coefficients** of the
physics latency model for a given model/hardware configuration. Today, users
hand-tune the three-element `beta = [beta_pf, beta_dc, beta_base]` vector
by reading the signed error from the evaluation report and adjusting manually.
This tool replaces that loop with an automated coordinate search.

**Parameters tuned:** `beta_pf`, `beta_dc`, `beta_base` (the three floats in
the `latency.beta` field of the sim config).

**Fixed for this design:** hardware constants (`peak_tflops`, `hbm_gbps`,
`weight_dtype`) and `tp` are treated as known inputs and are not searched.

**Starting workload:** `{isl: 1024, osl: 128, concurrency: 1, num_prompts: 32}`.

---

## 2. Architecture

```
tune.py
 │
 ├─ run real benchmark once ──────────────────────────────► real_results
 │
 └─ coordinate search (3 phases):
     for each candidate beta:
       POST /sim/config {"beta": [b_pf, b_dc, b_base]}   ← new endpoint
            │
            ▼
       sim-config file on disk  ←── model.forward() checks mtime each call
            │                        and reloads PhysicsLatencyModel if changed
            ▼
       vllm bench serve ──────────────────────────────────► sim_results
            │
            ▼
       compare_point(real_results, sim_results) → MAPE
 │
 └─ output: tuned sim-config.json + report
```

Zero pod restarts. The config update round-trip (POST → file write → mtime
detect → reload) completes before the next benchmark request arrives.

---

## 3. Components

### 3.1 `POST /sim/config` endpoint (`src/vllm_simulated/tuner_api.py`)

Registered as a `vllm.endpoint_plugins` entry point (the vLLM-supported
mechanism for adding FastAPI routes to the API server).

**Activation guard:** the plugin is a no-op unless `VLLM_SIM_TUNER=1` is set
in the environment. Production deployments that do not set this variable are
unaffected.

**Request body:** `{"beta": [beta_pf, beta_dc, beta_base]}` — a JSON object
with a single key `beta` containing a three-element array of non-negative
floats.

**Behaviour:**
1. Validate that `beta` has exactly three non-negative elements (reuse the
   validation already in `PhysicsConfig.from_dict`).
2. Read the current sim-config JSON from `$VLLM_SIM_CONFIG_PATH`.
3. Replace `latency.beta` with the new values.
4. Write atomically: write to a temp file in the same directory, then
   `os.replace` to the target path (prevents a partial read by the model).
5. Return `{"status": "ok", "beta": [...]}`.

**`required_tasks = None`** — eligible on any task. Does not use
`engine_client`; no RPC needed.

**`$VLLM_SIM_CONFIG_PATH`** must be set to the path of the sim-config file
accessible from both the API server process and the model worker process
(typically the same mounted volume path, e.g. `/model/config.json`).

### 3.2 Model reload (`src/vllm_simulated/model.py`)

`SimulatedForCausalLM.__init__` records two new fields:
- `_config_path: str | None` — value of `$VLLM_SIM_CONFIG_PATH` at init
- `_config_mtime: int` — initialized to `0`

**Bootstrapping:** if `$VLLM_SIM_CONFIG_PATH` is set and the file does not
yet exist, `__init__` serializes the current config (reconstructed from
`hf_config`, which already contains the full `latency` block) to that path.
This lets the operator point `VLLM_SIM_CONFIG_PATH` at any writable location
(e.g. `/tmp/vllm_sim_config.json`) without a separate init-container step;
the model self-populates the file on first startup.

On every `forward()` call, before sleeping:

```python
if self._config_path:
    mtime = os.stat(self._config_path).st_mtime_ns
    if mtime != self._config_mtime:
        self._reload_latency_config()
        self._config_mtime = mtime
```

`_reload_latency_config` reads the JSON and calls
`build_latency_model(new_cfg["latency"], hf_config=self._hf_config)`,
replacing `self.latency` in-place. The `os.stat` syscall (~0.3 µs) is
negligible against the step-time sleep that follows.

**Tuner barrier:** `tune.py` sleeps 0.5 s after `POST /sim/config` before
starting the benchmark. Any request mid-flight when the config changes will
finish with the old config; the sleep ensures all such requests have exited
`forward()` before new measurements begin.

### 3.3 Tuner driver (`evaluation/tune.py`)

**CLI:**

```
python -m evaluation.tune \
  --real-url   http://...        \
  --sim-url    http://...        \
  --tokenizer  Qwen/Qwen3-32B   \
  --model-config models/qwen3-32b/deployments/h100-sxm5-tp1/latency/physics/sim-config.json \
  --out        tune-out/
```

`--model-config` is the sim-config.json whose `latency.beta` will be tuned.
The architecture block is read from it so the tuner can compute expected
physics step times for logging; the beta is overwritten in the output.

**Algorithm — coordinate search:**

```
1. run_real_bench(point) → real_result       (saved to out/real.json; runs once)

2. Phase 1 — beta_pf:
   minimize_scalar(
     f(b) = bench_sim([b, 1.0, 0.0]) → TTFT mean MAPE,
     bounds=(0.05, 5.0), method="bounded"    # scipy Brent on bounded interval
   ) → best_pf   (~10–12 evaluations)

3. Phase 2 — beta_dc:
   minimize_scalar(
     f(b) = bench_sim([best_pf, b, 0.0]) → ITL mean MAPE,
     bounds=(0.05, 5.0), method="bounded"
   ) → best_dc   (~10–12 evaluations)

4. Phase 3 — beta_base:
   minimize_scalar(
     f(b) = bench_sim([best_pf, best_dc, b]) → overall median MAPE,
     bounds=(0.0, 200.0), method="bounded"
   ) → best_base   (~10–12 evaluations)

5. Write out/tuned-sim-config.json  (original config with beta replaced)
6. Write out/tuning-report.md + out/tuning-report.json
```

`bench_sim(beta)`:
1. `POST sim_url/sim/config {"beta": beta}`
2. `time.sleep(0.5)` — barrier
3. `vllm bench serve` (same `bench_argv` helper as `run_eval.py`, no warmup)
4. `compare_point(real_result, sim_result)` — returns per-metric MAPE dict

**Per-phase objective:** TTFT mean APE (phase 1), ITL mean APE (phase 2),
overall median MAPE across all metrics (phase 3). All are already computed
by the existing `compare_point`.

**Intermediate results:** each `bench_sim` call saves its result JSON to
`out/sim-{phase}-{n}.json` so a failed run can be inspected.

**Estimated runtime** for the starting workload (concurrency=1, 32 prompts,
no warmup per sim run): ~80 s per sim benchmark + 0.5 s barrier. ~35 total
sim evaluations → ~50 minutes. The single real run adds ~80 s.

### 3.4 Output

**`out/tuned-sim-config.json`** — a drop-in replacement for the existing
sim-config: identical architecture block, `latency.beta` replaced with the
tuned triple. Paste its `beta` into the ConfigMap to update the deployed sim.

**`out/tuning-report.md`** and **`out/tuning-report.json`** — per-phase
convergence table (evaluated beta, per-metric MAPE at each step), final
tuned beta triple, and a before/after MAPE comparison at the tuning workload.

---

## 4. Files Touched

| File | Change |
|------|--------|
| `src/vllm_simulated/tuner_api.py` | **new** — `SimTunerEndpointPlugin` |
| `src/vllm_simulated/model.py` | **modified** — mtime check + `_reload_latency_config` |
| `pyproject.toml` | **modified** — add `vllm.endpoint_plugins` entry point + `scipy` to `eval` extras |
| `evaluation/tune.py` | **new** — tuner driver |
| `tests/test_tune.py` | **new** — unit tests for tuner logic |

---

## 5. Testing

**Unit (CI, no GPU/cluster):**
- `tests/test_tune.py` — tests coordinate search logic with a mock `bench_sim`
  that returns synthetic MAPE values; verifies each phase picks the minimum and
  passes the best forward to the next phase.
- `tests/test_tuner_api.py` — tests the endpoint: validates that a valid POST
  updates the config file atomically, rejects malformed beta (wrong length,
  negative values), and is a no-op when `VLLM_SIM_TUNER` is not set.
- The mtime-reload path in `model.py` is tested by writing a new config file
  in the test and confirming `self.latency` is replaced on the next
  `_maybe_reload_config` call.

**Manual smoke test:** deploy the sim with `VLLM_SIM_TUNER=1` and
`VLLM_SIM_CONFIG_PATH=/model/config.json`, run
`python -m evaluation.tune --real-url ... --sim-url ...`, confirm
`tuned-sim-config.json` is written and its MAPE is lower than the default
`beta=[1,1,0]` baseline.

---

## 6. Correctness Decisions

- **Warmup disabled for sim runs** during tuning: warmup adds ~80 s per
  evaluation and is not needed since the sim has no warm-up effect (no JIT
  after the first request). The real baseline run uses warmup (matches
  existing `run_eval.py` default).
- **Fixed seed** (`--seed 0`) on every bench invocation — same as `run_eval.py`.
- **`--ignore-eos`** passed to all invocations — same fairness requirement as
  the existing evaluation.
- **Phase ordering** (pf → dc → base) is correct because at concurrency=1,
  TTFT is dominated by prefill and ITL by decode; the phases are nearly
  orthogonal, so sequential optimization converges close to the joint minimum.

---

## 7. Risks & Notes

- **`scipy` dependency:** `minimize_scalar` is in `scipy.optimize`. Added to
  the `eval` extras in `pyproject.toml`. Not needed for the core package.
- **`VLLM_SIM_CONFIG_PATH` must be writable** from the API server process.
  The recommended path is `/tmp/vllm_sim_config.json` (always writable inside
  the container). The model self-bootstraps this file on startup (§3.2), so
  no init-container change is required. The ConfigMap volume at `/model`
  remains the read-only source of truth for non-tuning deployments.
- **Single-phase optimum is not global optimum.** Coordinate descent can
  miss the joint minimum when parameters are correlated. For concurrency=1
  the phases are nearly independent (prefill and decode don't overlap), so
  this is acceptable for the starting workload. A joint search (e.g. Nelder-Mead)
  can be added later if needed for higher-concurrency workloads.
- **`minimize_scalar` may call `bench_sim` with the same beta twice** (Brent
  caches nothing). Caching evaluated betas by value (rounded to 4 decimal
  places) would eliminate redundant runs; deferred to a follow-up.
