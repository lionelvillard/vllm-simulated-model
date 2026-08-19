# Sim-vs-Real Accuracy Evaluation — Design

**Date:** 2026-08-19
**Status:** Approved (pending spec review)
**Type:** Evaluation tooling (automation script + manifests, in this package repo)

## 1. Purpose & Scope

Provide an **automated tool** that measures how faithfully `vllm-simulated-model`
reproduces the latency/throughput behavior of a **real** model on a **real GPU**.
The tool runs an identical benchmark workload against two vLLM servers — the real
model on an H100, and the simulated model on CPU — and reports the per-metric
error (MAPE) between them.

**Primary use case:** answer "how accurate is the simulator?" for a given model /
hardware, and expose the direction of any systematic error so a user can hand-tune
the physics `beta` coefficients per the README.

**First target (fixed for the MVP):**
- Model: `Qwen/Qwen3-32B` (dense, GQA, ungated — no HF token required)
- Hardware: **1× H100 SXM5**, tensor-parallel degree **1**
- Both servers run **in-cluster on OpenShift**

### Why this is simpler than BLIS's evaluation

BLIS *reimplements* the vLLM scheduler, so its evaluation must **replay a captured
arrival trace** (`observe` → `replay` → `calibrate`) to isolate model error from
scheduler differences. `vllm-simulated-model` runs the **real vLLM scheduler and
batching** — only the forward pass is replaced with a sleep. Therefore:

- No trace capture/replay is needed. The real scheduler *is* the scheduler on both
  sides; the only thing under test is whether the physics `step_time_ms` predicts
  real H100 step time well enough that the emergent end-to-end metrics match.
- We run the **same benchmark** (`vllm bench serve`) against both endpoints and
  compare aggregate metrics directly.

We adopt BLIS's **reporting** shape (MAPE per metric, broken into
mean/p90/p99 for TTFT/ITL/E2E plus throughput; median MAPE headline) but not its
capture/replay mechanism.

### Explicitly out of scope (MVP)

- Auto-calibration / auto-fitting of `beta` from a real run (report only surfaces
  the error and its sign; tuning stays manual — possible future work).
- Multi-model / multi-GPU / multi-TP evaluation matrix (one config for now).
- Trace-driven replay (not needed — the real scheduler runs).
- Statistical rigor beyond repeated runs (no confidence intervals in v1).

## 2. Deliverables & Layout

```
deploy/eval/
  real-deployment.yaml      # Qwen/Qwen3-32B on H100 (GPU), TP=1, real weights
  real-service.yaml         # ClusterIP :8000 for the real server
  sim-deployment.yaml       # SimulatedForCausalLM matching Qwen3-32B, CPU
  sim-service.yaml          # ClusterIP :8000 for the sim server
  sim-configmap.yaml        # generated config.json (arch of Qwen3-32B + physics latency)
  benchmark-job.yaml        # optional: run the driver fully in-cluster
evaluation/
  README.md                 # how to deploy, run, and read the report (see §8)
  run_eval.py               # driver: sweep -> vllm bench serve -> parse -> report
  compare.py                # pure MAPE + report-rendering logic (unit-tested)
  gen_sim_config.py         # fetch real config.json, inject physics latency block
  sweep.yaml                # the benchmark matrix (ISL/OSL/concurrency points)
  run_eval.sh               # wrapper: oc port-forward both Services, run run_eval.py
tests/
  test_compare.py           # MAPE + report against committed fixture result JSONs
  fixtures/eval/            # sample vllm-bench result JSONs (real + sim)
docs/
  2026-08-19-sim-vs-real-evaluation-design.md   # this file
```

The core comparison logic (`compare.py`) is **URL- and cluster-agnostic pure
Python** so it is fully unit-testable in CI without a GPU or a cluster. Everything
that touches the cluster lives in `run_eval.py` / `run_eval.sh` / manifests and is
validated by a documented manual smoke test.

## 3. Architecture & Data Flow

```
                        gen_sim_config.py
   Qwen/Qwen3-32B  ─────────────────────────►  sim config.json
   config.json           (inject physics             │
   (from HF)              latency block)              ▼
                                              deploy/eval/sim-configmap.yaml

  ┌─────────────────────────────────────────────────────────────────┐
  │ OpenShift cluster                                                 │
  │                                                                   │
  │   real-deployment (H100, GPU)  ──►  Service vllm-real :8000       │
  │   sim-deployment  (CPU)        ──►  Service vllm-sim  :8000       │
  └───────────────▲───────────────────────────▲─────────────────────┘
                  │ oc port-forward            │ oc port-forward
                  │                            │
              run_eval.sh ─────────────► run_eval.py
                                              │
                for each sweep point:         │
                  vllm bench serve ──► real result.json
                  vllm bench serve ──► sim  result.json
                                              │
                                              ▼
                                         compare.py
                                              │
                                    report.md + report.json
```

**Approach C (hybrid), chosen.** `run_eval.py` takes two base URLs and is
agnostic to how they are reached. The default path is `run_eval.sh`, which
port-forwards both in-cluster Services to localhost and runs the driver locally so
the report lands on the user's machine. `benchmark-job.yaml` is also shipped for
users who want the driver to run fully in-cluster (results read from pod logs / a
mounted volume).

## 4. Components

### 4.1 `gen_sim_config.py`

- **What it does:** produces the simulated model's `config.json`.
- **How:** fetch the real model's `config.json` (via `huggingface_hub` /
  `transformers` `AutoConfig`, or a local path), keep the architecture fields
  verbatim, set `architectures: ["SimulatedForCausalLM"]`, and inject a `latency`
  block of `type: "physics"` with the H100 SXM5 hardware spec
  (`peak_tflops: 989`, `hbm_gbps: 3350`, `weight_dtype: "bfloat16"`),
  `tp: 1`, `beta: [1.0, 1.0, 0.0]`, `deterministic_length: true`.
- **Why generate rather than hand-copy:** guarantees the physics FLOPs math is
  computed against the *exact* architecture of the real model — no manual drift in
  layer count / hidden size / head config.
- **Output:** writes `deploy/eval/sim-configmap.yaml` (a ConfigMap embedding the
  generated `config.json`), overwriting any stale copy.
- **Depends on:** the real model id or a path to its `config.json`; the physics
  hardware constants (defaulted to H100 SXM5, overridable via flags).

### 4.2 Manifests (`deploy/eval/`)

- **`real-deployment.yaml` + `real-service.yaml`:** a GPU vLLM Deployment serving
  `Qwen/Qwen3-32B` with `--served-model-name qwen3-32b`, `--tokenizer
  Qwen/Qwen3-32B`, TP=1, `--gpu-memory-utilization 0.9`, and a `--max-model-len`
  capped so weights (~64 GB bf16) plus KV cache fit in 80 GB. Requests
  `nvidia.com/gpu: 1`; carries the H100 node selector / tolerations for the target
  OpenShift cluster. Weight handling: ephemeral download to an `emptyDir` sized for
  ~64 GB for the MVP (a PVC cache is a documented optimization, not required).
- **`sim-deployment.yaml` + `sim-service.yaml` + `sim-configmap.yaml`:** the CPU
  sim, reusing the proven `deploy/` pattern (init-container plugin injection,
  non-root SCC cache redirection, `/dev/shm` sizing) but with
  `--served-model-name qwen3-32b`, `--tokenizer Qwen/Qwen3-32B`, and the
  **generated** config.json from §4.1.
- **`benchmark-job.yaml`:** a Job that runs `run_eval.py` against the two
  cluster-internal Service DNS names; writes report artifacts to a mounted volume
  or stdout.

### 4.3 `run_eval.py` (driver)

- **Inputs:** `--real-url`, `--sim-url`, `--model qwen3-32b`, `--tokenizer
  Qwen/Qwen3-32B`, `--sweep sweep.yaml`, `--out <dir>`, `--warmup N`.
- **Per sweep point** (ISL, OSL, concurrency/rate): invoke `vllm bench serve` as a
  subprocess with a fixed `--seed`, `--ignore-eos`, `--num-prompts`,
  `--max-concurrency`, `--random-input-len`, `--random-output-len`,
  `--save-result --result-filename <point>-{real,sim}.json`, once per endpoint
  (real, then sim). Discard `--warmup` requests (or a warmup pre-run).
- **Then:** hand the two result JSONs per point to `compare.py`.
- **Depends on:** `vllm` CLI available on the driver host; reachable URLs.

### 4.4 `compare.py` (pure logic — the tested core)

- **What it does:** parse two `vllm bench serve` result JSONs, extract the metric
  set, compute Absolute Percentage Error per metric
  (`APE = |sim - real| / real`), aggregate to a median MAPE, and render both a
  markdown report and a machine-readable JSON.
- **Metrics compared** (as emitted by `vllm bench serve`): TTFT mean/median/p90/p99,
  ITL (a.k.a. TPOT) mean/median/p99, E2E mean/p99, output-token throughput,
  request throughput.
- **Report additions:** a headline median MAPE; and for TTFT and ITL the **signed**
  error (sim − real) so the user can see whether to raise `beta_pf` / `beta_dc`
  (per the README tuning guidance) — auto-tuning remains out of scope.
- **No I/O beyond reading given JSON and returning strings/dicts** — so it is fully
  unit-tested.

### 4.5 `run_eval.sh` (wrapper)

- Port-forwards `svc/vllm-real` and `svc/vllm-sim` to two local ports, waits for
  `/health` on both, runs `run_eval.py` with the forwarded URLs, tears the
  forwards down, and prints the report path. This is the default, lowest-friction
  path for a laptop-driven run against the in-cluster servers.

## 5. Correctness Decisions (fairness of the comparison)

These are the decisions that make the two runs comparable; each is a hard
requirement, not a preference.

1. **Identical architecture.** The sim config is generated from the real config
   (§4.1), so the physics model's FLOPs/bytes are computed for the real model's
   exact shape.
2. **Identical tokenizer.** Both servers run `--tokenizer Qwen/Qwen3-32B`, so ISL
   and OSL token accounting is the same on both sides.
3. **Fixed output length on both sides.** The sim masks EOS
   (`deterministic_length: true`) and always runs to `max_tokens`; a real model
   would stop early on EOS. The benchmark therefore passes `--ignore-eos` to
   **both** endpoints so every request emits exactly OSL tokens — otherwise ITL and
   E2E are not comparable.
4. **Identical served-model-name.** Both expose `qwen3-32b`, so one unchanged
   `vllm bench serve --model qwen3-32b` invocation targets either endpoint.
5. **Warmup discarded.** The sim pays a one-time plugin-import / JIT cost on its
   first request; warmup requests are excluded from the measured window.
6. **Fixed seed & request stream shape.** The same `--seed`, `--num-prompts`,
   input/output lengths, and concurrency are used for the real and sim run of each
   sweep point.

## 6. Workload Sweep (MVP)

Kept deliberately small — each point consumes real H100 time. Defined in
`sweep.yaml`:

| ISL | OSL | max-concurrency | num-prompts |
|----:|----:|----------------:|------------:|
| 1024 | 128 | 1  | 32  |
| 1024 | 128 | 16 | 256 |
| 1024 | 128 | 64 | 512 |
| 256  | 256 | 32 | 256 |

The three concurrency levels at fixed ISL/OSL exercise batch-composition effects
(the sim's whole reason to exist); the fourth point varies the ISL/OSL balance.
`num-prompts` scales with concurrency so each run reaches steady state. The matrix
is data, not code — users edit `sweep.yaml` to extend it.

## 7. Report Format

`report.md` (human) + `report.json` (machine). Per sweep point:

```
### ISL=1024 OSL=128 concurrency=16

| Metric              |    Real |     Sim |    APE |
|---------------------|--------:|--------:|-------:|
| TTFT mean (ms)      |   142.0 |   150.3 |   5.8% |
| TTFT p99 (ms)       |   210.5 |   233.1 |  10.7% |
| ITL mean (ms)       |    18.4 |    17.9 |   2.7% |
| E2E mean (ms)       |  2490.0 |  2431.0 |   2.4% |
| Output tok/s        |  1830.0 |  1875.0 |   2.5% |

Signed TTFT error: +5.8% (sim slower) -> consider lowering beta_pf
Signed ITL  error: -2.7% (sim faster) -> consider raising beta_dc
```

Followed by an **aggregate** section: median MAPE across all points per metric, and
one overall headline median MAPE (the BLIS-style single number).

## 8. Documentation (`evaluation/README.md`)

The `evaluation/` directory ships its own `README.md` — the operator-facing
runbook, self-contained so a user never has to read this design doc to run an
evaluation. It covers:

- **Prerequisites:** access to the H100 OpenShift cluster (`oc login`), `vllm`
  CLI available on the driver host, the target namespace.
- **Deploy:** `oc apply -f deploy/eval/` (real GPU server + sim CPU server +
  services), including how to (re)generate the sim ConfigMap with
  `gen_sim_config.py` for a different model or hardware spec.
- **Run:** the default `run_eval.sh` path (port-forward both Services, run the
  sweep locally) and the in-cluster `benchmark-job.yaml` alternative.
- **Read the report:** what each metric and the MAPE headline mean, and how to act
  on the signed TTFT/ITL error to hand-tune `beta` (cross-links the main README's
  physics-model tuning section).
- **Customize:** editing `sweep.yaml`, switching model / GPU spec / TP, and the
  fairness requirements from §5 that must be preserved when doing so.
- **Troubleshooting:** weight-download time, `/dev/shm` sizing, OpenShift non-root
  SCC notes (reused from the k8s deployment design), and `vllm bench serve`
  version/field mismatches.

This is documented as a first-class deliverable, not an afterthought: the plan
includes writing it, and the manual smoke test verifies the runbook's steps
actually work end to end.

## 9. Testing

- **Unit (CI, no GPU/cluster):** `tests/test_compare.py` runs `compare.py` against
  committed fixture result JSONs (`tests/fixtures/eval/`) — verifies APE math,
  aggregation, sign of error, and report rendering. `gen_sim_config.py` is tested
  against a fixture real `config.json` to verify the physics block is injected and
  architecture fields are preserved.
- **Manual smoke test (documented):** deploy the manifests to the H100 OpenShift
  cluster, run `run_eval.sh`, confirm both servers serve and a report is produced.
  Findings/amendments recorded in this doc's "Post-Smoke-Test" section (mirroring
  the k8s deployment design's pattern).
- Per `CLAUDE.md`, all tests run under the worktree `.venv` via `pytest`.

## 10. Risks & Notes

- **Weight download (~64 GB).** First real-server startup downloads Qwen3-32B;
  readiness-probe delay and `emptyDir` sizing must account for it. A PVC cache is a
  documented follow-up, not required for MVP.
- **CPU sleep granularity (~1 ms).** Fine for a 32B/H100 workload where ITL is tens
  of ms; noted as a floor for very small models.
- **`beta` starts at [1,1,0].** The first report is expected to show non-zero error;
  its purpose is to *quantify and direct* tuning, not to pass a threshold on run
  one.
- **`vllm bench serve` field names** may vary across vLLM versions; `compare.py`
  reads them defensively and fails loudly with the offending JSON if a field is
  missing.

## 11. Post-Smoke-Test Amendments

> **Status: NOT YET RUN.** The manual smoke test (spec §9) requires the H100
> OpenShift cluster and has **not** been executed in the development
> environment where this feature was built. This section is a template: the
> operator who runs the first real deployment fills in the checklist below and
> records any amendments. Do not treat the placeholders as observed results.

### Smoke-test procedure

Run these from the repo (or worktree) root, with `oc login` already pointing at
the target H100 namespace:

1. **Generate the sim config from the real model** (overwrites the committed
   fixture-derived placeholder in `deploy/eval/sim-configmap.yaml`):

   ```bash
   .venv/bin/python -m evaluation.gen_sim_config \
     --model Qwen/Qwen3-32B \
     --out deploy/eval/sim-configmap.yaml
   ```

2. **Deploy both servers and the config:**

   ```bash
   oc apply -f deploy/eval/
   oc get pods -w        # wait for vllm-real and vllm-sim pods to be Ready
   ```

   The real server downloads ~64 GB of weights on first start; readiness can
   take many minutes (probe `initialDelaySeconds` is 600).

3. **Run the sweep** (local port-forward driver):

   ```bash
   evaluation/run_eval.sh          # writes eval-out/report.md and report.json
   ```

   Or run it in-cluster instead:

   ```bash
   oc apply -f deploy/eval/benchmark-job.yaml
   oc logs -f job/vllm-benchmark
   ```

4. **Confirm the report** — `eval-out/report.md` exists, every sweep point has
   non-empty metrics, and an overall median MAPE is printed.

### Amendment checklist (operator fills in)

Record the actual value or "no change" for each; anything that differs from the
committed manifests/scripts becomes a follow-up commit.

- [ ] **Real image tag** — did `vllm/vllm-openai:latest` serve Qwen3-32B, or was
      a pinned tag required? Value: _____
- [ ] **GPU/type** — confirmed 1× H100 SXM5 (peak_tflops 989, hbm_gbps 3350)?
      If the cluster GPU differs, update `H100_SXM5` in `gen_sim_config.py` and
      regenerate. Value: _____
- [ ] **Resource sizes** — was the 80Gi weight `emptyDir` and any memory/CPU
      request sufficient? Value: _____
- [ ] **Probe delays** — did `initialDelaySeconds: 600` cover download + load,
      or did the real pod get killed before Ready? Value: _____
- [ ] **`/dev/shm`** — was 256Mi enough, or did loading need more? Value: _____
- [ ] **`vllm bench serve` field/flag drift** — did any `--percentile-metrics`,
      `--metric-percentiles`, dataset flag, or result-JSON field name differ from
      what `run_eval.py`/`compare.py` expect? If a field was missing, `compare.py`
      fails loudly naming the key — record it here. Value: _____
- [ ] **Served-model-name parity** — did both servers answer to `qwen3-32b`?
      Value: _____
- [ ] **Overall median MAPE (baseline `beta=[1,1,0]`)** — record it as the
      pre-tuning baseline; non-zero is expected (spec §10). Value: _____
- [ ] **Per-metric signed error** — note TTFT and ITL sign/magnitude to direct
      the first `beta` tuning pass (positive ms = sim slower). Value: _____
