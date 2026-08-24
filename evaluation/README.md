# Sim-vs-Real Evaluation

This tool measures how faithfully `vllm-simulated-model` reproduces the latency and throughput behavior of a real model on a real GPU. It runs an identical benchmark workload against two vLLM servers — the real model on an H100, and the simulated model on CPU — and reports the per-metric error (MAPE) between them.

For design details and rationale, see [../docs/2026-08-19-sim-vs-real-evaluation-design.md](../docs/2026-08-19-sim-vs-real-evaluation-design.md).

## What this measures

The evaluation produces:
- **Absolute Percentage Error (APE)** per metric: `|sim - real| / real`
- **Signed error** for TTFT and ITL: `(sim - real) / real * 100` (shows direction — positive means sim slower)
- **Median MAPE** across all metrics and sweep points (single headline number)

Metrics compared (from `vllm bench serve`):
- TTFT (time to first token): mean, median, p90, p99
- ITL (inter-token latency / TPOT): mean, median, p99
- E2E (end-to-end latency): mean, p99
- Output token throughput and request throughput

The baseline physics model configuration (`beta = [1.0, 1.0, 0.0]`) is expected to show non-zero error. The report quantifies the error and indicates which direction to tune — it is not a pass/fail threshold.

## Prerequisites

1. **Access to the H100 OpenShift cluster:**
   ```bash
   oc login <cluster-url>
   ```

2. **Local Python environment with the `eval` extra:**
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[eval]"
   ```

3. **vLLM CLI** available on the driver host (required for `vllm bench serve`):
   ```bash
   # Ensure vLLM is installed from source (see main README)
   vllm --version
   ```

4. **Target namespace** where the real and sim servers will be deployed.

## Generate the sim config

The simulated model's `config.json` must match the real model's architecture. Generate it with:

```bash
.venv/bin/python -m evaluation.gen_sim_config \
  --model Qwen/Qwen3-32B \
  --out models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/configmap.yaml
```

**Flags:**
- `--model <HF id or path to config.json>` (required): the real model to match
- `--out <path>` (required): output ConfigMap YAML path
- `--tp <int>` (default: 1): tensor-parallel degree
- `--peak-tflops <float>` (default: 989.0): H100 SXM5 peak TFLOPs
- `--hbm-gbps <float>` (default: 3350.0): H100 SXM5 HBM bandwidth (GB/s)
- `--weight-dtype <str>` (default: bfloat16): weight data type

To change the target model or GPU spec, adjust these flags and regenerate. The ConfigMap name is derived from the latency config hash (e.g. `vllm-qwen3-32b-standalone-eae748-config` for the `physics` variant).

## Deploy

Apply the manifests to deploy both the real GPU server and the simulated CPU server:

```bash
# 1. Shared PVC (once per cluster/namespace):
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/k8s/pvc.yaml

# 2. ConfigMap for the chosen latency variant (e.g. physics):
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/configmap.yaml

# 3. Eval stack (sim + real Deployments and Services):
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/sim-deployment.yaml
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/sim-service.yaml
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/real-deployment.yaml
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/real-service.yaml
```

This creates:
- `vllm-qwen3-32b-standalone-eae748-real` Deployment and Service (H100 GPU, real Qwen/Qwen3-32B weights)
- `vllm-qwen3-32b-standalone-eae748-sim` Deployment and Service (CPU, simulated model)
- `vllm-qwen3-32b-standalone-eae748-config` ConfigMap (model architecture and latency parameters)

**Note:** The real server downloads ~64 GB of weights on first startup. Readiness may take 5–10 minutes depending on network bandwidth. Watch pod status:

```bash
oc get pods -n <namespace> -w
```

## Run the evaluation (default, local)

The default approach uses `run_eval.sh`, which takes a **variant directory** (a
tuned-deployment dir such as `latency/physics`), port-forwards both Services to
localhost, and runs the sweep from your machine:

```bash
evaluation/run_eval.sh models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics
```

From the variant directory the script derives everything it needs:
- **real/sim Service names** from `eval/real-service.yaml` and `eval/sim-service.yaml`
- **`--model-config`** from `configmap.yaml` (embeds the latency type, β values, and hardware spec in the report)
- **sweep matrix** from the evaluation dir's `sweep.yaml` (`<eval>/sweep.yaml`), if present
- **output directory** `<eval>/results/<category>/<variant>` — e.g. `evaluation/results/latency/physics`

**Environment overrides** (rarely needed; each defaults to a value derived from the variant dir):
- `NAMESPACE` (empty): OpenShift namespace
- `REAL_SVC` / `SIM_SVC`: real / sim Service names
- `REAL_PORT=9001` / `SIM_PORT=9002`: local ports
- `OUT`: output directory for reports and result JSONs
- `MODEL_CONFIG`: path to the sim ConfigMap YAML or `sim-config.json`
- `SWEEP`: path to the sweep matrix YAML
- `KUBECTL=oc`: the Kubernetes CLI to use (e.g., `kubectl` or `oc`)

Example targeting a specific namespace:

```bash
NAMESPACE=my-namespace \
evaluation/run_eval.sh models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics
```

The script waits for both Services' `/health` endpoints to respond, then runs `run_eval.py` to execute the benchmark sweep. Results land in `$OUT/report.md` and `$OUT/report.json`.

## Run the evaluation (in-cluster)

Alternatively, run the driver fully in-cluster via a Job. **Note:** This in-cluster Job path has NOT been validated against a live cluster (the validated path is `run_eval.sh`). It uses the vLLM CPU image (for the `vllm bench serve` CLI) and puts the evaluation source tree on PYTHONPATH.

```bash
oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/benchmark-job.yaml
```

The Job is named **`vllm-qwen3-32b-standalone-eae748-benchmark`**. View logs with:

```bash
oc logs -n <namespace> job/vllm-qwen3-32b-standalone-eae748-benchmark -f
```

Report artifacts are written to `/out` in the pod (backed by an emptyDir). To retrieve them:

```bash
pod=$(oc get pod -n <namespace> -l app=vllm-qwen3-32b-standalone-eae748-benchmark -o jsonpath='{.items[0].metadata.name}')
oc cp -n <namespace> "$pod:/out/report.md" ./report.md
oc cp -n <namespace> "$pod:/out/report.json" ./report.json
```

## Read the report

The `report.md` contains one section per sweep point, showing real vs. sim metrics and APE, followed by an aggregate summary.

**Per-point example:**

```
### ISL=1024 OSL=128 c=16

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

**Acting on signed error:**
- **Positive TTFT error** (sim slower): sim overpredicts prefill time. Consider *lowering* `beta_pf` in the physics model.
- **Negative TTFT error** (sim faster): sim underpredicts prefill time. *Raise* `beta_pf`.
- **Positive ITL error** (sim slower): sim overpredicts decode time. Consider *lowering* `beta_dc`.
- **Negative ITL error** (sim faster): sim underpredicts decode time. *Raise* `beta_dc`.

The `beta` coefficients live in the `latency` block of the simulated model's `config.json`. For full guidance on tuning the physics model, see the [Physics model](../README.md#physics-model-type-physics) section of the main README.

**Aggregate summary:**

The end of the report shows the median APE per metric across all sweep points and an overall median MAPE headline.

## Auto-tune the physics parameters

When the evaluation report shows significant error, use `tune.py` to automatically find better `beta` values. It runs a coordinate search — one phase per parameter — against a fixed tuning workload (ISL=1024, OSL=128, concurrency=1, 32 prompts) and writes a ready-to-use sim config.

```bash
python -m evaluation.tune \
  --real-url http://localhost:9001 \
  --sim-url  http://localhost:9002 \
  --model-config models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/configmap.yaml \
  --tokenizer Qwen/Qwen3-32B \
  --out tune-out
```

**Flags:**
- `--real-url` (required): base URL of the real vLLM server
- `--sim-url` (required): base URL of the simulated vLLM server
- `--model-config` (required): path to the sim `config.json` (or ConfigMap YAML) whose `beta` will be tuned
- `--tokenizer` (default: `Qwen/Qwen3-32B`): tokenizer used for benchmarking
- `--out` (default: `tune-out`): directory where outputs are written
- `--seed` (default: `0`): random seed for reproducibility

**How it works:**

The search runs three sequential phases, each using `scipy.optimize.minimize_scalar` (bounded Brent's method):

1. **Phase 1 — `beta_pf`** (`0.05–5.0`): minimises TTFT mean APE while holding `beta_dc=1.0` and `beta_base=0.0`.
2. **Phase 2 — `beta_dc`** (`0.05–5.0`): minimises ITL mean APE using the `beta_pf` found in phase 1.
3. **Phase 3 — `beta_base`** (`0.0–200.0`): minimises overall median MAPE using the `beta_pf` and `beta_dc` from phases 1–2.

Each optimizer call runs `vllm bench serve` against the sim server (after POSTing the candidate `beta` to `/sim/config`) and compares the result against a single real-benchmark run taken at the start.

**Outputs** (written to `--out`):

| File | Contents |
|------|----------|
| `tuned-sim-config.json` | Copy of the input config with `latency.beta` set to the tuned values |
| `tuning-report.md` | Per-phase iteration table (beta → MAPE) and final comparison at tuned beta |
| `tuning-report.json` | Same in JSON form |
| `real.json` | Real-server benchmark result used as the reference |
| `sim-tune-<n>.json` | Sim benchmark result for each optimizer call |
| `sim-final.json` | Sim benchmark result at the final tuned beta |

After tuning, copy `tuned-sim-config.json` into your deployment (e.g. update the ConfigMap) and re-run the full evaluation sweep to confirm the improvement generalises across all sweep points.

## Customize the evaluation

### Extend the sweep

Edit `evaluation/sweep.yaml` to add more benchmark points:

```yaml
points:
  - {isl: 1024, osl: 128, concurrency: 1,  num_prompts: 32}
  - {isl: 1024, osl: 128, concurrency: 16, num_prompts: 256}
  - {isl: 1024, osl: 128, concurrency: 64, num_prompts: 512}
  - {isl: 256,  osl: 256, concurrency: 32, num_prompts: 256}
  # Add your own points here
```

Each point runs on the real H100, so keep the matrix small.

### Change the model or GPU

To evaluate a different model or hardware:
1. Regenerate the sim config with `gen_sim_config.py` using the appropriate `--model`, `--tp`, `--peak-tflops`, `--hbm-gbps`, and `--weight-dtype` flags.
2. Update `models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/real-deployment.yaml` to request the correct model and GPU type.
3. Redeploy: `oc apply -n <namespace> -f models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/`

### Fairness invariants

When customizing, preserve these requirements to keep the comparison valid:
1. **Identical architecture** — generate the sim config from the real model's `config.json`.
2. **Identical tokenizer** — both servers must use the same `--tokenizer`.
3. **Fixed output length** — the sim masks EOS (`deterministic_length: true`), and the benchmark must pass `--ignore-eos` to both endpoints (already done by `run_eval.py`).
4. **Identical served-model-name** — both expose the same model name so one benchmark invocation targets either endpoint.
5. **Warmup discarded** — warmup requests are excluded from the measured window (use `--no-warmup` to disable if needed).
6. **Fixed seed & stream** — the same `--seed`, `--num-prompts`, and input/output lengths are used for both runs (enforced by `run_eval.py`).

## Troubleshooting

### Weight download time

The real server downloads Qwen3-32B (~64 GB) on first startup. If readiness takes longer than expected:
- Check the pod's events: `oc describe pod -n <namespace> <pod-name>`
- Increase the readiness probe `initialDelaySeconds` in `models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/physics/eval/real-deployment.yaml` if the default is insufficient.
- For repeated runs, consider using a PersistentVolumeClaim to cache the weights (not required for MVP).

### `/dev/shm` sizing

Both deployments mount a 256 Mi `/dev/shm` emptyDir. If vLLM's engine IPC requires more (e.g., for larger models or higher concurrency), increase the `sizeLimit` in the relevant deployment YAML.

### OpenShift non-root SCC

The manifests already redirect cache directories for non-root clusters:
- `HOME=/tmp`
- `XDG_CACHE_HOME=/tmp/cache`

If you encounter permission errors, verify the pod's `securityContext` and ensure the `emptyDir` volumes are writable.

### `vllm bench serve` version mismatch

If `compare.py` fails with a message like `KeyError: 'some_metric'`, the vLLM version on the driver host may emit different result JSON fields than expected. Compare the real and sim result JSON files in the output directory to identify the missing or renamed field, and update `compare.py` to handle it.

The comparison logic reads result JSONs defensively and fails loudly naming the offending key if a required field is missing.
