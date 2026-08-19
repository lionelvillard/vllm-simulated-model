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
  --out deploy/eval/sim-configmap.yaml
```

**Flags:**
- `--model <HF id or path to config.json>` (required): the real model to match
- `--out <path>` (required): output ConfigMap YAML path
- `--tp <int>` (default: 1): tensor-parallel degree
- `--peak-tflops <float>` (default: 989.0): H100 SXM5 peak TFLOPs
- `--hbm-gbps <float>` (default: 3350.0): H100 SXM5 HBM bandwidth (GB/s)
- `--weight-dtype <str>` (default: bfloat16): weight data type

To change the target model or GPU spec, adjust these flags and regenerate. The default ConfigMap name is `vllm-sim-model-config`.

## Deploy

Apply the manifests to deploy both the real GPU server and the simulated CPU server:

```bash
oc apply -n <namespace> -f deploy/eval/
```

This creates:
- `vllm-real` Deployment and Service (H100 GPU, real Qwen/Qwen3-32B weights)
- `vllm-sim` Deployment and Service (CPU, simulated model with generated config)
- `vllm-sim-model-config` ConfigMap (from the previous step)

**Note:** The real server downloads ~64 GB of weights on first startup. Readiness may take 5–10 minutes depending on network bandwidth. Watch pod status:

```bash
oc get pods -n <namespace> -w
```

## Run the evaluation (default, local)

The default approach uses `run_eval.sh`, which port-forwards both Services to localhost and runs the sweep from your machine:

```bash
evaluation/run_eval.sh
```

**Environment knobs** (with defaults):
- `NAMESPACE` (empty): OpenShift namespace
- `REAL_SVC=vllm-real`: real server Service name
- `SIM_SVC=vllm-sim`: sim server Service name
- `REAL_PORT=9001`: local port for real server
- `SIM_PORT=9002`: local port for sim server
- `OUT=eval-out`: output directory for reports and result JSONs
- `KUBECTL=oc`: the Kubernetes CLI to use (e.g., `kubectl` or `oc`)

Example with a specific namespace:

```bash
NAMESPACE=my-namespace OUT=results evaluation/run_eval.sh
```

The script waits for both Services' `/health` endpoints to respond, then runs `run_eval.py` to execute the benchmark sweep. Results land in `$OUT/report.md` and `$OUT/report.json`.

## Run the evaluation (in-cluster)

Alternatively, run the driver fully in-cluster via a Job:

```bash
oc apply -n <namespace> -f deploy/eval/benchmark-job.yaml
```

The Job is named **`vllm-benchmark`**. View logs with:

```bash
oc logs -n <namespace> job/vllm-benchmark -f
```

Report artifacts are written to `/out` in the pod (backed by an emptyDir). To retrieve them:

```bash
pod=$(oc get pod -n <namespace> -l app=vllm-benchmark -o jsonpath='{.items[0].metadata.name}')
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
2. Update `deploy/eval/real-deployment.yaml` to request the correct model and GPU type.
3. Redeploy: `oc apply -n <namespace> -f deploy/eval/`

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
- Increase the readiness probe `initialDelaySeconds` in `deploy/eval/real-deployment.yaml` if the default is insufficient.
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
