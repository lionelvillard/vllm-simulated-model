#!/usr/bin/env bash
# Port-forward the in-cluster real+sim vLLM services and run the evaluation sweep.
#
# Usage:
#   run_eval.sh <variant-dir>
#
# <variant-dir> is a tuned-deployment directory such as
#   models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/latency/flat
# It must contain:
#   - configmap.yaml            (sim model config, recorded in the report)
#   - eval/real-service.yaml    (real model Service — its name is port-forwarded)
#   - eval/sim-service.yaml     (sim model Service — its name is port-forwarded)
# and its parent layout <eval>/<category>/<variant> determines where the report
# is written: <eval>/results/<category>/<variant> (e.g. evaluation/results/latency/flat).
#
# Environment overrides (rarely needed): NAMESPACE, REAL_SVC, SIM_SVC,
# REAL_PORT, SIM_PORT, OUT, MODEL_CONFIG, SWEEP, KUBECTL, PYTHON.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $(basename "$0") <variant-dir>" >&2
  exit 2
fi

VARIANT_DIR="$1"
if [ ! -d "$VARIANT_DIR" ]; then
  echo "ERROR: variant dir not found: $VARIANT_DIR" >&2
  exit 2
fi
VARIANT_DIR="$(cd "$VARIANT_DIR" && pwd)"

VARIANT="$(basename "$VARIANT_DIR")"                     # e.g. flat
CATEGORY="$(basename "$(dirname "$VARIANT_DIR")")"       # e.g. latency
EVAL_DIR="$(dirname "$(dirname "$VARIANT_DIR")")"        # e.g. .../standalone/evaluation

# Extract a resource's metadata.name from a single-document manifest.
svc_name() {
  awk '/^kind:[[:space:]]*Service/{s=1} s && /^[[:space:]]+name:/{print $2; exit}' "$1"
}

NAMESPACE="${NAMESPACE:-}"
REAL_SVC="${REAL_SVC:-$(svc_name "$VARIANT_DIR/eval/real-service.yaml")}"
SIM_SVC="${SIM_SVC:-$(svc_name "$VARIANT_DIR/eval/sim-service.yaml")}"
REAL_PORT="${REAL_PORT:-9001}"
SIM_PORT="${SIM_PORT:-9002}"
OUT="${OUT:-$EVAL_DIR/results/$CATEGORY/$VARIANT}"
MODEL_CONFIG="${MODEL_CONFIG:-$VARIANT_DIR/configmap.yaml}"
SWEEP="${SWEEP:-$EVAL_DIR/sweep.yaml}"
KUBECTL="${KUBECTL:-oc}"
PYTHON="${PYTHON:-}"

if [ -z "$REAL_SVC" ] || [ -z "$SIM_SVC" ]; then
  echo "ERROR: could not resolve service names from $VARIANT_DIR/eval/*.yaml" >&2
  exit 2
fi

# Run from the repo root so `python -m evaluation.run_eval` resolves.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo venv so subprocess calls (vllm bench serve via sys.executable)
# use the same Python that has vllm installed from source.
if [ -z "$PYTHON" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

echo "variant:  $CATEGORY/$VARIANT"
echo "real svc: $REAL_SVC   sim svc: $SIM_SVC"
echo "config:   $MODEL_CONFIG"
echo "sweep:    $SWEEP"
echo "out:      $OUT"

ns_flag=()
[ -n "$NAMESPACE" ] && ns_flag=(-n "$NAMESPACE")

sweep_flag=()
[ -f "$SWEEP" ] && sweep_flag=(--sweep "$SWEEP")

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

"$KUBECTL" "${ns_flag[@]}" port-forward "svc/$REAL_SVC" "$REAL_PORT:8000" >/dev/null 2>&1 &
pids+=($!)
"$KUBECTL" "${ns_flag[@]}" port-forward "svc/$SIM_SVC" "$SIM_PORT:8000" >/dev/null 2>&1 &
pids+=($!)

wait_health() {
  local port="$1" name="$2" deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "$name ready on :$port"; return 0
    fi
    sleep 2
  done
  echo "ERROR: $name not ready on :$port after 120 s" >&2; return 1
}

wait_health "$REAL_PORT" real
wait_health "$SIM_PORT"  sim

model_config_flag=()
[ -n "$MODEL_CONFIG" ] && model_config_flag=(--model-config "$MODEL_CONFIG")

"$PYTHON" -m evaluation.run_eval \
  --real-url "http://127.0.0.1:$REAL_PORT" \
  --sim-url "http://127.0.0.1:$SIM_PORT" \
  --out "$OUT" \
  "${sweep_flag[@]}" \
  "${model_config_flag[@]}"
