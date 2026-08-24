#!/usr/bin/env bash
# Port-forward the in-cluster real+sim vLLM services and run the evaluation sweep.
set -euo pipefail

NAMESPACE="${NAMESPACE:-}"
REAL_SVC="${REAL_SVC:-vllm-qwen3-32b-standalone-eae748-real}"
SIM_SVC="${SIM_SVC:-vllm-qwen3-32b-standalone-eae748-sim}"
REAL_PORT="${REAL_PORT:-9001}"
SIM_PORT="${SIM_PORT:-9002}"
OUT="${OUT:-eval-out}"
KUBECTL="${KUBECTL:-oc}"
PYTHON="${PYTHON:-}"
MODEL_CONFIG="${MODEL_CONFIG:-}"

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
  "${model_config_flag[@]}"
