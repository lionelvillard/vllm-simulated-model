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
  # shellcheck disable=SC2034
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
