#!/usr/bin/env bash
# Lifecycle driver for sim-vs-real work: deploy the in-cluster stack, run an
# evaluation sweep or an auto-tuning search against it, then tear it down.
#
# Usage:
#   eval.sh [setup|run|tune|promote|teardown|all] <variant-dir>
#   eval.sh <variant-dir>                 # shorthand for: all <variant-dir>
#
# Commands:
#   setup     Apply the variant's ConfigMap + real/sim Deployments and Services,
#             then wait for both Deployments to roll out.
#   run       Port-forward both Services and run the benchmark sweep (the stack
#             must already be up — e.g. from a prior `setup`).
#   tune      Full lifecycle: setup, auto-tune the physics `beta` parameters
#             against the real server, then teardown. Only meaningful for a
#             physics variant (needs the sim's POST /sim/config endpoint).
#   promote   Write the tuned beta from evaluation/results/<variant>/ into the
#             deployment's k8s/sim-config.json and k8s/configmap.yaml. No cluster
#             access needed; run this after `tune` to persist the result.
#   teardown  Delete the ConfigMap + real/sim Deployments and Services.
#   all       Full lifecycle: setup → run → teardown (the default).
#
# `all` and `tune` keep the (slow-to-load) real Deployment up on teardown so
# repeated runs reuse its weights; set KEEP=1 to skip teardown entirely, or use
# `teardown` to remove the real Deployment.
#
# <variant-dir> is a tuned-deployment directory such as
#   models/qwen3-32b/deployments/h100-sxm5/standalone/evaluation/physics
# It must contain:
#   - configmap.yaml          (sim model config; applied and recorded in the report)
#   - sim-config.json         (plain-JSON sim config; its `beta` is tuned by `tune`)
#   - real-service.yaml, sim-service.yaml   (entry-point services)
#
# Standalone variant (one pod per side):
#   - real-deployment.yaml, sim-deployment.yaml
#
# P/D disaggregated variant (prefill + decode pods per side):
#   - real-prefill-deployment.yaml, real-prefill-service.yaml, real-decode-deployment.yaml
#   - sim-prefill-deployment.yaml,  sim-prefill-service.yaml,  sim-decode-deployment.yaml
#
# Its parent layout <eval>/<variant> determines where outputs are written:
# <eval>/results/<variant> (e.g. evaluation/results/physics).
#
# Environment overrides (rarely needed): VLLM_SIM_NAMESPACE (or NAMESPACE),
# REAL_SVC, SIM_SVC, REAL_DEP, SIM_DEP, REAL_PORT, SIM_PORT, OUT, MODEL_CONFIG,
# SIM_CONFIG, SWEEP, KUBECTL, PYTHON, SETUP_TIMEOUT, HEALTH_TIMEOUT, KEEP.
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") [setup|run|tune|teardown|all] <variant-dir>" >&2
  exit 2
}

CMD="all"
case "${1:-}" in
  setup|run|tune|promote|teardown|all) CMD="$1"; shift ;;
  -h|--help) usage ;;
esac

[ "$#" -eq 1 ] || usage

VARIANT_DIR="$1"
if [ ! -d "$VARIANT_DIR" ]; then
  echo "ERROR: variant dir not found: $VARIANT_DIR" >&2
  exit 2
fi
VARIANT_DIR="$(cd "$VARIANT_DIR" && pwd)"
EVAL_MANIFESTS="$VARIANT_DIR"

VARIANT="$(basename "$VARIANT_DIR")"                     # e.g. physics
EVAL_DIR="$(dirname "$VARIANT_DIR")"                     # e.g. .../standalone/evaluation

# Extract a resource's metadata.name from a single-document manifest.
res_name() {  # <kind> <file>
  awk -v kind="$1" '
    $1=="kind:" { k=$2 }
    k==kind && $1=="name:" { print $2; exit }
  ' "$2"
}

# Detect P/D disaggregated layout by presence of prefill manifests.
PD_MODE=0
[ -f "$EVAL_MANIFESTS/sim-prefill-deployment.yaml" ] && PD_MODE=1

NAMESPACE="${VLLM_SIM_NAMESPACE:-${NAMESPACE:-}}"
REAL_SVC="${REAL_SVC:-$(res_name Service    "$EVAL_MANIFESTS/real-service.yaml")}"
SIM_SVC="${SIM_SVC:-$(res_name Service    "$EVAL_MANIFESTS/sim-service.yaml")}"

if [ "$PD_MODE" = "1" ]; then
  REAL_PREFILL_DEP="${REAL_PREFILL_DEP:-$(res_name Deployment "$EVAL_MANIFESTS/real-prefill-deployment.yaml")}"
  REAL_DEP="${REAL_DEP:-$(res_name Deployment "$EVAL_MANIFESTS/real-decode-deployment.yaml")}"
  SIM_PREFILL_DEP="${SIM_PREFILL_DEP:-$(res_name Deployment "$EVAL_MANIFESTS/sim-prefill-deployment.yaml")}"
  SIM_DEP="${SIM_DEP:-$(res_name Deployment "$EVAL_MANIFESTS/sim-decode-deployment.yaml")}"
else
  REAL_DEP="${REAL_DEP:-$(res_name Deployment "$EVAL_MANIFESTS/real-deployment.yaml")}"
  SIM_DEP="${SIM_DEP:-$(res_name Deployment "$EVAL_MANIFESTS/sim-deployment.yaml")}"
fi

REAL_PORT="${REAL_PORT:-9001}"
SIM_PORT="${SIM_PORT:-9002}"
SIM_TUNER_PORT="${SIM_TUNER_PORT:-9003}"
OUT="${OUT:-$EVAL_DIR/results/$VARIANT}"
MODEL_CONFIG="${MODEL_CONFIG:-$VARIANT_DIR/configmap.yaml}"
SIM_CONFIG="${SIM_CONFIG:-$VARIANT_DIR/sim-config.json}"
SWEEP="${SWEEP:-$EVAL_DIR/sweep.yaml}"
KUBECTL="${KUBECTL:-oc}"
PYTHON="${PYTHON:-}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-900s}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"

_missing=0
[ -z "$REAL_SVC" ] && _missing=1
[ -z "$SIM_SVC" ]  && _missing=1
[ -z "$REAL_DEP" ] && _missing=1
[ -z "$SIM_DEP" ]  && _missing=1
if [ "$PD_MODE" = "1" ]; then
  [ -z "${REAL_PREFILL_DEP:-}" ] && _missing=1
  [ -z "${SIM_PREFILL_DEP:-}" ]  && _missing=1
fi
if [ "$_missing" = "1" ]; then
  echo "ERROR: could not resolve resource names from $EVAL_MANIFESTS/*.yaml" >&2
  exit 2
fi

# Run from the repo root so `python -m evaluation.*` resolves.
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
kc() { "$KUBECTL" "${ns_flag[@]}" "$@"; }

echo "command:  $CMD"
echo "variant:  $VARIANT"
if [ "$PD_MODE" = "1" ]; then
  echo "mode:     P/D disaggregated"
  echo "real:     prefill/$REAL_PREFILL_DEP  decode/$REAL_DEP  svc/$REAL_SVC"
  echo "sim:      prefill/$SIM_PREFILL_DEP  decode/$SIM_DEP  svc/$SIM_SVC"
else
  echo "real:     dep/$REAL_DEP  svc/$REAL_SVC"
  echo "sim:      dep/$SIM_DEP  svc/$SIM_SVC"
fi
echo "out:      $OUT"

do_setup() {
  echo "== setup =="
  kc apply -f "$MODEL_CONFIG"
  if [ "$PD_MODE" = "1" ]; then
    kc apply -f "$EVAL_MANIFESTS/sim-prefill-deployment.yaml"
    kc apply -f "$EVAL_MANIFESTS/sim-prefill-service.yaml"
    kc apply -f "$EVAL_MANIFESTS/sim-decode-deployment.yaml"
    kc apply -f "$EVAL_MANIFESTS/sim-service.yaml"
    kc apply -f "$EVAL_MANIFESTS/real-prefill-deployment.yaml"
    kc apply -f "$EVAL_MANIFESTS/real-prefill-service.yaml"
    kc apply -f "$EVAL_MANIFESTS/real-decode-deployment.yaml"
    kc apply -f "$EVAL_MANIFESTS/real-service.yaml"
    echo "waiting for rollouts (timeout $SETUP_TIMEOUT; real downloads weights on first start)…"
    kc rollout status "deployment/$SIM_PREFILL_DEP"  --timeout="$SETUP_TIMEOUT"
    kc rollout status "deployment/$SIM_DEP"          --timeout="$SETUP_TIMEOUT"
    kc rollout status "deployment/$REAL_PREFILL_DEP" --timeout="$SETUP_TIMEOUT"
    kc rollout status "deployment/$REAL_DEP"         --timeout="$SETUP_TIMEOUT"
  else
    kc apply -f "$EVAL_MANIFESTS/sim-deployment.yaml"
    kc apply -f "$EVAL_MANIFESTS/sim-service.yaml"
    kc apply -f "$EVAL_MANIFESTS/real-deployment.yaml"
    kc apply -f "$EVAL_MANIFESTS/real-service.yaml"
    echo "waiting for rollouts (timeout $SETUP_TIMEOUT; real downloads weights on first start)…"
    kc rollout status "deployment/$SIM_DEP"  --timeout="$SETUP_TIMEOUT"
    kc rollout status "deployment/$REAL_DEP" --timeout="$SETUP_TIMEOUT"
  fi
}

# teardown <scope>: "all" removes real too; "sim" keeps the real Deployment up.
do_teardown() {
  local scope="${1:-all}"
  echo "== teardown ($scope) =="
  if [ "$PD_MODE" = "1" ]; then
    kc delete --ignore-not-found -f "$EVAL_MANIFESTS/sim-service.yaml"
    kc delete --ignore-not-found -f "$EVAL_MANIFESTS/sim-decode-deployment.yaml"
    kc delete --ignore-not-found -f "$EVAL_MANIFESTS/sim-prefill-service.yaml"
    kc delete --ignore-not-found -f "$EVAL_MANIFESTS/sim-prefill-deployment.yaml"
    kc delete --ignore-not-found -f "$MODEL_CONFIG"
    if [ "$scope" = "all" ]; then
      kc delete --ignore-not-found -f "$EVAL_MANIFESTS/real-service.yaml"
      kc delete --ignore-not-found -f "$EVAL_MANIFESTS/real-decode-deployment.yaml"
      kc delete --ignore-not-found -f "$EVAL_MANIFESTS/real-prefill-service.yaml"
      kc delete --ignore-not-found -f "$EVAL_MANIFESTS/real-prefill-deployment.yaml"
    fi
  else
    kc delete --ignore-not-found -f "$EVAL_MANIFESTS/sim-service.yaml"
    kc delete --ignore-not-found -f "$EVAL_MANIFESTS/sim-deployment.yaml"
    kc delete --ignore-not-found -f "$MODEL_CONFIG"
    if [ "$scope" = "all" ]; then
      kc delete --ignore-not-found -f "$EVAL_MANIFESTS/real-service.yaml"
      kc delete --ignore-not-found -f "$EVAL_MANIFESTS/real-deployment.yaml"
    fi
  fi
}

teardown_or_keep() {
  if [ -n "${KEEP:-}" ]; then
    echo "KEEP set — leaving the stack up (run 'teardown' to remove it)."
  else
    do_teardown sim
  fi
}

wait_health() {
  local port="$1" name="$2" deadline=$((SECONDS + HEALTH_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "$name ready on :$port"; return 0
    fi
    sleep 2
  done
  echo "ERROR: $name not ready on :$port after ${HEALTH_TIMEOUT}s" >&2; return 1
}

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

start_forwards() {
  kc port-forward "svc/$REAL_SVC" "$REAL_PORT:8000" >/dev/null 2>&1 &
  pids+=($!)
  kc port-forward "svc/$SIM_SVC" "$SIM_PORT:8000" >/dev/null 2>&1 &
  pids+=($!)
  wait_health "$REAL_PORT" real
  wait_health "$SIM_PORT"  sim
}

do_run() {
  echo "== run (eval sweep) =="
  echo "config: $MODEL_CONFIG"
  echo "sweep:  $SWEEP"
  start_forwards

  local sweep_flag=() model_config_flag=()
  [ -f "$SWEEP" ] && sweep_flag=(--sweep "$SWEEP")
  [ -n "$MODEL_CONFIG" ] && model_config_flag=(--model-config "$MODEL_CONFIG")

  "$PYTHON" -m evaluation.run_eval eval \
    --real-url "http://127.0.0.1:$REAL_PORT" \
    --sim-url "http://127.0.0.1:$SIM_PORT" \
    --out "$OUT" \
    "${sweep_flag[@]}" \
    "${model_config_flag[@]}"
}

do_promote() {
  echo "== promote (tuned beta → k8s config) =="
  "$PYTHON" -m evaluation.run_eval promote \
    --deployment-dir "$(dirname "$EVAL_DIR")" \
    --latency-model "$VARIANT"
}

do_tune() {
  echo "== tune (auto-tune physics beta) =="
  echo "sim-config: $SIM_CONFIG"
  if [ ! -f "$SIM_CONFIG" ]; then
    echo "ERROR: sim-config.json not found: $SIM_CONFIG" >&2
    exit 2
  fi
  start_forwards

  local tuner_url_flag=()
  if [ "$PD_MODE" = "1" ]; then
    # The P/D proxy (port 8000) does not forward /sim/config to the vllm
    # backend. Forward directly to the decode pod's vllm (port 8200) so
    # beta POSTs reach the tuner endpoint.
    kc port-forward "deployment/$SIM_DEP" "$SIM_TUNER_PORT:8200" >/dev/null 2>&1 &
    pids+=($!)
    wait_health "$SIM_TUNER_PORT" sim-tuner
    tuner_url_flag=(--sim-tuner-url "http://127.0.0.1:$SIM_TUNER_PORT")
  fi

  "$PYTHON" -m evaluation.tune \
    --real-url "http://127.0.0.1:$REAL_PORT" \
    --sim-url "http://127.0.0.1:$SIM_PORT" \
    --model-config "$SIM_CONFIG" \
    --out "$OUT" \
    "${tuner_url_flag[@]}"
}

case "$CMD" in
  setup)    do_setup ;;
  run)      do_run ;;
  promote)  do_promote ;;
  teardown) do_teardown all ;;
  all)      do_setup; do_run;  teardown_or_keep ;;
  tune)     do_setup; do_tune; teardown_or_keep ;;
esac
