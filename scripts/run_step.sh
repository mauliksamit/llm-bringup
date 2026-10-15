#!/bin/bash
# Full evidence for one ladder step: numerics, bench run 1 + run 2, drift
# check, profiler trace. Usage: bash scripts/run_step.sh <step> [python]
set -u
STEP="$1"
PY="${2:-python}"
"$PY" scripts/numerics.py --step "$STEP" || exit 1
"$PY" scripts/bench.py --step "$STEP" --run 1 || exit 1
"$PY" scripts/bench.py --step "$STEP" --run 2 || exit 1
"$PY" scripts/check_drift.py --step "$STEP"; echo "drift exit=$?"
"$PY" scripts/trace_step.py --step "$STEP" || exit 1
