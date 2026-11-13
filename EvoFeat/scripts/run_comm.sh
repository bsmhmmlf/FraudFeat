#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
DATA=${1:-${DATA:-}}
SEEDS=${SEEDS:-42 43 44}
for seed in $SEEDS; do
  configure_backend "$seed"
  "$PY" -m experiments.baseline_runner --baseline comm --run-id "comm__s${seed}" --random-seed "$seed" "${BACKEND_ARGS[@]}" --account-type individual --target-sample 100 --valid-target 100
done
