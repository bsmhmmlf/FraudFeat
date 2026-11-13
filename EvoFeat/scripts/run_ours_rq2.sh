#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
DATA=${1:-${DATA:-}}
SEEDS=${SEEDS:-42 43 44}
for seed in $SEEDS; do
  configure_backend "$seed"
  "$PY" -m system.main --run-id "ours_rq2__s${seed}" --random-seed "$seed" "${BACKEND_ARGS[@]}" --account-type individual --target-sample 100 --max-episodes 25 --episode-size 4 --max-hops 5
done
