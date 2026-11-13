#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
RUNS=${RUNS:-runs}
POOL_LIMIT=${POOL_LIMIT:-100}
WORKERS=${WORKERS:-8}
STABILITY=${STABILITY:-20}
METHODS=${METHODS:-ours,manual,stat,comm,base_agent,static_agent,wo_exp,wo_syn,wo_eval,wo_cov}
for dir in "$RUNS"/*; do
  if [ -d "$dir" ] && [ -f "$dir/features.json" ]; then
    name=$(basename "$dir")
    method=${name%%__s*}
    if [[ "$name" == ours_rq2__s* ]]; then
      "$PY" -X utf8 -m experiments.final_eval --run-dir "$dir" --limit 0
      "$PY" -X utf8 -m experiments.diversity_embedding "$dir" --pool-limit 0 --curve-only
      "$PY" -X utf8 -m experiments.report ours_rq2 --runs "$RUNS" --gen --pool-limit 0
    else
      "$PY" -X utf8 -m experiments.final_eval --run-dir "$dir" --limit "$POOL_LIMIT"
      "$PY" -X utf8 -m experiments.diversity_embedding "$dir" --pool-limit "$POOL_LIMIT"
      "$PY" -X utf8 -m experiments.report "$method" --runs "$RUNS" --gen --pool-limit "$POOL_LIMIT"
    fi
  fi
done
"$PY" -X utf8 -m experiments.judge --runs "$RUNS" --methods "$METHODS" --pool-limit "$POOL_LIMIT" --stability-features "$STABILITY" --stage requests
"$PY" -X utf8 -m experiments.judge --runs "$RUNS" --methods "$METHODS" --pool-limit "$POOL_LIMIT" --stability-features "$STABILITY" --stage score --workers "$WORKERS"
"$PY" -X utf8 -m experiments.judge --runs "$RUNS" --methods "$METHODS" --pool-limit "$POOL_LIMIT" --stage report
"$PY" -X utf8 -m experiments.aggregate "$RUNS"
