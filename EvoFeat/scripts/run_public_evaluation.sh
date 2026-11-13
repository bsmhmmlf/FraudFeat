#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
DATA=${1:?Usage: bash scripts/run_public_evaluation.sh <fraudfeat-data-dir> <run-dir> [run-dir ...]}
shift
if [[ $# -eq 0 ]]; then
  echo "At least one run directory is required" >&2
  exit 2
fi

DATA=$(resolve_input_path "$DATA")
POOL_LIMIT=${POOL_LIMIT:-100}
EVAL_PANEL=$(resolve_input_path "${EVAL_PANEL:-$DATA/splits/evaluation_panel.parquet}")
NO_EMBED=${NO_EMBED:-0}
NO_JUDGE=${NO_JUDGE:-0}
FORCE=${FORCE:-0}
FINAL_EVAL_ARGS=()
if [[ "$FORCE" == "1" ]]; then
  FINAL_EVAL_ARGS+=(--force)
fi
JUDGES=${JUDGES:-${FRAUDFEAT_JUDGE_MODEL:-deepseek-v4-pro,gpt-6.1-sol,claude-opus-5.5}}
EMBED_MODEL=${FRAUDFEAT_EMBED_MODEL:-Qwen3-Embedding-0.6B}
WORKERS=${WORKERS:-4}

for RUN in "$@"; do
  RUN=$(resolve_input_path "$RUN")
  "$PY" -m experiments.final_eval --run-dir "$RUN" --data "$DATA" \
    --evaluation-panel "$EVAL_PANEL" --limit "$POOL_LIMIT" "${FINAL_EVAL_ARGS[@]}"
  if [[ "$NO_EMBED" != "1" ]]; then
    "$PY" -m fraudfeat.embedding_diversity --features "$RUN/features.json" \
      --out-dir "$RUN" --pool-limit "$POOL_LIMIT" --iv-jsonl "$RUN/effectiveness.jsonl" \
      --model "$EMBED_MODEL"
  fi
  if [[ "$NO_JUDGE" != "1" ]]; then
    "$PY" -m fraudfeat.interpretability --features "$RUN/features.json" \
      --out-dir "$RUN" --schema "$DATA" --pool-limit "$POOL_LIMIT" \
      --iv-jsonl "$RUN/effectiveness.jsonl" --judges "$JUDGES" --workers "$WORKERS"
  fi
done
