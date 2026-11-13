#!/usr/bin/env bash

CALLER_DIR=$PWD
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PY=${PY:-python}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

resolve_input_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$CALLER_DIR/$1" ;;
  esac
}

cd "$REPO_ROOT"

configure_backend() {
  local seed=$1
  : "${DATA:?Pass the FraudFeat dataset path as the first argument or set DATA}"
  DATA=$(resolve_input_path "$DATA")
  local targets
  targets=$(resolve_input_path "${TARGETS:-$DATA/splits/target_streams_s${seed}.json}")
  if [ ! -f "$targets" ]; then
    echo "Official splits not found: $targets" >&2
    echo "The dataset ships fixed discovery targets and panels under data/splits/." >&2
    exit 2
  fi
  BACKEND_ARGS=(
    --data "$DATA"
    --offline-panel "$(resolve_input_path "${OFFLINE_PANEL:-$DATA/splits/offline_panel.parquet}")"
    --targets "$targets"
  )
}
