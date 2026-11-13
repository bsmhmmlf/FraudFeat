#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PY=${PY:-python}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

configure_backend() {
  local seed=$1
  : "${DATA:?Pass the FraudFeat dataset path as the first argument or set DATA}"
  if [ ! -f "$DATA/splits/target_streams_s${seed}.json" ]; then
    echo "Official splits not found: $DATA/splits/target_streams_s${seed}.json" >&2
    echo "The dataset ships fixed discovery targets and panels under data/splits/." >&2
    exit 2
  fi
  BACKEND_ARGS=(
    --data "$DATA"
    --offline-panel "${OFFLINE_PANEL:-$DATA/splits/offline_panel.parquet}"
    --targets "${TARGETS:-$DATA/splits/target_streams_s${seed}.json}"
  )
}
