#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
DATA=${1:?Usage: bash scripts/run_public.sh <fraudfeat-data-dir> [run-id]}
RUN_ID=${2:-ours_public_s${SEED:-42}}
SEED=${SEED:-42}
EPISODES=${EPISODES:-25}
EPISODE_SIZE=${EPISODE_SIZE:-4}
TARGET_SAMPLE=${TARGET_SAMPLE:-100}
OFFLINE_PANEL=${OFFLINE_PANEL:-"$DATA/splits/offline_panel.parquet"}
EVAL_PANEL=${EVAL_PANEL:-"$DATA/splits/evaluation_panel.parquet"}
TARGETS=${TARGETS:-"$DATA/splits/target_streams_s${SEED}.json"}
DATA=$(resolve_input_path "$DATA")
OFFLINE_PANEL=$(resolve_input_path "$OFFLINE_PANEL")
EVAL_PANEL=$(resolve_input_path "$EVAL_PANEL")
TARGETS=$(resolve_input_path "$TARGETS")
NO_EMBED=${NO_EMBED:-0}
NO_JUDGE=${NO_JUDGE:-0}
FORCE=${FORCE:-0}

if [ ! -f "$TARGETS" ] || [ ! -f "$OFFLINE_PANEL" ] || [ ! -f "$EVAL_PANEL" ]; then
  echo "Official splits not found under $DATA/splits/ (target streams + panels)." >&2
  echo "The dataset ships them alongside schema.json; no preparation step is needed." >&2
  exit 2
fi
"$PY" -m system.main --data "$DATA" --offline-panel "$OFFLINE_PANEL" \
  --targets "$TARGETS" --run-id "$RUN_ID" --random-seed "$SEED" \
  --target-sample "$TARGET_SAMPLE" --max-episodes "$EPISODES" --episode-size "$EPISODE_SIZE"
NO_EMBED="$NO_EMBED" NO_JUDGE="$NO_JUDGE" EVAL_PANEL="$EVAL_PANEL" FORCE="$FORCE" \
  bash "$SCRIPT_DIR/run_public_evaluation.sh" "$DATA" "$REPO_ROOT/runs/$RUN_ID"
