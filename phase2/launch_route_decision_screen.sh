#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PHASE2_PYTHON:-${PROJECT_PYTHON:-python}}"
PYTHON_DIR="$(dirname "$PYTHON_BIN")"
OUT_DIR="${ROUTE_DECISION_OUT_DIR:-data/phase2/route_decision_20260715}"
DEVICE="${DEVICE:-cuda:0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="${ROUTE_DECISION_SCREEN_NAME:-route_decision_${STAMP}}"
LOG_ROOT="${ROUTE_DECISION_LOG_ROOT:-$PROJECT_ROOT/logs/route_decision}"

mkdir -p "$LOG_ROOT"
STDOUT_LOG="$LOG_ROOT/${SESSION}.stdout.log"
STDERR_LOG="$LOG_ROOT/${SESSION}.stderr.log"
LAUNCH_LOG="$LOG_ROOT/${SESSION}.launch.log"

cd "$PROJECT_ROOT"

if ! command -v screen >/dev/null 2>&1; then
  echo "[route-screen] screen is not installed" >&2
  exit 1
fi

if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "[route-screen] session already exists: $SESSION" >&2
  exit 1
fi

CMD="export PATH='$PYTHON_DIR':\"\$PATH\" && '$PYTHON_BIN' -u phase2/run_route_decision_pipeline.py --project-root '$PROJECT_ROOT' --python-bin '$PYTHON_BIN' --out-dir '$OUT_DIR' --device '$DEVICE'"
echo "[route-screen] launching session=$SESSION" | tee "$LAUNCH_LOG"
echo "[route-screen] command=$CMD" | tee -a "$LAUNCH_LOG"
screen -dmS "$SESSION" bash -lc "cd '$PROJECT_ROOT' && $CMD >>'$STDOUT_LOG' 2>>'$STDERR_LOG'"
echo "[route-screen] stdout=$STDOUT_LOG" | tee -a "$LAUNCH_LOG"
echo "[route-screen] stderr=$STDERR_LOG" | tee -a "$LAUNCH_LOG"
echo "[route-screen] session=$SESSION" | tee -a "$LAUNCH_LOG"
