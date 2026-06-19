#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-/home/teacher1/miniconda3/envs/UT-p1/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LAUNCH_LOG="$LOG_DIR/final_fast_eval_launcher_$(date +%Y%m%d_%H%M%S).log"
FINAL_FAST_CANDIDATES="${FINAL_FAST_CANDIDATES:-}"

setsid env LOG="$LAUNCH_LOG" ./phase2/run_final_fast_eval_with_notify.sh > "$LAUNCH_LOG" 2>&1 &
RUN_PID=$!

WATCHDOG_ARGS=(
  --name final_fast_eval
  --pid "$RUN_PID"
  --progress data/phase2/final_fast_eval/base/eval_benchmarks_progress.json
  --log "$LAUNCH_LOG"
  --poll-sec 30
  --grace-sec 15
  --sound
)
for run_name in $FINAL_FAST_CANDIDATES; do
  WATCHDOG_ARGS+=(--progress "data/phase2/final_fast_eval/${run_name}/eval_benchmarks_progress.json")
done
if [[ "${NOTIFY_ON_COMPLETE:-1}" != "0" ]]; then
  WATCHDOG_ARGS+=(--notify-success)
fi

nohup "$PYTHON_BIN" -u phase2/watch_eval_progress.py "${WATCHDOG_ARGS[@]}" >> "$LAUNCH_LOG" 2>&1 &
WATCH_PID=$!

echo "run_pid=$RUN_PID"
echo "watchdog_pid=$WATCH_PID"
echo "launcher_log=$LAUNCH_LOG"
