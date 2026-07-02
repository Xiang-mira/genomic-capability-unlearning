#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

INITIAL_PID="${1:-}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-60}"
WATCHDOG_LOG="${WATCHDOG_LOG:-logs/hvue_pipeline_watchdog.log}"

mkdir -p logs

log() {
  printf '%s [watchdog] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCHDOG_LOG"
}

pipeline_complete() {
  /home/teacher1/miniconda3/envs/UT-p1/bin/python - <<'PY'
import json
from pathlib import Path

pilot_path = Path("data/phase2/benchmark_pilot_lora/pilot_rankings.json")
full_path = Path("data/phase2/full_benchmarks_lora_selected/full_rankings.json")
if not pilot_path.exists() or not full_path.exists():
    raise SystemExit(1)

pilot = json.loads(pilot_path.read_text())
full = json.loads(full_path.read_text())
pilot_rows = pilot.get("rows", [])
full_rows = full.get("rows", [])

pilot_ok = (
    len(pilot_rows) == 12
    and all(row.get("n_primary_pairs") == 2 for row in pilot_rows)
    and all(row.get("n_secondary_pairs") == 3 for row in pilot_rows)
    and all(row.get("n_gue_pairs", 0) >= 1 for row in pilot_rows)
)
full_ok = (
    len(full_rows) == 4
    and all(row.get("n_primary_pairs") == 2 for row in full_rows)
    and all(row.get("n_secondary_pairs") == 3 for row in full_rows)
    and all(row.get("n_gue_pairs") == 33 for row in full_rows)
    and all(row.get("n_viral_pairs") == 6 for row in full_rows)
)
raise SystemExit(0 if pilot_ok and full_ok else 1)
PY
}

if pipeline_complete; then
  log "verified complete results already exist"
  exit 0
fi

if [[ -n "$INITIAL_PID" ]] && kill -0 "$INITIAL_PID" 2>/dev/null; then
  log "watching existing pipeline pid=$INITIAL_PID"
  while kill -0 "$INITIAL_PID" 2>/dev/null; do
    sleep 60
  done
  if pipeline_complete; then
    log "existing pipeline completed and passed result verification"
    exit 0
  fi
  log "existing pipeline exited before verified completion; switching to auto-resume"
fi

for ((attempt = 1; attempt <= MAX_RESTARTS; attempt++)); do
  run_log="logs/hvue_complete_selection_and_full_retry_${attempt}_$(date -u +%Y%m%d_%H%M%S).log"
  log "starting/resuming pipeline attempt=$attempt/$MAX_RESTARTS run_log=$run_log"
  bash phase2/run_hvue_complete_selection_and_full.sh >>"$run_log" 2>&1
  status=$?
  if pipeline_complete; then
    log "pipeline passed final verification on attempt=$attempt"
    exit 0
  fi
  log "attempt=$attempt exited status=$status without complete results; retrying in ${RETRY_DELAY_SEC}s"
  sleep "$RETRY_DELAY_SEC"
done

log "pipeline did not complete after $MAX_RESTARTS automatic resume attempts"
exit 1
