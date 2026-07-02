#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

MAX_RESTARTS="${MAX_RESTARTS:-10}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-60}"
OUT_ROOT="${FULL_BENCHMARK_OUT_ROOT:-data/phase2/full_benchmarks_lora_optimized_s600}"
WATCHDOG_LOG="${WATCHDOG_LOG:-logs/optimized_full_watchdog.log}"
mkdir -p logs

log() {
  printf '%s [optimized-watchdog] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCHDOG_LOG"
}

pipeline_complete() {
  /home/teacher1/miniconda3/envs/UT-p1/bin/python - "$OUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
profile_path = root / "batch_profile.json"
rankings_path = root / "full_rankings.json"
run_names = [
    "base",
    "lora_gd_full_ar5_s500",
    "lora_gd_full_ar3_s200",
    "lora_rmu_full_sc200_s200",
    "lora_rmu_full_sc50_s200",
]
if not profile_path.exists() or not rankings_path.exists():
    raise SystemExit(1)
profile = json.loads(profile_path.read_text())
if profile.get("max_steps") != 600 or profile.get("eval_every_optimizer_steps") != 200:
    raise SystemExit(1)
for run_name in run_names:
    result_path = root / run_name / "eval_benchmarks.csv"
    if not result_path.exists():
        raise SystemExit(1)
    with result_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 44:
        raise SystemExit(1)
    if any(int(row["train_batch_size"]) != profile["train_batch_size"] for row in rows):
        raise SystemExit(1)
    if any(int(row["eval_batch_size"]) != profile["eval_batch_size"] for row in rows):
        raise SystemExit(1)
    if any(int(row["n_val_early_stop"]) > 2000 for row in rows):
        raise SystemExit(1)
    if any(str(row["checkpoint_retained"]).lower() not in {"false", "0"} for row in rows):
        raise SystemExit(1)

rankings = json.loads(rankings_path.read_text())
rows = rankings.get("rows", [])
ok = (
    len(rows) == 4
    and all(row.get("n_primary_pairs") == 2 for row in rows)
    and all(row.get("n_secondary_pairs") == 3 for row in rows)
    and all(row.get("n_gue_pairs") == 33 for row in rows)
    and all(row.get("n_viral_pairs") == 6 for row in rows)
)
raise SystemExit(0 if ok else 1)
PY
}

if pipeline_complete; then
  log "verified optimized full results already complete"
  exit 0
fi

for ((attempt = 1; attempt <= MAX_RESTARTS; attempt++)); do
  run_log="logs/optimized_full_attempt_${attempt}_$(date -u +%Y%m%d_%H%M%S).log"
  log "starting/resuming full-only attempt=$attempt/$MAX_RESTARTS run_log=$run_log"
  bash phase2/run_optimized_full_benchmark.sh >>"$run_log" 2>&1
  status=$?
  if pipeline_complete; then
    log "optimized full benchmark passed verification on attempt=$attempt"
    exit 0
  fi
  log "attempt=$attempt exited status=$status; retrying in ${RETRY_DELAY_SEC}s"
  sleep "$RETRY_DELAY_SEC"
done

log "optimized full benchmark failed after $MAX_RESTARTS attempts"
exit 1
