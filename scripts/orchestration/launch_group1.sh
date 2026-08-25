#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-status}"
TASK="${2:-}"

PREFLIGHT_MODE="submit"
if [[ "$MODE" == "--smoke" ]]; then
  PREFLIGHT_MODE="smoke"
fi

if ! python scripts/orchestration/group1_status.py --preflight --mode "$PREFLIGHT_MODE"; then
  echo "Group 1 preflight failed; no experiments submitted." >&2
  exit 1
fi

if [[ "$MODE" == "status" ]]; then
  echo "Preflight passed. Use --smoke <task> for a one-task run or --submit after the 13-task manifest is populated."
  exit 0
fi

if [[ "$MODE" == "--smoke" ]]; then
  if [[ -z "$TASK" ]]; then
    MANIFEST="${VB_GENEB_TASK_MANIFEST:-${VB_GENEB_DIR:-${VB_OUT:-$ROOT/results_viral_bench}/geneb}/sentinel_tasks.csv}"
    if [[ -f "$MANIFEST" ]]; then
      TASK="$(python - "$MANIFEST" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        if row.get("task"):
            print(row["task"])
            break
PY
)"
    fi
  fi
  if [[ -z "$TASK" ]]; then
    echo "ERROR: --smoke requires a task argument or a sentinel_tasks.csv manifest with a task column." >&2
    exit 2
  fi
  cd scripts/common
  exec python capacity_sweep.py --dataset geneb --task "$TASK" --seeds 42 --epochs 1 --max_cells 1 --device "${VB_DEVICE:-cuda:0}"
fi

if [[ "$MODE" == "--submit" ]]; then
  echo "ERROR: full Group 1 submission is intentionally gated until a validated 13-task GENEB manifest exists and smoke passes." >&2
  echo "Run first: scripts/orchestration/launch_group1.sh --smoke <one_geneb_task>" >&2
  exit 3
fi

echo "Usage: $0 [--smoke <task>|--submit]" >&2
exit 2
