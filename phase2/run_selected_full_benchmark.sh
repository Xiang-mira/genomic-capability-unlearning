#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PHASE2_PYTHON:-/home/teacher1/miniconda3/envs/UT-p1/bin/python}"
OUT_ROOT="${FULL_BENCHMARK_OUT_ROOT:-data/phase2/full_benchmarks_lora_selected}"

# Full suite minus the two Calici tasks excluded by the final benchmark plan.
TASK_FILTER="$($PYTHON_BIN - <<'PY'
import csv
import sys

csv.field_size_limit(sys.maxsize)
excluded = {
    "hvue_human_transmissibility_caliciviridae",
    "hvue_human_virus_pathogenicity_bvbrc_calici",
}
tasks = set()
with open("data/benchmarks/hvue_gue_manifest.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["task"] not in excluded:
            tasks.add(row["task"])
print(",".join(sorted(tasks)))
PY
)"

exec "$PYTHON_BIN" -u phase2/run_benchmark_pilot.py full-top \
  --full-manifest data/benchmarks/hvue_gue_manifest.csv \
  --benchmark-scope all \
  --task-filter "$TASK_FILTER" \
  --ckpt-root data/phase2/checkpoints_lora_grid \
  --rankings-json data/phase2/benchmark_pilot_lora/pilot_rankings.json \
  --top-k-per-method 2 \
  --rank-full-results \
  --full-base-out-dir "$OUT_ROOT/base" \
  --full-out-root "$OUT_ROOT" \
  --device cuda:0 \
  --batch-size 1 \
  --cpu-threads 16 \
  --epochs 3 \
  --eval-every 100 \
  --patience 3 \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --metric-for-best auto \
  --max-length 512 \
  --seed 42
