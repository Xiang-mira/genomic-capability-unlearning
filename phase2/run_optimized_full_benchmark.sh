#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PHASE2_PYTHON:-${PROJECT_PYTHON:-python}}"
OUT_ROOT="${FULL_BENCHMARK_OUT_ROOT:-data/phase2/full_benchmarks_lora_optimized_s600}"
PROFILE_PATH="$OUT_ROOT/batch_profile.json"
PREFLIGHT_LOG="$OUT_ROOT/batch_preflight.log"
mkdir -p "$OUT_ROOT" logs

stop_vllm() {
  mapfile -t vllm_pids < <(
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, '$2 ~ /VLLM/ {gsub(/[[:space:]]/, "", $1); print $1}'
  )
  if ((${#vllm_pids[@]} == 0)); then
    echo "[optimized-full] no VLLM GPU process found"
    return
  fi
  echo "[optimized-full] stopping VLLM pid(s): ${vllm_pids[*]}"
  kill -TERM "${vllm_pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    alive=0
    for pid in "${vllm_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=1; fi
    done
    if ((alive == 0)); then return; fi
    sleep 1
  done
  echo "[optimized-full] VLLM did not stop after 60s; sending SIGKILL"
  kill -KILL "${vllm_pids[@]}" 2>/dev/null || true
  sleep 2
}

write_profile() {
  "$PYTHON_BIN" - "$PROFILE_PATH" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "train_batch_size": int(sys.argv[2]),
    "eval_batch_size": int(sys.argv[3]),
    "validation_max_rows": 2000,
    "eval_every_optimizer_steps": 200,
    "max_steps": 600,
    "seed": 42,
    "max_length": 512,
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

read_profile() {
  "$PYTHON_BIN" - "$PROFILE_PATH" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
print(payload["train_batch_size"], payload["eval_batch_size"])
PY
}

run_preflight() {
  local train_batch=$1
  local eval_batch=$2
  echo "[optimized-full] preflight train=$train_batch eval=$eval_batch" | tee -a "$PREFLIGHT_LOG"
  "$PYTHON_BIN" -u phase2/eval_benchmarks.py \
    --preflight-only \
    --device cuda:0 \
    --cpu-threads 16 \
    --train-batch-size "$train_batch" \
    --eval-batch-size "$eval_batch" \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.0 \
    --max-length 512 \
    --seed 42 >>"$PREFLIGHT_LOG" 2>&1
}

stop_vllm

if [[ -f "$PROFILE_PATH" ]]; then
  read -r TRAIN_BATCH EVAL_BATCH < <(read_profile)
  echo "[optimized-full] reusing batch profile train=$TRAIN_BATCH eval=$EVAL_BATCH"
else
  if run_preflight 8 32; then
    TRAIN_BATCH=8
    EVAL_BATCH=32
  elif run_preflight 4 16; then
    TRAIN_BATCH=4
    EVAL_BATCH=16
  else
    echo "[optimized-full] both batch profiles failed; see $PREFLIGHT_LOG" >&2
    exit 1
  fi
  write_profile "$TRAIN_BATCH" "$EVAL_BATCH"
  echo "[optimized-full] selected global batch profile train=$TRAIN_BATCH eval=$EVAL_BATCH"
fi

# Full suite minus the two Calici tasks excluded by the final benchmark protocol.
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
  --full-runs \
    lora_gd_full_ar5_s500 \
    lora_gd_full_ar3_s200 \
    lora_rmu_full_sc200_s200 \
    lora_rmu_full_sc50_s200 \
  --rank-full-results \
  --full-base-out-dir "$OUT_ROOT/base" \
  --full-out-root "$OUT_ROOT" \
  --device cuda:0 \
  --batch-size 1 \
  --train-batch-size "$TRAIN_BATCH" \
  --eval-batch-size "$EVAL_BATCH" \
  --validation-max-rows 2000 \
  --discard-task-checkpoint \
  --cpu-threads 16 \
  --epochs 3 \
  --max-steps 600 \
  --eval-every 200 \
  --patience 3 \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --metric-for-best auto \
  --max-length 512 \
  --seed 42
