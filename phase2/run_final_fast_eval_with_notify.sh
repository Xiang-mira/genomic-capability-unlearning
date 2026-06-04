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
LOG="${LOG:-$LOG_DIR/final_fast_eval_$(date +%Y%m%d_%H%M%S).log}"

COMMON_ARGS=(
  --benchmark-manifest data/benchmarks/final_fast_eval_manifest.csv
  --resume
  --device cuda:0
  --layers 3-9
  --batch-size 0
  --auto-batch-size 96
  --cpu-threads 16
  --probe-jobs 7
  --progress-every 25000
  --max-length 512
  --feature-cache-dir data/phase2/final_fast_eval/feature_cache
  --feature-cache-compression none
  --no-feature-cache-write
  --notify-sound
)

if [[ -n "${FEISHU_WEBHOOK:-}" ]]; then
  export FEISHU_WEBHOOK
fi
if [[ "${NOTIFY_ON_COMPLETE:-1}" != "0" ]]; then
  COMMON_ARGS+=(--notify-on-complete)
fi

RUNNER_ARGS=(
  --name final_fast_eval
  --sound
)
if [[ "${NOTIFY_ON_COMPLETE:-1}" != "0" ]]; then
  RUNNER_ARGS+=(--notify-success)
fi

echo "logging to $LOG"
"$PYTHON_BIN" -u phase2/run_with_notify.py "${RUNNER_ARGS[@]}" -- bash -lc '
  set -euo pipefail
  echo "=== final fast eval gd_full_ar5 $(date -Is) ==="
  "$1" -u phase2/eval_benchmarks.py \
    --ckpt data/phase2/checkpoints_tuned/gd_full_ar5/weights.safetensors \
    --out-dir data/phase2/final_fast_eval/gd_full_ar5 \
    "${@:3}"

  echo "=== final fast eval rmu_full_sc200 $(date -Is) ==="
  "$1" -u phase2/eval_benchmarks.py \
    --ckpt data/phase2/checkpoints_tuned/rmu_full_sc200/weights.safetensors \
    --out-dir data/phase2/final_fast_eval/rmu_full_sc200 \
    "${@:3}"

  echo "=== final fast eval complete $(date -Is) ==="
' bash "$PYTHON_BIN" -- "${COMMON_ARGS[@]}"
