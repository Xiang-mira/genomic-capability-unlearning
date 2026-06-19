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
FINAL_FAST_CANDIDATES="${FINAL_FAST_CANDIDATES:-}"
FINAL_FAST_GD_NAME="${FINAL_FAST_GD_NAME:-}"
FINAL_FAST_RMU_NAME="${FINAL_FAST_RMU_NAME:-}"
export FINAL_FAST_CANDIDATES FINAL_FAST_GD_NAME FINAL_FAST_RMU_NAME

COMMON_ARGS=(
  --benchmark-manifest data/benchmarks/final_fast_eval_manifest.csv
  --resume
  --device cuda:0
  --batch-size 1
  --cpu-threads 16
  --epochs 3
  --max-steps 0
  --eval-every 100
  --patience 3
  --lr 1e-4
  --weight-decay 0.0
  --lora-rank 8
  --lora-alpha 16
  --lora-dropout 0.0
  --metric-for-best auto
  --progress-every 1
  --max-length 512
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
  echo "=== final fast eval base $(date -Is) ==="
  "$1" -u phase2/eval_benchmarks.py \
    --out-dir data/phase2/final_fast_eval/base \
    "${@:3}"

  if [[ -z "${FINAL_FAST_CANDIDATES:-}" ]]; then
    echo "=== no final fast candidate checkpoints requested ==="
    echo "Set FINAL_FAST_CANDIDATES=\"run_a run_b\" to evaluate explicit checkpoints."
  else
    for run_name in $FINAL_FAST_CANDIDATES; do
      ckpt="data/phase2/checkpoints_tuned/${run_name}/weights.safetensors"
      if [[ ! -f "$ckpt" ]]; then
        echo "missing checkpoint: $ckpt" >&2
        exit 2
      fi
      if [[ "$run_name" == "gd_full_ar5" || "$run_name" == "rmu_full_sc200" ]]; then
        echo "warning: $run_name is a probe-selected legacy candidate; use only for reference."
      fi
      echo "=== final fast eval ${run_name} $(date -Is) ==="
      "$1" -u phase2/eval_benchmarks.py \
        --ckpt "$ckpt" \
        --out-dir "data/phase2/final_fast_eval/${run_name}" \
        "${@:3}"
    done
  fi

  if [[ -n "${FINAL_FAST_GD_NAME:-}" && -n "${FINAL_FAST_RMU_NAME:-}" ]]; then
    "$1" -u phase2/aggregate_hvue_lora.py \
      --base-dir data/phase2/final_fast_eval/base \
      --gd-dir "data/phase2/final_fast_eval/${FINAL_FAST_GD_NAME}" \
      --rmu-dir "data/phase2/final_fast_eval/${FINAL_FAST_RMU_NAME}" \
      --out-csv data/phase2/final_fast_eval/hvue_lora_comparison.csv
  else
    echo "=== skipping GD/RMU aggregate; set FINAL_FAST_GD_NAME and FINAL_FAST_RMU_NAME after LoRA-based candidate selection ==="
  fi

  echo "=== final fast eval complete $(date -Is) ==="
' bash "$PYTHON_BIN" -- "${COMMON_ARGS[@]}"
