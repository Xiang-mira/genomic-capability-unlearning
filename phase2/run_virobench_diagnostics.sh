#!/usr/bin/env bash
set -euo pipefail

# Orchestrates the ViroBench diagnostic workflow. Long-running commands are
# intentionally explicit so they can be resumed or copied into a scheduler.

PYTHON=${PYTHON:-/home/teacher1/miniconda3/envs/UT-p1/bin/python}
DEVICE=${DEVICE:-cuda:0}
OUT_ROOT=${OUT_ROOT:-data/phase2/virobench_diagnostics}
CKPT_ROOT=${CKPT_ROOT:-data/phase2/checkpoints_tuned}
BENCHMARK_MANIFEST=${BENCHMARK_MANIFEST:-data/benchmarks/hvue_gue_manifest.csv}
LAYERS=${LAYERS:-3-9}
SAVE_STEPS=${SAVE_STEPS:-100,200,500,1000}

run_internal_eval() {
  local ckpt="$1"
  "$PYTHON" phase2/eval_unlearn.py \
    --ckpt "$ckpt" \
    --device "$DEVICE" \
    --batch-size 4 \
    --max-length 512 \
    --layers 0-10
}

run_external_eval() {
  local ckpt="$1"
  local out_dir="$2"
  "$PYTHON" -u phase2/eval_benchmarks_probe_legacy.py \
    --ckpt "$ckpt" \
    --benchmark-manifest "$BENCHMARK_MANIFEST" \
    --out-dir "$out_dir" \
    --resume \
    --device "$DEVICE" \
    --layers "$LAYERS" \
    --batch-size 0 \
    --auto-batch-size 64 \
    --cpu-threads 16 \
    --probe-jobs 7 \
    --progress-every 25000 \
    --max-length 512 \
    --feature-cache-dir "$OUT_ROOT/feature_cache"
}

case "${1:-help}" in
  train-gd-localized)
    "$PYTHON" phase2/unlearn_gd.py --out-dir "$CKPT_ROOT" --run-name gd_localized_ar5_s1000 \
      --condition localized --alpha-forget 1 --alpha-retain 5 --steps 1000 \
      --lr 1e-5 --device "$DEVICE" --save-steps "$SAVE_STEPS"
    ;;
  train-gd-random)
    "$PYTHON" phase2/unlearn_gd.py --out-dir "$CKPT_ROOT" --run-name gd_random_ar5 \
      --condition random --alpha-forget 1 --alpha-retain 5 --steps 1000 \
      --lr 1e-5 --device "$DEVICE" --save-steps "$SAVE_STEPS"
    ;;
  train-rmu-localized)
    "$PYTHON" phase2/unlearn_rmu.py --out-dir "$CKPT_ROOT" --run-name rmu_localized_sc50_l4 \
      --condition localized --target-layer 4 --steer-coef 50 --steps 1000 \
      --lr 1e-5 --device "$DEVICE" --save-steps "$SAVE_STEPS"
    ;;
  train-rmu-random)
    "$PYTHON" phase2/unlearn_rmu.py --out-dir "$CKPT_ROOT" --run-name rmu_random_sc50 \
      --condition random --target-layer 6 --steer-coef 50 --steps 1000 \
      --lr 1e-5 --device "$DEVICE" --save-steps "$SAVE_STEPS"
    ;;
  eval-checkpoint)
    ckpt="${2:?usage: $0 eval-checkpoint path/to/weights.safetensors}"
    out_dir="${3:-$(dirname "$ckpt")}"
    run_internal_eval "$ckpt"
    run_external_eval "$ckpt" "$out_dir"
    ;;
  aggregate)
    "$PYTHON" phase2/aggregate_trajectory.py --checkpoint-roots "$CKPT_ROOT" --out-dir "$OUT_ROOT"
    "$PYTHON" phase2/plot_convergence_diagnostics.py --checkpoint-roots "$CKPT_ROOT" --out-dir "$OUT_ROOT"
    "$PYTHON" phase2/summarize_controlled_splits.py --out "$OUT_ROOT/host_tropism_controlled_split_results.csv"
    ;;
  probe-vs-sft)
    "$PYTHON" phase2/probe_vs_sft.py \
      --benchmark-manifest "$BENCHMARK_MANIFEST" \
      --out-dir "$OUT_ROOT/probe_vs_sft" \
      --device "$DEVICE" \
      --checkpoints "gd_localized_ar5_s1000=$CKPT_ROOT/gd_localized_ar5_s1000/weights.safetensors,rmu_localized_sc50_l4=$CKPT_ROOT/rmu_localized_sc50_l4/weights.safetensors,gd_random_ar5=$CKPT_ROOT/gd_random_ar5/weights.safetensors,rmu_random_sc50=$CKPT_ROOT/rmu_random_sc50/weights.safetensors" \
      --feature-cache-dir "$OUT_ROOT/probe_vs_sft_feature_cache"
    ;;
  help|*)
    cat <<EOF
Usage: $0 <command>

Commands:
  train-gd-localized
  train-gd-random
  train-rmu-localized
  train-rmu-random
  eval-checkpoint <weights.safetensors> [out_dir]
  aggregate
  probe-vs-sft

Environment overrides:
  PYTHON=$PYTHON
  DEVICE=$DEVICE
  CKPT_ROOT=$CKPT_ROOT
  BENCHMARK_MANIFEST=$BENCHMARK_MANIFEST
EOF
    ;;
esac
