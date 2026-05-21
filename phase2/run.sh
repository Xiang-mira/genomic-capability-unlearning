#!/usr/bin/env bash
# Phase 2 orchestration: splits -> unlearning runs -> per-run evaluation.
# Usage:
#   bash phase2/run.sh splits   # one-time: build forget/retain CSVs
#   bash phase2/run.sh gd       # all four GD conditions (full, localized, probe, random)
#   bash phase2/run.sh rmu      # all three RMU conditions
#   bash phase2/run.sh eval     # eval every checkpoint
#   bash phase2/run.sh all      # everything in sequence

set -euo pipefail

CKPT_ROOT=data/phase2/checkpoints
STEPS=${STEPS:-200}
LR=${LR:-1e-5}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}

run_gd() {
    for cond in localized probe random full; do
        echo "=== GD $cond ==="
        python phase2/unlearn_gd.py \
            --condition "$cond" \
            --steps "$STEPS" --lr "$LR" \
            --batch-size "$BATCH" --max-length "$MAX_LEN"
    done
}

run_rmu() {
    for cond in localized random full; do
        echo "=== RMU $cond ==="
        python phase2/unlearn_rmu.py \
            --condition "$cond" \
            --steps "$STEPS" --lr "$LR" \
            --batch-size "$BATCH" --max-length "$MAX_LEN"
    done
}

run_eval() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        if [ -f "$ckpt" ]; then
            echo "=== eval $run ==="
            python phase2/eval_unlearn.py --ckpt "$ckpt"
        fi
    done
}

case "${1:-all}" in
    splits) python phase2/build_unlearn_splits.py ;;
    gd) run_gd ;;
    rmu) run_rmu ;;
    eval) run_eval ;;
    all)
        python phase2/build_unlearn_splits.py
        run_gd
        run_rmu
        run_eval
        ;;
    *)
        echo "Unknown target: $1"
        echo "Usage: bash phase2/run.sh [splits|gd|rmu|eval|all]"
        exit 1
        ;;
esac
