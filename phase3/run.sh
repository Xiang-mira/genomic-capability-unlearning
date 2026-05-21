#!/usr/bin/env bash
# Phase 3 orchestration: run SFT and LoRA attacks on all 6 unlearned checkpoints.
# Usage:
#   bash phase3/run.sh sft    # SFT attack on all checkpoints (sequential, GPU memory)
#   bash phase3/run.sh lora   # LoRA attack on all checkpoints
#   bash phase3/run.sh all    # both attacks

set -euo pipefail

CKPT_ROOT=data/phase2/checkpoints
STEPS=${STEPS:-200}
LR_SFT=${LR_SFT:-1e-5}
LR_LORA=${LR_LORA:-1e-4}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}

run_sft() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        [ -f "$ckpt" ] || continue
        echo "=== SFT $(basename $run) ==="
        python phase3/attack_sft.py \
            --ckpt "$ckpt" \
            --steps "$STEPS" --lr "$LR_SFT" \
            --batch-size "$BATCH" --max-length "$MAX_LEN"
    done
}

run_lora() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        [ -f "$ckpt" ] || continue
        echo "=== LoRA $(basename $run) ==="
        python phase3/attack_lora.py \
            --ckpt "$ckpt" \
            --steps "$STEPS" --lr "$LR_LORA" \
            --batch-size "$BATCH" --max-length "$MAX_LEN"
    done
}

case "${1:-all}" in
    sft)  run_sft ;;
    lora) run_lora ;;
    all)  run_sft; run_lora ;;
    *)
        echo "Usage: bash phase3/run.sh [sft|lora|all]"
        exit 1 ;;
esac
