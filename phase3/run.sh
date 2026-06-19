#!/usr/bin/env bash
# Phase 3 orchestration: run SFT and LoRA attacks on all 6 unlearned checkpoints.
# Usage:
#   bash phase3/run.sh sft    # SFT attack on all checkpoints (sequential, GPU memory)
#   bash phase3/run.sh lora   # LoRA attack on all checkpoints
#   bash phase3/run.sh all    # both attacks

set -euo pipefail

CKPT_ROOT=data/phase2/checkpoints
STEPS=${STEPS:-2000}
LR_SFT_GRID=${LR_SFT_GRID:-5e-6 1e-5 2e-5}
LR_LORA_GRID=${LR_LORA_GRID:-5e-5 1e-4 2e-4}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}
TARGET_MANIFEST=${TARGET_MANIFEST:-data/family_targets/coronaviridae/manifest.csv}
TARGET_PROBE_DIR=${TARGET_PROBE_DIR:-data/family_targets/coronaviridae/probes}
LOCALIZED_LAYERS_PATH=${LOCALIZED_LAYERS_PATH:-data/family_targets/coronaviridae/localized_layers.json}

run_sft() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        [ -f "$ckpt" ] || continue
        for lr in $LR_SFT_GRID; do
            echo "=== SFT $(basename $run) lr=$lr ==="
            python phase3/attack_sft.py \
                --ckpt "$ckpt" \
                --manifest "$TARGET_MANIFEST" \
                --probe-dir "$TARGET_PROBE_DIR" \
                --steps "$STEPS" --lr "$lr" \
                --batch-size "$BATCH" --max-length "$MAX_LEN" \
                --run-name "$(basename "$run")_sft_lr${lr}"
        done
    done
}

run_lora() {
    for run in "$CKPT_ROOT"/*/; do
        ckpt="$run/weights.safetensors"
        [ -f "$ckpt" ] || continue
        for lr in $LR_LORA_GRID; do
            echo "=== LoRA $(basename $run) lr=$lr ==="
            python phase3/attack_lora.py \
                --ckpt "$ckpt" \
                --manifest "$TARGET_MANIFEST" \
                --probe-dir "$TARGET_PROBE_DIR" \
                --localized-layers-path "$LOCALIZED_LAYERS_PATH" \
                --steps "$STEPS" --lr "$lr" \
                --batch-size "$BATCH" --max-length "$MAX_LEN" \
                --run-name "$(basename "$run")_lora_lr${lr}"
        done
    done
}

run_matrix() {
    python phase3/aggregate_attack_results.py --localized-layers-path "$LOCALIZED_LAYERS_PATH"
}

case "${1:-all}" in
    sft)  run_sft ;;
    lora) run_lora ;;
    matrix) run_matrix ;;
    all)  run_sft; run_lora; run_matrix ;;
    *)
        echo "Usage: bash phase3/run.sh [sft|lora|matrix|all]"
        exit 1 ;;
esac
