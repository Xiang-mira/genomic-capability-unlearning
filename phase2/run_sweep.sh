#!/usr/bin/env bash
# Declarative Task 2 hyperparameter sweeps for GD and RMU.
#
# Usage:
#   bash phase2/run_sweep.sh gd_alpha
#   bash phase2/run_sweep.sh gd
#   bash phase2/run_sweep.sh rmu
#   bash phase2/run_sweep.sh all
#   bash phase2/run_sweep.sh summary
#
# Backward-compatible aliases are kept:
#   A=gd_lr, B=gd_steps, C=rmu_steer+rmu_controls, D=rmu_direction

set -euo pipefail

# Activate the correct conda environment (requires stripedhyena / Evo dependencies)
# Temporarily disable -u: conda's own activate scripts reference unbound variables
set +u
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate UT-p1
set -u

TUNED_ROOT=data/phase2/checkpoints_tuned
DEVICE=${DEVICE:-cuda:0}
BATCH=${BATCH:-2}
MAX_LEN=${MAX_LEN:-512}
SWEEP_CONFIG=${SWEEP_CONFIG:-phase2/sweep_configs/task2_sweeps.json}

print_summary() {
    python phase2/aggregate_task2_results.py \
        --ckpt-roots "$TUNED_ROOT" \
        --print-table
}

case "${1:-all}" in
    summary) print_summary ;;
    *)
        python phase2/run_task2_sweeps.py \
            --config "$SWEEP_CONFIG" \
            --out-dir "$TUNED_ROOT" \
            "$@"
        print_summary
        ;;
esac
