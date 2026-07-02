#!/usr/bin/env bash
set -euo pipefail

cd /home/teacher1/UT-project1/project1

while [[ ! -f data/phase2/checkpoints_rmu_tuning/rmu_full_l6_base/eval_representation.csv ||
         ! -f data/phase2/checkpoints_rmu_tuning/rmu_full_l8_base/eval_representation.csv ]]; do
    sleep 30
done

/home/teacher1/miniconda3/envs/UT-p1/bin/python -u phase2/run_task2_sweeps.py tune \
    --config phase2/sweep_configs/rmu_l6_l8_tuning.json \
    --out-dir data/phase2/checkpoints_rmu_tuning \
    --device cuda:0 \
    --batch-size 2 \
    --eval-batch-size 4 \
    --max-length 512 \
    --run-internal-eval \
    --internal-layers 0-31 \
    --delete-checkpoint-after-internal-eval \
    --progress-path data/phase2/checkpoints_rmu_tuning/tuning_progress.json \
    --resume
