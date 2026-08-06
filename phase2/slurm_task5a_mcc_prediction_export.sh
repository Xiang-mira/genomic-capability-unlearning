#!/usr/bin/env bash
#SBATCH --job-name=task5a_mcc_export
#SBATCH --output=logs/task5a_mcc_export_%j.out
#SBATCH --error=logs/task5a_mcc_export_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

python phase2/run_task5a_identity_reaudit.py \
  --out-root data/phase2/audits/task5a_identity_reaudit_20260713_mcc_export \
  --resume \
  --export-predictions

python phase2/mcc_audit.py \
  --out-dir data/phase2/audits/mcc_audit_20260720_with_task5a \
  --task5a-dir data/phase2/audits/task5a_identity_reaudit_20260713_mcc_export \
  --prediction-shards 'data/phase2/audits/task5a_identity_reaudit_20260713_mcc_export/*/eval_predictions.csv' \
  --bootstrap 500
