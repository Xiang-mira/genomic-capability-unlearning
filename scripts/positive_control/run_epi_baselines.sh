#!/usr/bin/env bash
# Runs epi_baselines.py over every epi_{cell}_{split} produced by
# prepare_epi_splits.py, spread across the nodes of the current SLURM
# allocation (see run_baselines.sh for why -- 1 GPU per node here).
set -uo pipefail

export VB_ROOT="${VB_ROOT:-/scratch/10906/arisk/genomic_unlearning_pc}"
export VB_OUT="${VB_OUT:-$VB_ROOT/results}"
export VB_GUE_DIR="${VB_GUE_DIR:-$VB_OUT/gue_dir}"
LOG_DIR="${PC_LOG_DIR:-$VB_ROOT/logs/epi_baselines}"
RESULT_DIR="$VB_OUT/epi_baselines"

PY="${PC_PYTHON:-/scratch/10906/arisk/biojepa-env/.pixi/envs/default/bin/python3}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t NODES < <(scontrol show hostname "$SLURM_JOB_NODELIST")
N_NODES=${#NODES[@]}
echo "Nodes (${N_NODES}): ${NODES[*]}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$DIR"

i=0
pids=()
for f in "$VB_GUE_DIR"/epi_*__train.csv; do
  task="$(basename "$f" __train.csv)"
  if [[ -f "$RESULT_DIR/${task}__baselines.json" ]]; then
    echo "=== $task already done, skipping ==="
    continue
  fi
  node="${NODES[$(( i % N_NODES ))]}"
  echo "=== $task -> $node (cuda:0 there) ==="
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$node" \
    "$PY" epi_baselines.py --task "$task" --device cuda:0 --seeds 42 43 44 \
    > "$LOG_DIR/${task}.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  if (( i % N_NODES == 0 )); then
    for p in "${pids[@]}"; do wait "$p" || echo "  (job pid $p exited nonzero, see its log)"; done
    pids=()
  fi
done
for p in "${pids[@]}"; do wait "$p" || echo "  (job pid $p exited nonzero, see its log)"; done
echo "All done. Logs in $LOG_DIR"
