#!/usr/bin/env bash
# Dispatches virobench_glm.py jobs across the nodes of the current SLURM
# allocation (1 GPU per node on Vista, see run_baselines.sh in
# scripts/positive_control/ for why). Skips (model,regime,seed,lr) combos
# whose result JSON already exists -- safe here since each file is a single
# seed's result, not an aggregate (unlike the gue_baselines.py caching bug).
set -uo pipefail

export VB_ROOT="${VB_ROOT:-/scratch/10906/arisk/genomic_unlearning_pc}"
export VB_OUT="${VB_OUT:-$VB_ROOT/results}"
export VB_VIRO_DIR="${VB_VIRO_DIR:-$VB_ROOT/results/virobench}"
export HF_HOME="${HF_HOME:-/scratch/10906/arisk/hf_cache}"
LOG_DIR="${PC_LOG_DIR:-$VB_ROOT/logs/virobench_glm}"

PY="${PC_PYTHON:-/scratch/10906/arisk/virobench-glm-env/.pixi/envs/default/bin/python3}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t NODES < <(scontrol show hostname "$SLURM_JOB_NODELIST")
N_NODES=${#NODES[@]}
echo "Nodes (${N_NODES}): ${NODES[*]}"

mkdir -p "$LOG_DIR"
cd "$DIR"

# model bs seed
JOBS=(
  "nt_v2_500m full 2 42"
  "nt_v2_500m probe 4 42"
  "gena_lm full 8 42"
  "gena_lm probe 8 42"
  "hyenadna full 4 42"
  "hyenadna probe 8 42"
  "nt_v2_500m full 2 43"
  "nt_v2_500m full 2 44"
)

i=0
pids=()
for job in "${JOBS[@]}"; do
  read -r model regime bs seed <<< "$job"
  node="${NODES[$(( i % N_NODES ))]}"
  tag="${model}__${regime}__DNA_times_family__s${seed}"
  echo "=== $tag -> $node (cuda:0 there) ==="
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$node" \
    env PYTHONNOUSERSITE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PY" virobench_glm.py --model "$model" --mod DNA --split times --level family \
    --regime "$regime" --seeds "$seed" --bs "$bs" --cap 999999 --device cuda:0 \
    > "$LOG_DIR/${tag}.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  if (( i % N_NODES == 0 )); then
    for p in "${pids[@]}"; do wait "$p" || echo "  (job pid $p exited nonzero, see its log)"; done
    pids=()
  fi
done
for p in "${pids[@]}"; do wait "$p" || echo "  (job pid $p exited nonzero, see its log)"; done
echo "All done. Logs in $LOG_DIR"
