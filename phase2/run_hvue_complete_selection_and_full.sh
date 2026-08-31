#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PHASE2_PYTHON:-${PROJECT_PYTHON:-python}}"

# Candidate selection now follows the route switch: only the formal target
# tasks participate in ranking. Confounded transmissibility tasks remain
# available for separate control-only diagnostics, not for selection.
SELECTION_TASKS="hvue_human_host_tropism,hvue_human_virus_pathogenicity_cini,gue_human_tf_0"

"$PYTHON_BIN" -u phase2/run_benchmark_pilot.py pilot \
  --pilot-manifest data/benchmarks/hvue_gue_pilot_manifest.csv \
  --benchmark-scope task \
  --task-filter "$SELECTION_TASKS" \
  --pilot-root data/phase2/benchmark_pilot_lora \
  --ckpt-root data/phase2/checkpoints_lora_grid \
  --discover-candidates \
  --top-k 12 \
  --device cuda:0 \
  --batch-size 1 \
  --cpu-threads 16 \
  --epochs 3 \
  --eval-every 100 \
  --patience 3 \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --metric-for-best auto \
  --max-length 512 \
  --seed 42

exec bash phase2/run_selected_full_benchmark.sh
