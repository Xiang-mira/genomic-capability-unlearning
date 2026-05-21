#!/usr/bin/env bash
set -e

# Optional bacteria-negative rerun for Phase 1.
# This script reruns the same localization pipeline using bacterial chromosome
# sequences as the non-viral class.
#
# This is separate from the reported viral-vs-plasmid pilot result.
# Output is written to data/phase1_bacteria to avoid overwriting the plasmid pilot.

# 1) Download RefSeq sequences and build viral-vs-bacteria manifest
python phase1/download_refseq.py \
  --out-dir data/phase1_bacteria \
  --target-per-class 2000 \
  --max-length 2048 \
  --viral-max-files 4 \
  --nonviral-max-files 4 \
  --nonviral-group bacteria \
  --download-workers 4 \
  --use-aria2c \
  --aria2c-connections 8 \
  --seed 42

# 2) Extract layer-wise Evo-1 activations
python phase1/extract_features.py \
  --manifest data/phase1_bacteria/manifest.csv \
  --model-dir ./evo-1-8k-base \
  --config-path configs/evo-1-8k-base_inference.yml \
  --out-dir data/phase1_bacteria/features \
  --batch-size 8 \
  --max-length 2048

# 3) Train layer-wise L2 logistic probes
python phase1/train_probes.py \
  --feature-dir data/phase1_bacteria/features \
  --out-dir data/phase1_bacteria/probes \
  --c-grid 0.1,1,10

# 4) Plot layer-wise validation/test AUROC curve
python phase1/plot_metrics.py \
  --metrics data/phase1_bacteria/probes/probe_metrics_by_layer.csv \
  --out-dir data/phase1_bacteria/probes

# 5) Run GC + 1-gram baseline
python phase1/baseline_gc_1gram.py \
  --manifest data/phase1_bacteria/manifest.csv \
  --out data/phase1_bacteria/baselines/gc_1gram_metrics.csv
