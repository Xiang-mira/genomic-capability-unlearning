#!/usr/bin/env bash
set -e

# Phase 1 pilot pipeline for Evo-1 viral-family representation localization.
# Current reported results are based on a viral-vs-plasmid pilot dataset:
# 2,000 viral sequences + 2,000 plasmid sequences, 2,048 bp windows, seed = 42.

# 1) Download RefSeq sequences and build viral-vs-plasmid manifest
python phase1/download_refseq.py \
  --out-dir data/phase1 \
  --target-per-class 2000 \
  --max-length 2048 \
  --viral-max-files 4 \
  --nonviral-max-files 4 \
  --nonviral-group plasmid \
  --download-workers 4 \
  --use-aria2c \
  --aria2c-connections 8 \
  --seed 42

# 2) Extract layer-wise Evo-1 activations
python phase1/extract_features.py \
  --manifest data/phase1/manifest.csv \
  --model-dir ./evo-1-8k-base \
  --config-path configs/evo-1-8k-base_inference.yml \
  --out-dir data/phase1/features \
  --batch-size 8 \
  --max-length 2048

# 3) Train layer-wise L2 logistic probes and write probe metrics
python phase1/train_probes.py \
  --feature-dir data/phase1/features \
  --out-dir data/phase1/probes \
  --c-grid 0.1,1,10

# 4) Plot layer-wise validation/test AUROC curve
python phase1/plot_metrics.py \
  --metrics data/phase1/probes/probe_metrics_by_layer.csv \
  --out-dir data/phase1/probes

# 5) Run GC + 1-gram baseline
python phase1/baseline_gc_1gram.py \
  --manifest data/phase1/manifest.csv \
  --out data/phase1/baselines/gc_1gram_metrics.csv
