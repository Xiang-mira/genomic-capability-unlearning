#!/usr/bin/env bash
set -euo pipefail

# Orchestrates Phase 2 around scientific claims, not just benchmark coverage.
# Run target validation first, then probe-vs-SFT, then trade-off/trajectory.

PYTHON=${PYTHON:-${PROJECT_PYTHON:-python}}
DEVICE=${DEVICE:-cuda:0}
OUT_ROOT=${OUT_ROOT:-data/phase2/causal_chain}
CKPT_ROOT=${CKPT_ROOT:-data/phase2/checkpoints_tuned}
BENCHMARK_MANIFEST=${BENCHMARK_MANIFEST:-data/benchmarks/hvue_gue_manifest.csv}
HOST_MANIFEST=${HOST_MANIFEST:-data/host_tropism/manifest.csv}
HIYATA_MANIFEST=${HIYATA_MANIFEST:-data/host_tropism_hiyata/manifest.csv}
TARGET_DATASET=${TARGET_DATASET:-local}
LAYERS=${LAYERS:-3-9}
SEEDS=${SEEDS:-42,43,44}
CONTROLLED_SPLIT=${CONTROLLED_SPLIT:-taxonomy}
if [[ "$TARGET_DATASET" == "hiyata" ]]; then
  CONTROLLED_GROUP_KEY=${CONTROLLED_GROUP_KEY:-family}
else
  CONTROLLED_GROUP_KEY=${CONTROLLED_GROUP_KEY:-virus_tax_id}
fi

CHECKPOINTS="gd_localized_ar5_s1000=$CKPT_ROOT/gd_localized_ar5_s1000/weights.safetensors,gd_random_ar5=$CKPT_ROOT/gd_random_ar5/weights.safetensors,rmu_localized_sc50_l4=$CKPT_ROOT/rmu_localized_sc50_l4/weights.safetensors,rmu_random_sc50=$CKPT_ROOT/rmu_random_sc50/weights.safetensors"

target_split() {
  local split_mode="$1"
  local group_key="${2:-$CONTROLLED_GROUP_KEY}"
  local manifest="$HOST_MANIFEST"
  local dataset_tag="local"
  if [[ "$TARGET_DATASET" == "hiyata" ]]; then
    manifest="$HIYATA_MANIFEST"
    dataset_tag="hiyata"
  fi
  if [[ ! -s "$manifest" ]]; then
    echo "[causal-chain] missing host-tropism manifest: $manifest" >&2
    if [[ "$TARGET_DATASET" == "hiyata" ]]; then
      echo "[causal-chain] run: $0 prepare-hiyata" >&2
    fi
    exit 1
  fi
  local out_dir="$OUT_ROOT/host_tropism_${dataset_tag}_base/${split_mode}_${group_key}"
  "$PYTHON" -u phase2/eval_taxonomy_heldout.py \
    --dataset host_tropism \
    --manifest "$manifest" \
    --split-mode "$split_mode" \
    --group-key "$group_key" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --layers "$LAYERS" \
    --batch-size 0 \
    --auto-batch-size 64 \
    --cpu-threads 16 \
    --probe-jobs 7 \
    --progress-every 2048
}

case "${1:-help}" in
  prepare-hiyata)
    "$PYTHON" -u phase2/prepare_hiyata_host_tropism.py \
      --out "$HIYATA_MANIFEST" \
      --summary-out "$(dirname "$HIYATA_MANIFEST")/summary.json"
    ;;
  prepare-hiyata-high-confidence)
    "$PYTHON" -u phase2/prepare_hiyata_host_tropism.py \
      --exclude-gemini \
      --out "${HIYATA_MANIFEST%.csv}_no_gemini.csv" \
      --summary-out "$(dirname "$HIYATA_MANIFEST")/summary_no_gemini.json"
    ;;
  target-random)
    target_split random "$CONTROLLED_GROUP_KEY"
    ;;
  target-taxonomy)
    target_split taxonomy "$CONTROLLED_GROUP_KEY"
    ;;
  target-family)
    TARGET_DATASET=hiyata
    target_split taxonomy family
    ;;
  target-genus)
    TARGET_DATASET=hiyata
    target_split taxonomy genus
    ;;
  target-within-family)
    TARGET_DATASET=hiyata
    target_split within_group family
    ;;
  target-within-genus)
    TARGET_DATASET=hiyata
    target_split within_group genus
    ;;
  target-homology)
    target_split homology "$CONTROLLED_GROUP_KEY"
    ;;
  target-within-group)
    target_split within_group "$CONTROLLED_GROUP_KEY"
    ;;
  target-all)
    target_split random "$CONTROLLED_GROUP_KEY"
    target_split taxonomy "$CONTROLLED_GROUP_KEY"
    target_split homology "$CONTROLLED_GROUP_KEY"
    target_split within_group "$CONTROLLED_GROUP_KEY"
    ;;
  probe-vs-sft)
    dataset_tag="local"
    if [[ "$TARGET_DATASET" == "hiyata" ]]; then
      dataset_tag="hiyata"
    fi
    controlled_manifest="$OUT_ROOT/host_tropism_${dataset_tag}_base/${CONTROLLED_SPLIT}_${CONTROLLED_GROUP_KEY}/controlled_split_manifest.csv"
    if [[ ! -s "$controlled_manifest" ]]; then
      echo "[causal-chain] missing controlled split manifest: $controlled_manifest" >&2
      echo "[causal-chain] run target-${CONTROLLED_SPLIT} first, or set CONTROLLED_SPLIT/CONTROLLED_GROUP_KEY." >&2
      exit 1
    fi
    "$PYTHON" -u phase2/probe_vs_sft.py \
      --benchmark-manifest "$BENCHMARK_MANIFEST" \
      --controlled-split-csv "$controlled_manifest" \
      --controlled-task-name "host_tropism_${dataset_tag}_${CONTROLLED_SPLIT}_${CONTROLLED_GROUP_KEY}" \
      --tasks hvue_human_host_tropism,hvue_human_virus_pathogenicity_cini,gue_prom_300_all,virobench_all_taxon_genus \
      --checkpoints "$CHECKPOINTS" \
      --seeds "$SEEDS" \
      --out-dir "$OUT_ROOT/probe_vs_sft" \
      --feature-cache-dir "$OUT_ROOT/probe_vs_sft_feature_cache" \
      --device "$DEVICE" \
      --layers "$LAYERS"
    ;;
  aggregate-controlled)
    "$PYTHON" phase2/summarize_controlled_splits.py \
      --summary-globs "$OUT_ROOT/host_tropism_*_base/*/*summary.json" \
      --out "$OUT_ROOT/host_tropism_controlled_split_results.csv"
    ;;
  aggregate-trajectory)
    "$PYTHON" phase2/aggregate_trajectory.py \
      --checkpoint-roots "$CKPT_ROOT" \
      --out-dir "$OUT_ROOT/trajectory"
    "$PYTHON" phase2/plot_convergence_diagnostics.py \
      --checkpoint-roots "$CKPT_ROOT" \
      --out-dir "$OUT_ROOT/trajectory"
    ;;
  status)
    "$PYTHON" phase2/summarize_causal_chain_evidence.py \
      --root "$OUT_ROOT" \
      --out-json "$OUT_ROOT/causal_chain_evidence_status.json" \
      --out-md "$OUT_ROOT/causal_chain_evidence_status.md"
    ;;
  help|*)
    cat <<EOF
Usage: $0 <command>

Priority order:
  prepare-hiyata         Download/convert hiyata/Virus-Host-Genomes manifest
  prepare-hiyata-high-confidence
                         Convert hiyata manifest excluding gemini_annotated rows
  target-random          Baseline host-tropism decodability
  target-taxonomy        Hold out selected taxonomy key (default: virus_tax_id)
  target-family          Hiyata family-held-out target validation
  target-genus           Hiyata genus-held-out target validation
  target-homology        Hold out approximate k-mer sequence clusters
  target-within-group    Test mixed-label within-group prediction
  target-within-family   Hiyata within-family host prediction
  target-within-genus    Hiyata within-genus host prediction
  target-all             Run all Base target-validation splits
  probe-vs-sft           Compare frozen probes with SFT using same splits
  aggregate-controlled   Summarize target-validation split outputs
  aggregate-trajectory   Aggregate existing step checkpoint trajectories
  status                 Summarize which causal-chain claims have evidence

Environment overrides:
  PYTHON=$PYTHON
  DEVICE=$DEVICE
  OUT_ROOT=$OUT_ROOT
  CKPT_ROOT=$CKPT_ROOT
  TARGET_DATASET=$TARGET_DATASET
  HOST_MANIFEST=$HOST_MANIFEST
  HIYATA_MANIFEST=$HIYATA_MANIFEST
  CONTROLLED_SPLIT=$CONTROLLED_SPLIT
  CONTROLLED_GROUP_KEY=$CONTROLLED_GROUP_KEY
EOF
    ;;
esac
