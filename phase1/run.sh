#!/usr/bin/env bash
set -euo pipefail

PHASE1_PYTHON=${PHASE1_PYTHON:-python}
TARGET_ROOT=${TARGET_ROOT:-data/family_targets/coronaviridae}
TARGET_MANIFEST=${TARGET_MANIFEST:-$TARGET_ROOT/manifest.csv}
FEATURE_DIR=${FEATURE_DIR:-$TARGET_ROOT/features}
PROBE_DIR=${PROBE_DIR:-$TARGET_ROOT/probes}
PATCH_DIR=${PATCH_DIR:-$TARGET_ROOT/activation_patching}
LOCALIZED_PATH=${LOCALIZED_PATH:-$TARGET_ROOT/localized_layers.json}
BATCH=${BATCH:-80}
MAX_LEN=${MAX_LEN:-512}
SEED=${SEED:-42}
DATA_SOURCE=${DATA_SOURCE:-refseq}
TARGET_FAMILY=${TARGET_FAMILY:-Coronaviridae}
REFSEQ_RAW_DIR=${REFSEQ_RAW_DIR:-data/refseq_family}
WINDOWS_PER_SEQUENCE=${WINDOWS_PER_SEQUENCE:-4}
MAX_POSITIVE_RECORDS=${MAX_POSITIVE_RECORDS:-0}
NEGATIVE_POOL_MULTIPLIER=${NEGATIVE_POOL_MULTIPLIER:-2.5}
MAX_NEGATIVE_RECORDS=${MAX_NEGATIVE_RECORDS:-0}
MIN_VAL_PER_LABEL=${MIN_VAL_PER_LABEL:-0}
MIN_TEST_PER_LABEL=${MIN_TEST_PER_LABEL:-0}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
PATCH_LAYERS=${PATCH_LAYERS:-all}
STABLE_LAYERS=${STABLE_LAYERS:-$PATCH_LAYERS}

build_manifest() {
    if [[ "$DATA_SOURCE" == "refseq" ]]; then
        "$PHASE1_PYTHON" phase1/build_refseq_family_target_dataset.py \
            --out-dir "$TARGET_ROOT" \
            --raw-dir "$REFSEQ_RAW_DIR" \
            --target-family "$TARGET_FAMILY" \
            --max-length "$MAX_LEN" \
            --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
            --max-positive-records "$MAX_POSITIVE_RECORDS" \
            --negative-pool-multiplier "$NEGATIVE_POOL_MULTIPLIER" \
            --max-negative-records "$MAX_NEGATIVE_RECORDS" \
            --min-val-per-label "$MIN_VAL_PER_LABEL" \
            --min-test-per-label "$MIN_TEST_PER_LABEL" \
            --seed "$SEED" \
            --download-workers "$DOWNLOAD_WORKERS"
    elif [[ "$DATA_SOURCE" == "virobench" ]]; then
        "$PHASE1_PYTHON" phase1/build_family_target_dataset.py \
            --out-dir "$TARGET_ROOT" \
            --target-family "$TARGET_FAMILY" \
            --max-length "$MAX_LEN" \
            --seed "$SEED"
    else
        echo "Unsupported DATA_SOURCE=$DATA_SOURCE (expected refseq or virobench)"
        exit 1
    fi
}

run_baselines() {
    "$PHASE1_PYTHON" phase1/baseline_gc_1gram.py --manifest "$TARGET_MANIFEST" --out-dir "$TARGET_ROOT/baselines" --feature gc_1gram_length
    "$PHASE1_PYTHON" phase1/baseline_gc_1gram.py --manifest "$TARGET_MANIFEST" --out-dir "$TARGET_ROOT/baselines" --feature kmer --kmer-max 4 --kmer-binary --max-iter 1000
}

run_probes() {
    "$PHASE1_PYTHON" phase1/extract_features.py \
        --manifest "$TARGET_MANIFEST" \
        --out-dir "$FEATURE_DIR" \
        --batch-size "$BATCH" \
        --max-length "$MAX_LEN" \
        --representation next_norm
    "$PHASE1_PYTHON" phase1/diagnose_features.py --feature-dir "$FEATURE_DIR" --out "$FEATURE_DIR/feature_diagnostics.csv"
    "$PHASE1_PYTHON" phase1/train_probes.py --feature-dir "$FEATURE_DIR" --out-dir "$PROBE_DIR" --c-grid 0.001,0.01,0.1,1 --max-iter 1000
    "$PHASE1_PYTHON" phase1/plot_metrics.py --metrics "$PROBE_DIR/probe_metrics_by_layer.csv" --out-dir "$PROBE_DIR"
}

run_patching() {
    "$PHASE1_PYTHON" phase1/activation_patching.py \
        --manifest "$TARGET_MANIFEST" \
        --probe-dir "$PROBE_DIR" \
        --out-dir "$PATCH_DIR" \
        --split test \
        --n-pairs 16 \
        --max-length "$MAX_LEN" \
        --layers "$PATCH_LAYERS" \
        --directions both
    "$PHASE1_PYTHON" phase1/select_localized_layers.py --summary-csv "$PATCH_DIR/patching_layer_summary.csv" --out "$LOCALIZED_PATH" --stable-layers "$STABLE_LAYERS"
    "$PHASE1_PYTHON" phase1/plot_patching.py --probe-csv "$PROBE_DIR/probe_metrics_by_layer.csv" --patching-csv "$PATCH_DIR/patching_by_layer.csv" --out-dir "$PATCH_DIR" --localized-layers-path "$LOCALIZED_PATH"
}

case "${1:-all}" in
    manifest) build_manifest ;;
    baselines) run_baselines ;;
    probes) run_probes ;;
    patching) run_patching ;;
    all)
        build_manifest
        run_baselines
        run_probes
        run_patching
        ;;
    *)
        echo "Usage: bash phase1/run.sh [manifest|baselines|probes|patching|all]"
        exit 1
        ;;
esac
