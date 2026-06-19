#!/usr/bin/env bash
set -euo pipefail

PHASE1_PYTHON=${PHASE1_PYTHON:-python}
TARGET_FAMILY=${TARGET_FAMILY:-Coronaviridae}
TARGET_ROOT_PREFIX=${TARGET_ROOT_PREFIX:-data/family_targets/coronaviridae_seed}
REFSEQ_RAW_DIR=${REFSEQ_RAW_DIR:-data/refseq_family}
SEEDS=${SEEDS:-42,43,44}
MAX_LEN=${MAX_LEN:-512}
WINDOWS_PER_SEQUENCE=${WINDOWS_PER_SEQUENCE:-32}
MAX_POSITIVE_RECORDS=${MAX_POSITIVE_RECORDS:-0}
NEGATIVE_POOL_MULTIPLIER=${NEGATIVE_POOL_MULTIPLIER:-1.5}
MAX_NEGATIVE_RECORDS=${MAX_NEGATIVE_RECORDS:-0}
MIN_VAL_PER_LABEL=${MIN_VAL_PER_LABEL:-200}
MIN_TEST_PER_LABEL=${MIN_TEST_PER_LABEL:-200}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}

IFS=',' read -r -a seed_array <<< "$SEEDS"
for seed in "${seed_array[@]}"; do
    seed="$(echo "$seed" | xargs)"
    out_dir="${TARGET_ROOT_PREFIX}${seed}"
    echo "[phase1] building RefSeq manifest seed=$seed out_dir=$out_dir"
    "$PHASE1_PYTHON" phase1/build_refseq_family_target_dataset.py \
        --out-dir "$out_dir" \
        --raw-dir "$REFSEQ_RAW_DIR" \
        --target-family "$TARGET_FAMILY" \
        --max-length "$MAX_LEN" \
        --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
        --max-positive-records "$MAX_POSITIVE_RECORDS" \
        --negative-pool-multiplier "$NEGATIVE_POOL_MULTIPLIER" \
        --max-negative-records "$MAX_NEGATIVE_RECORDS" \
        --min-val-per-label "$MIN_VAL_PER_LABEL" \
        --min-test-per-label "$MIN_TEST_PER_LABEL" \
        --seed "$seed" \
        --download-workers "$DOWNLOAD_WORKERS"
done
